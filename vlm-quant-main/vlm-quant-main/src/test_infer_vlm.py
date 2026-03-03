from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, FPQuantConfig
import torch
import argparse
import time

original_model_path = "/bigdata/models/Qwen2.5-VL-7B-Instruct/"  
quant_model_path = "/root/work/vlm-quant-main/vlm-quant-main/saved_models/vlm/quantized_models/-nvfp-w4-a4-RTN-identity-transform"  

tokenizer = AutoTokenizer.from_pretrained(original_model_path)

quantization_config = FPQuantConfig(
    forward_dtype="nvfp4",
    hadamard_group_size=32,
    pseudoquantization=True,  
)

quantized_lm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    quant_model_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
).model.language_model

original_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    original_model_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
).eval()

del original_model.model.language_model
torch.cuda.empty_cache()

original_model.model.language_model = quantized_lm

model = original_model.to("cuda")


import nvtx

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MAX_NEW_TOKENS = 20
PROMPT = "How are you?"
WARMUP_ITERS = 3

def run_prefill(model, input_ids, attention_mask, batch_size):
    # 手动构造 position_ids [batch, seq_len]
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0).repeat(batch_size, 1)

    with nvtx.annotate(f"prefill_bs{batch_size}", color="blue"):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,  # 新增
            use_cache=True,
        )
    return outputs.past_key_values, outputs.logits

def run_decode(model, input_ids, attention_mask, past_key_values, batch_size, step, seq_len_prefill): # 新增参数
    # 计算当前步骤的 position_id
    current_pos = seq_len_prefill + step
    position_ids = torch.tensor([[current_pos]], dtype=torch.long, device=model.device).repeat(batch_size, 1)

    with nvtx.annotate(f"decode_bs{batch_size}_step{step}", color="green"):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids, # 新增
            use_cache=True,
        )
    return outputs.past_key_values, outputs.logits

def generate_with_nvtx(model, tokenizer, prompt, batch_size, max_new_tokens):
    prompts = [prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    seq_len_prefill = input_ids.shape[1]  # 获取 prompt 长度
    
    with nvtx.annotate(f"generation_bs{batch_size}", color="orange"):
        with torch.inference_mode():
            # 1. Prefill
            past_key_values, logits = run_prefill(model, input_ids, attention_mask, batch_size)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = next_token
            
            # 2. Decode Loop
            for step in range(max_new_tokens - 1):
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=model.device, dtype=attention_mask.dtype)
                ], dim=1)
                
                past_key_values, logits = run_decode(
                    model, next_token, attention_mask, past_key_values, batch_size, step, seq_len_prefill # 传入长度
                )
                
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                if (next_token == tokenizer.eos_token_id).all():
                    break
    
    return torch.cat([input_ids, generated_ids], dim=-1)

for bs in BATCH_SIZES:
    print(f"\n{'='*50}")
    print(f"Batch Size: {bs}")
    print(f"{'='*50}")
    
    for _ in range(WARMUP_ITERS):
        with torch.inference_mode():
            _ = generate_with_nvtx(model, tokenizer, PROMPT, bs, MAX_NEW_TOKENS)
    torch.cuda.synchronize()
    
    with nvtx.annotate(f"benchmark_bs{bs}", color="red"):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = generate_with_nvtx(model, tokenizer, PROMPT, bs, MAX_NEW_TOKENS)
        torch.cuda.synchronize()
        end = time.perf_counter()
    
    print(f"Total time: {end - start:.4f} s")
    print(f"Generated text: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")