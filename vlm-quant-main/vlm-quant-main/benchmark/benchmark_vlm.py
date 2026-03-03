import argparse
import gc
import functools
import pprint
import numpy as np
import torch
import time

import torch
import transformers
import dataclasses


num_warmup_steps = 1
num_bench_steps = 1

def repeated_run(num_repeats=10):
    def func(module):
        def _f(*args, **kwargs):
            times = []
            for i in range(num_repeats):
                times.append(module(*args, **kwargs))
            return tuple(zip(*times))
        return _f
    return func

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()

@repeated_run()
def module_benchmark(module):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    torch.cuda.reset_max_memory_allocated()
    
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps, peak_memory



def get_model_quantized(name, model_cfg):
    from transformers import Qwen2_5_VLForConditionalGeneration
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    # "/root/wja/project/github/VLM-Quant/saved_models/vlm/quantized_models/Qwen2.5-VL-7B-Instruct-nvfp-w4-a4-RTN-identity-transform",
    # device_map="cuda",
    # torch_dtype=torch.bfloat16,
    # )
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    # "/root/wja/data/models/Qwen/Qwen2.5-VL-7B-Instruct",
    # device_map="cuda",
    # torch_dtype=torch.bfloat16,
    # )    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    )

    return model



def run_prefill(model, bsz, prefill_length, config):
    device = 'cuda'
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int64, device=device)
    def _prefill():
        model(test_input)
   
    return module_benchmark(_prefill)

def run_decode(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int64, device=device)
    
    # Prefill to get past_key_values
    out = model(test_input, use_cache=True)
    past_key_values = out.past_key_values
    del out
    _cleanup()

    next_input = torch.randint(100, 200, (bsz, 1), dtype=torch.int64, device=device)

    def _decode_for_multiple_steps():
        nonlocal next_input, past_key_values
        pkv_for_run = past_key_values
        for _ in range(decode_steps):
            outputs = model(next_input, use_cache=True, past_key_values=pkv_for_run)
            pkv_for_run = outputs.past_key_values
            # For benchmarking, we can just take argmax.
            # In a real scenario, sampling methods would be used.
            next_input = torch.argmax(outputs.logits[:, -1:, :], dim=-1)

    # We need to divide the total time by decode_steps to get per-token latency
    times, peak_mem = module_benchmark(_decode_for_multiple_steps)
    times_per_step = [t / decode_steps for t in times]
    return times_per_step, peak_mem


@torch.no_grad
def run_all_for_model(model, bsz, prefill, decode, config):
    model = model.cuda()
    model.eval()
    time_prefill, memory_prefill = run_prefill(model, bsz, prefill, config)
    
    _cleanup()
    
    if decode and decode > 0:
        time_decode, memory_decode = run_decode(model, bsz, prefill, decode)
        _cleanup()
        return (time_prefill, memory_prefill), (time_decode, memory_decode)
    
    return (time_prefill, memory_prefill), (None, None)


def benchmark(args):
    times = []
   
    model = get_model_quantized(args.model, None)
    (time_prefill_i4, mem_i4), (time_decode_i4, mem_decode_i4) = run_all_for_model(
        model, args.batch_size, args.prefill_seq_len, args.decode_steps, None)
    del model
    _cleanup()

    print(f"Prefill time: {np.mean(time_prefill_i4):.3f} +- {1.96 * np.std(time_prefill_i4):.3f}ms")
    print(f"Prefill memory: {np.mean(mem_i4) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_i4) / (1024 * 1024 * 1024):.3f}GB")
    print('--------------')

    if time_decode_i4 and mem_decode_i4:
        print(f"Decode time per token: {np.mean(time_decode_i4):.3f} +- {1.96 * np.std(time_decode_i4):.3f}ms")
        print(f"Decode memory: {np.mean(mem_decode_i4) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_decode_i4) / (1024 * 1024 * 1024):.3f}GB")
        print('--------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    origin = "/root/wja/data/models/Qwen/Qwen2.5-VL-7B-Instruct"
    nvfp4 = "/root/wja/project/github/VLM-Quant/saved_models/vlm/quantized_models/Qwen2.5-VL-7B-Instruct-nvfp-w4-a4-RTN-identity-transform"
    mxfp4 = "/root/wja/project/github/VLM-Quant/saved_models/vlm/quantized_models/Qwen2.5-VL-7B-Instruct-mxfp-w4-a4-RTN-identity-transform"

    parser.add_argument(
        '--model', type=str,
        default='/root/wja/data/models/Qwen/Qwen2.5-VL-7B-Instruct'
    )
    
    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=64,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=32,
    )
    parser.add_argument(
        '--decode_steps', type=int,
        help='Decode steps',
        required=False,
        default=128,
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    benchmark(args)