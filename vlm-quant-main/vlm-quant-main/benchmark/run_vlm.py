import argparse
import csv
import datetime
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


def benchmark():
    
    output_filename = f"vlm_benchmark_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    
    models_to_test = {
        "origin": "/bigdata/models/Qwen2.5-VL-7B-Instruct",
        #"nvfp4": "/root/wja/project/github/VLM-Quant/saved_models/vlm/quantized_models/Qwen2.5-VL-7B-Instruct-nvfp-w4-a4-RTN-identity-transform",
        # "mxfp4": "/root/wja/project/github/VLM-Quant/saved_models/vlm/quantized_models/Qwen2.5-VL-7B-Instruct-mxfp-w4-a4-RTN-identity-transform"
    }

    batch_sizes = [1, 8, 64]
    # sequences = [(32, 128), (128, 32), (32, 32), (128, 128)]
    # sequences = [(32, 1024), (1024, 32), (1024, 1024), (32, 32)]
    sequences = [(32, 1024), (1024, 32)]

    with open(output_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow([
            "model_name", "batch_size", "prefill_len", "decode_len",
            "prefill_time_ms_mean", "prefill_time_ms_std", "prefill_mem_gb_mean", "prefill_mem_gb_std",
            "decode_time_ms_mean", "decode_time_ms_std", "decode_mem_gb_mean", "decode_mem_gb_std",
            "error"
        ])

        for model_name, model_path in models_to_test.items():
            for bsz in batch_sizes:
                for prefill_len, decode_len in sequences:
                    
                    print("-" * 80)
                    print(f"Testing model: {model_name}, batch_size: {bsz}, prefill: {prefill_len}, decode: {decode_len}")

                    result_row = [model_name, bsz, prefill_len, decode_len]
                    
                    try:
                        model = get_model_quantized(model_path, None)
                        (time_prefill, mem_prefill), (time_decode, mem_decode) = run_all_for_model(
                            model, bsz, prefill_len, decode_len, None)
                        
                        del model
                        _cleanup()

                        prefill_time_mean = np.mean(time_prefill)
                        prefill_time_std = 1.96 * np.std(time_prefill)
                        prefill_mem_mean = np.mean(mem_prefill) / (1024**3)
                        prefill_mem_std = 1.96 * np.std(mem_prefill) / (1024**3)
                        
                        print(f"  Prefill time: {prefill_time_mean:.3f} +- {prefill_time_std:.3f}ms")
                        print(f"  Prefill memory: {prefill_mem_mean:.3f} +- {prefill_mem_std:.3f}GB")

                        decode_time_mean, decode_time_std, decode_mem_mean, decode_mem_std = None, None, None, None
                        if time_decode and mem_decode:
                            decode_time_mean = np.mean(time_decode)
                            decode_time_std = 1.96 * np.std(time_decode)
                            decode_mem_mean = np.mean(mem_decode) / (1024**3)
                            decode_mem_std = 1.96 * np.std(mem_decode) / (1024**3)
                            print(f"  Decode time per token: {decode_time_mean:.3f} +- {decode_time_std:.3f}ms")
                            print(f"  Decode memory: {decode_mem_mean:.3f} +- {decode_mem_std:.3f}GB")

                        result_row.extend([
                            prefill_time_mean, prefill_time_std, prefill_mem_mean, prefill_mem_std,
                            decode_time_mean, decode_time_std, decode_mem_mean, decode_mem_std,
                            "" # No error
                        ])

                    except Exception as e:
                        print(f"  ERROR during benchmark: {e}")
                        result_row.extend([None, None, None, None, None, None, None, None, str(e)])
                        _cleanup() # Ensure cleanup even on error
                    
                    csv_writer.writerow(result_row)
    
    print("-" * 80)
    print(f"Benchmark finished. Results saved to {output_filename}")


if __name__ == '__main__':
    benchmark()