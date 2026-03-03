import argparse
import gc
import pprint
import sys
import time
from pathlib import Path

import numpy as np
import torch
from collections.abc import Mapping
from typing import Any, Tuple, TypeVar, cast, Optional

from torch import nn
import dataclasses
from transformers.cache_utils import DynamicCache

from accelerate import infer_auto_device_map,dispatch_model
from accelerate.utils import get_balanced_memory
import copy

# REPO_ROOT = Path(__file__).resolve().parents[1]
# SRC_ROOT = REPO_ROOT / "src"
# if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
#     sys.path.append(str(SRC_ROOT))

# from fp_quant.utils.model_utils import InputCollector, ForwardInterrupt  # type: ignore
# from utils.model_utils import InputCollector, ForwardInterrupt  # type: ignore

def to(data: Any, *args, **kwargs):
    """
    # adopted from https://github.com/Yura52/delu/blob/main/delu/_tensor_ops.py
    TODO
    """

    def _to(x):
        return to(x, *args, **kwargs)

    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list, set)):
        return type(data)(_to(x) for x in data)
    elif isinstance(data, dict):
        return type(data)((k, _to(v)) for k, v in data.items())
    elif dataclasses.is_dataclass(data):
        return type(data)(**{k: _to(v) for k, v in vars(data).items()})
    # do nothing if provided value is not tensor or collection of tensors
    else:
        return data

class InputCollector(nn.Module):

    def __init__(self, module: nn.Module, cpu_offload: bool = False):
        super().__init__()
        self.module = module
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []
        self.attention_type = getattr(module, "attention_type", None)

    def forward(self, *input_args, **input_kwargs):
        """
        Assumes that the wrapped module has a single
        input that can reside in inputs or input_kwargs.
        """
        if self.cpu_offload:
            input_args = to(input_args, device="cpu")
            input_kwargs = to(input_kwargs, device="cpu")
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt


class ForwardInterrupt(Exception):
    pass

num_warmup_steps = 2
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@repeated_run()
def module_benchmark(module):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    if torch.cuda.is_available():
        _cleanup()
        torch.cuda.synchronize()
        torch.cuda.reset_max_memory_allocated()
    

    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = module()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        peak_memory = 0

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps, peak_memory



def get_model_quantized(name, model_cfg):
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        attn_implementation="flash_attention_2",
    )

    return model



def get_language_model_layers(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Unable to locate transformer layers on the provided model.")


def prepare_sample_input(batch_size, seq_len, device):
    return torch.randint(100, 200, (batch_size, seq_len), dtype=torch.int64, device=device)


def collect_layer_inputs(model, layer_idx, *model_args, **model_kwargs) -> Tuple[torch.nn.Module, Tuple[Any, ...], dict[str, Any]]:
    layers = get_language_model_layers(model)
    if layer_idx >= len(layers):
        raise IndexError(f"Requested layer index {layer_idx} but model only has {len(layers)} layers.")

    if not model_args:
        raise ValueError("collect_layer_inputs requires at least one positional model argument (e.g., input ids).")

    collector = InputCollector(layers[layer_idx], cpu_offload=False)
    layers[layer_idx] = collector

    try:
        with torch.no_grad():
            model(*model_args, **model_kwargs)
    except ForwardInterrupt:
        pass

    if not collector.input_args:
        raise RuntimeError("Failed to capture inputs for the requested layer.")

    captured_args = tuple(collector.input_args[0])
    captured_kwargs = dict(collector.input_kwargs[0])
    original_layer = collector.module
    layers[layer_idx] = original_layer

    return original_layer, captured_args, captured_kwargs


T = TypeVar("T")


def _move_to_device(data: T, device, target_dtype) -> T:
    if isinstance(data, torch.Tensor):
        if data.is_floating_point():
            return cast(T, data.to(device=device, dtype=target_dtype))
        return cast(T, data.to(device=device))
    if isinstance(data, tuple):
        return cast(T, tuple(_move_to_device(x, device, target_dtype) for x in data))
    if isinstance(data, list):
        return cast(T, [_move_to_device(x, device, target_dtype) for x in data])
    if isinstance(data, Mapping):
        return cast(T, {k: _move_to_device(v, device, target_dtype) for k, v in data.items()})
    if isinstance(data, DynamicCache):
        for i in range(len(data.layers)):
            # data.layers[i] = _move_to_device(data.layers[i], device, target_dtype)
            if data.layers[i].keys is not None:
                data.layers[i].device = device
                data.layers[i].keys = data.layers[i].keys.to(device)
                data.layers[i].values = data.layers[i].values.to(device)
            
    return data


def _detach_to_cpu(data: T) -> T:
    if isinstance(data, torch.Tensor):
        return cast(T, data.detach().cpu())
    if isinstance(data, tuple):
        return cast(T, tuple(_detach_to_cpu(x) for x in data))
    if isinstance(data, list):
        return cast(T, [_detach_to_cpu(x) for x in data])
    if isinstance(data, Mapping):
        return cast(T, {k: _detach_to_cpu(v) for k, v in data.items()})
    return data


def _extract_present(outputs: Any) -> Any:
    if outputs is None:
        return None
    if isinstance(outputs, tuple):
        if len(outputs) >= 2:
            return outputs[1]
        return None
    if isinstance(outputs, Mapping):
        for key in ("present_key_value", "past_key_value", "present"):
            if key in outputs:
                return outputs[key]
        return None
    for attr in ("present_key_value", "past_key_value", "present"):
        if hasattr(outputs, attr):
            return getattr(outputs, attr)
    return None


def forward_layer_by_layer(model, sample_input, prefill_args, prefill_kargs, use_cache=True):
    """
        对model的language_model的layers逐层前向计算，最后返回最终输出
        每次加载一个layer到device上进行计算，计算完成后移动到cpu释放显存
        得到的out完全等价于model(sample_input, use_cache=use_cache)
    """
    def maybe_first_element(x):
        if isinstance(x, Sequence):
            x = x[0]
        return x
    past_key_values = prefill_kargs.pop("past_key_values", None)
    position_embeddings = prefill_kargs.pop("position_embeddings", None)
    if position_embeddings is not None:
        position_embeddings = _move_to_device(position_embeddings, device='cuda', target_dtype=torch.bfloat16)
    # use_cache = prefill_kargs.get("use_cache", False)
    # if use_cache and past_key_values is None and not torch.jit.is_tracing():
    #     past_key_values = DynamicCache(config=model.model.language_model.config)
    layers = get_language_model_layers(model)
    # num_layers = len(layers)
    # 遍历每一层
    # for i in range(num_layers):
    #     block = blocks[i].to('cuda')
    #     for inp_args, inp_kwargs in zip(prefill_args, prefill_kargs):
    #         with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=args.amp):
    #             out = block(*to(inp_args, device='cuda'), **to(inp_kwargs, device='cuda'))
    #         out = maybe_first_element(out).to('cuda')
    #         # change only first input argument
    #         if len(inp_args) > 0:
    #             inp_args[0].data = out
    #         elif "hidden_states" in inp_kwargs:
    #             inp_kwargs["hidden_states"] = out
    #         else:
    #             raise ValueError("Unsupported block input format.")
    #     block.to('cpu')
    hidden_states = prefill_args[0].to('cuda')
    with torch.no_grad():
        for decoder_layer in layers:
            decoder_layer.cuda()
            layer_outputs = decoder_layer(
                hidden_states,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **prefill_kargs
            )
            decoder_layer.cpu()
            hidden_states = layer_outputs[0]
    _cleanup()


    # hidden_states = model.model.language_model.norm(hidden_states)
    if past_key_values is None:
        raise ValueError("past_key_values is None")
    return past_key_values


def collect_decode_inputs(model, sample_input, layer_idx, prefill_args,prefill_kargs) -> Tuple[Optional[Tuple[Any, ...]], Optional[dict[str, Any]]]:

    sample_input = sample_input.to('cuda')
    # 对prefill_args和prefill_kargs进行深拷贝
    tmp_prefill_args = copy.deepcopy(prefill_args)
    tmp_prefill_kargs = copy.deepcopy(prefill_kargs)
    past_key_values = forward_layer_by_layer(model, sample_input, tmp_prefill_args, tmp_prefill_kargs, use_cache=True)
    past_key_values = past_key_values

    if past_key_values is None:
        print("Warning: Model did not return past key values; skipping decode benchmarking.")
        return None, None

    next_input = prepare_sample_input(sample_input.shape[0], 1, device="cpu")
    _, decode_args, decode_kwargs = collect_layer_inputs(
        model,
        layer_idx,
        next_input,
        use_cache=True,
        past_key_values=past_key_values,
    )

    del next_input

    return decode_args, decode_kwargs


def benchmark_layer(layer: torch.nn.Module, input_args: Tuple[Any, ...], input_kwargs: dict[str, Any], device: str):
    layer_device = torch.device(device)
    layer = layer.to(layer_device)
    layer.eval()

    try:
        layer_dtype = next(layer.parameters()).dtype
    except StopIteration:
        layer_dtype = torch.bfloat16

    input_args = _move_to_device(input_args, layer_device, layer_dtype)
    input_kwargs = _move_to_device(input_kwargs, layer_device, layer_dtype)

    def _forward_once():
        with torch.no_grad():
            layer(*input_args, **input_kwargs)

    times, peak_mem = module_benchmark(_forward_once)
    return times, peak_mem


def benchmark_single(args):
    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    if args.layer != 0:
        raise NotImplementedError("Decode benchmarking currently supports only layer 0.")

    model = get_model_quantized(args.model, None)
    model.eval()

    sample_input = prepare_sample_input(args.batch_size, args.prefill_seq_len, device="cpu")
    layer: torch.nn.Module
    prefill_args: Tuple[Any, ...]
    prefill_kwargs: dict[str, Any]
    layer, prefill_args, prefill_kwargs = collect_layer_inputs(
        model,
        args.layer,
        sample_input,
        use_cache=True,
    )

    decode_args, decode_kwargs = collect_decode_inputs(
        model,
        sample_input,
        args.layer,
        prefill_args,
        prefill_kwargs
    )

    del sample_input

    # 删除除了要benchmark的layer以外的model部分，释放内存
    # 深拷贝layer
    import copy
    layer = copy.deepcopy(layer)
    del model

    _cleanup()

    times_prefill, peak_mem_prefill = benchmark_layer(layer, prefill_args, prefill_kwargs, args.device)
    _cleanup()

    if decode_args is not None:
        times_decode, peak_mem_decode = benchmark_layer(layer, decode_args, decode_kwargs, args.device)
        _cleanup()
    else:
        times_decode, peak_mem_decode = None, None

    times_prefill = np.array(times_prefill)
    peak_mem_prefill = np.array(peak_mem_prefill)

    print(f"Layer {args.layer} prefill time: {np.mean(times_prefill):.3f} +- {1.96 * np.std(times_prefill):.3f}ms")
    if torch.cuda.is_available() and "cuda" in args.device:
        mem_prefill_gb = peak_mem_prefill / (1024 ** 3)
        print(f"Layer {args.layer} prefill peak memory: {np.mean(mem_prefill_gb):.3f}GB +- {1.96 * np.std(mem_prefill_gb):.3f}GB")
    else:
        print("CUDA unavailable; prefill peak memory reported as 0.")

    if times_decode is not None:
        times_decode = np.array(times_decode)
        peak_mem_decode = np.array(peak_mem_decode)
        print(f"Layer {args.layer} decode time per token: {np.mean(times_decode):.3f} +- {1.96 * np.std(times_decode):.3f}ms")
        if torch.cuda.is_available() and "cuda" in args.device:
            mem_decode_gb = peak_mem_decode / (1024 ** 3)
            print(f"Layer {args.layer} decode peak memory: {np.mean(mem_decode_gb):.3f}GB +- {1.96 * np.std(mem_decode_gb):.3f}GB")
        else:
            print("CUDA unavailable; decode peak memory reported as 0.")
    else:
        print("Decode benchmarking skipped due to unsupported model configuration.")

    # del model
    _cleanup()
    if times_decode is None or peak_mem_decode is None:
        times_decode = np.array([0.0])
        peak_mem_decode = np.array([0.0])
    return np.mean(times_prefill), np.mean(peak_mem_prefill / (1024 ** 3)), np.mean(times_decode), np.mean(peak_mem_decode / (1024 ** 3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        default='/root/wja/data/models/Qwen/Qwen2.5-VL-72B-Instruct',
        # default='/root/wja/project/github/VLM-Quant/saved_models/vlm/quantized_models/Qwen2.5-VL-72B-Instruct-nvfp-w4-a4-RTN-identity-transform',
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size for synthetic input',
        default=1,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Synthetic input sequence length',
        default=1024,
    )
    parser.add_argument(
        '--layer', type=int,
        help='Layer index to benchmark',
        default=0,
    )
    parser.add_argument(
        '--device', type=str,
        help='Device used for benchmarking the layer',
        default='cuda',
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Enable benchmarking mode',
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Enable visualization mode',
    )
    parser.add_argument(
        '--save_path_dir', type=str,
        help='Path to save the benchmark results',
        default='./benchmark_results',
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    if not args.benchmark:
        benchmark_single(args)
    else:
        # benchmark all batch sizes
        models = {'original':'/root/wja/data/models/Qwen/Qwen2.5-VL-72B-Instruct', 
                  'quantized':'/root/wja/project/github/VLM-Quant/saved_models/vlm/quantized_models/Qwen2.5-VL-72B-Instruct-nvfp-w4-a4-RTN-identity-transform'}
        batch_sizes = [1, 2, 4, 8, 16, 32]
        results = {}

        for model_name, model_path in models.items():
            for batch_size in batch_sizes:
                args.batch_size = batch_size
                args.model = model_path
                prefill_time, prefill_mem, decode_time, decode_mem = benchmark_single(args)
                results[(model_name, batch_size)] = {
                    'prefill_time_ms': prefill_time,
                    'prefill_mem_gb': prefill_mem,
                    'decode_time_ms': decode_time,
                    'decode_mem_gb': decode_mem,
                }
        # save results to csv file
        save_path = Path(args.save_path_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        csv_file = save_path / 'vlm_single_layer_benchmark_results.csv'
        with open(csv_file, 'w') as f:
            f.write('model,batch_size,prefill_time_ms,prefill_mem_gb,decode_time_ms,decode_mem_gb\n')
            for (model_name, batch_size), metrics in results.items():
                f.write(f"{model_name},{batch_size},{metrics['prefill_time_ms']},{metrics['prefill_mem_gb']},{metrics['decode_time_ms']},{metrics['decode_mem_gb']}\n")
        print(f"Benchmark results saved to {csv_file}")
        if args.visualize:
            # plot results
            import matplotlib.pyplot as plt

            for metric in ['prefill_time_ms', 'prefill_mem_gb', 'decode_time_ms', 'decode_mem_gb']:
                plt.figure()
                for model_name in models.keys():
                    x = []
                    y = []
                    for batch_size in batch_sizes:
                        x.append(batch_size)
                        y.append(results[(model_name, batch_size)][metric])
                    plt.plot(x, y, label=model_name)
                plt.xlabel('Batch Size')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'Benchmarking {metric.replace("_", " ").title()} vs Batch Size')
                plt.legend()
                plt.grid(True)
                plt.savefig(save_path / f'{metric}_vs_batch_size.png')
                plt.close()
            print(f"Benchmark visualizations saved to {save_path}")
                

        
