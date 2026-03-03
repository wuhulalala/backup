"""
    测试普通的Linear和FPQuant Linear的速度差异
    todo:
    1. 验证下qutlass改前和改后的差异
"""
import sys
import time
from pathlib import Path
from typing import Dict, List

# REPO_ROOT = Path(__file__).resolve().parents[1]
# INFERENCE_LIB_SRC = REPO_ROOT / "inference_lib" / "src"
# if str(INFERENCE_LIB_SRC) not in sys.path:
#     sys.path.insert(0, str(INFERENCE_LIB_SRC))

import torch
import torch.nn as nn

from fp_quant.module import FPQuantLinear
from fp_quant.utils import FPQuantConfig, FPQuantDtype


DEFAULT_TEST_CONFIGS: List[Dict[str, int]] = [
    # {"in_features": 4096, "out_features": 4096, "batch_size": 1, "seq_len": 128},
    # {"in_features": 4096, "out_features": 4096, "batch_size": 4, "seq_len": 128},
    # {"in_features": 4096, "out_features": 11008, "batch_size": 1, "seq_len": 128},
    # {"in_features": 11008, "out_features": 4096, "batch_size": 1, "seq_len": 128},
    
    # {"in_features": 1024, "out_features": 8192, "batch_size": 1, "seq_len": 1},
    # {"in_features": 1024, "out_features": 8192, "batch_size": 1, "seq_len": 128},
    # {"in_features": 1024, "out_features": 8192, "batch_size": 1, "seq_len": 1024},
    # {"in_features": 8192, "out_features": 8192, "batch_size": 1, "seq_len": 1},
    # {"in_features": 8192, "out_features": 8192, "batch_size": 1, "seq_len": 128},
    # {"in_features": 8192, "out_features": 8192, "batch_size": 1, "seq_len": 1024},
    # {"in_features": 29568, "out_features": 8192, "batch_size": 1, "seq_len": 1},
    # {"in_features": 29568, "out_features": 8192, "batch_size": 1, "seq_len": 128},
    # {"in_features": 29568, "out_features": 8192, "batch_size": 1, "seq_len": 1024},

    {"in_features": 1024, "out_features": 8192, "batch_size": 32, "seq_len": 1},
    {"in_features": 1024, "out_features": 8192, "batch_size": 32, "seq_len": 128},
    {"in_features": 1024, "out_features": 8192, "batch_size": 32, "seq_len": 1024},
    {"in_features": 8192, "out_features": 8192, "batch_size": 32, "seq_len": 1},
    {"in_features": 8192, "out_features": 8192, "batch_size": 32, "seq_len": 128},
    {"in_features": 8192, "out_features": 8192, "batch_size": 32, "seq_len": 1024},
    {"in_features": 29568, "out_features": 8192, "batch_size": 32, "seq_len": 1},
    {"in_features": 29568, "out_features": 8192, "batch_size": 32, "seq_len": 128},
    {"in_features": 29568, "out_features": 8192, "batch_size": 32, "seq_len": 1024},

    {"in_features": 1024, "out_features": 8192, "batch_size": 64, "seq_len": 1},
    {"in_features": 1024, "out_features": 8192, "batch_size": 64, "seq_len": 128},
    {"in_features": 1024, "out_features": 8192, "batch_size": 64, "seq_len": 1024},
    {"in_features": 8192, "out_features": 8192, "batch_size": 64, "seq_len": 1},
    {"in_features": 8192, "out_features": 8192, "batch_size": 64, "seq_len": 128},
    {"in_features": 8192, "out_features": 8192, "batch_size": 64, "seq_len": 1024},
    {"in_features": 29568, "out_features": 8192, "batch_size": 64, "seq_len": 1},
    {"in_features": 29568, "out_features": 8192, "batch_size": 64, "seq_len": 128},
    {"in_features": 29568, "out_features": 8192, "batch_size": 64, "seq_len": 1024},
]


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    module: nn.Module,
    inputs: torch.Tensor,
    warmup: int,
    runs: int,
    device: torch.device,
) -> float:
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            module(inputs)
        synchronize_if_needed(device)
        start = time.perf_counter()
        for _ in range(runs):
            module(inputs)
        synchronize_if_needed(device)
    return (time.perf_counter() - start) / runs


def benchmark_linear_layers(
    test_configs: List[Dict[str, int]] = DEFAULT_TEST_CONFIGS,
    warmup: int = 5,
    runs: int = 50,
) -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    fpquant_config = FPQuantConfig(
        forward_dtype=FPQuantDtype.NVFP4,
        forward_method="abs_max",
        backward_dtype=FPQuantDtype.BF16,
        store_master_weights=False,
        hadamard_group_size=32,
        pseudoquantization=False,
        transform_init="identity",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    for cfg in test_configs:
        in_features = cfg["in_features"]
        out_features = cfg["out_features"]
        batch_size = cfg["batch_size"]
        seq_len = cfg["seq_len"]

        print(
            f"\nShape (batch={batch_size}, seq={seq_len}, in={in_features}, out={out_features})"
        )

        standard_linear = nn.Linear(in_features, out_features,dtype=torch.bfloat16).to(device)
        fpquant_linear = FPQuantLinear(
            in_features,
            out_features,
            fpquant_config,
            dtype=torch.bfloat16,
        ).to(device)

        standard_linear.eval()
        fpquant_linear.eval()

        with torch.no_grad():
            standard_linear.weight.fill_(1.0)
            if standard_linear.bias is not None:
                standard_linear.bias.zero_()
            if fpquant_linear.weight is not None:
                fpquant_linear.weight.fill_(1.0)
            else:
                factory_kwargs = {"device": device, "dtype": torch.bfloat16}
                fpquant_linear.weight = nn.Parameter(
                    torch.empty((out_features, in_features), **factory_kwargs)
                )
                fpquant_linear.dqweight = nn.Parameter(
                    torch.empty((out_features, in_features), **factory_kwargs)
                )
            if fpquant_linear.bias is not None:
                fpquant_linear.bias.zero_()

        with torch.no_grad():
            fpquant_linear.pre_forward()

        x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            baseline_out = standard_linear(x)
            fpquant_out = fpquant_linear(x)

        standard_time = measure_latency(standard_linear, x, warmup, runs, device)
        fpquant_time = measure_latency(fpquant_linear, x, warmup, runs, device)

        speedup = standard_time / fpquant_time if fpquant_time > 0 else float("inf")

        print(f"nn.Linear latency   : {standard_time * 1e3:.3f} ms")
        print(f"FPQuantLinear latency: {fpquant_time * 1e3:.3f} ms")
        print(f"Speedup (baseline/fp): {speedup:.2f}x")

        results.append(
            {
                "shape": (batch_size, seq_len, in_features, out_features),
                "nn_linear_ms": standard_time * 1e3,
                "fpquant_ms": fpquant_time * 1e3,
                "speedup": speedup,
            }
        )

    print("\nBenchmark summary:")
    for item in results:
        shape = item["shape"]
        print(
            f"  shape={shape}\tnn.Linear={item['nn_linear_ms']:.3f} ms\t"
            f"FPQuantLinear={item['fpquant_ms']:.3f} ms\tspeedup={item['speedup']:.2f}x"
        )


if __name__ == "__main__":
    benchmark_linear_layers()

