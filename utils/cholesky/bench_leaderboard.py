"""
Runs the 15 official benchmark specs from
https://www.gpumode.com/leaderboard/776?tab=reference against a
custom_kernel implementation and reports per-case timing plus the
geometric mean (the leaderboard's ranking metric).
"""

import importlib
import math
import statistics
import sys
import time

import torch

BENCHMARKS = [
    {"batch": 4096, "n": 32, "cond": 2, "seed": 41032},
    {"batch": 1024, "n": 64, "cond": 2, "seed": 41064},
    {"batch": 256, "n": 128, "cond": 2, "seed": 41128},
    {"batch": 64, "n": 256, "cond": 2, "seed": 41256},
    {"batch": 16, "n": 512, "cond": 2, "seed": 41512},
    {"batch": 640, "n": 512, "cond": 2, "seed": 510512},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 42024},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 511024},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 44048},
    {"batch": 8, "n": 2048, "cond": 2, "seed": 512048},
    {"batch": 1, "n": 4096, "cond": 2, "seed": 48096},
    {"batch": 2, "n": 4096, "cond": 2, "seed": 514096},
    {"batch": 1, "n": 8192, "cond": 2, "seed": 48192},
    {"batch": 1, "n": 16384, "cond": 2, "seed": 48284},
    {"batch": 1, "n": 32768, "cond": 2, "seed": 48368},
]


def generate_input(batch: int, n: int, cond: int, seed: int, case: str = "dense") -> torch.Tensor:
    assert batch > 0 and n > 0 and cond >= 0
    assert case == "dense", "only the 'dense' case is used by this leaderboard"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)

    a = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    out = (a @ a.transpose(-1, -2)) / float(max(n, 1))
    out.diagonal(dim1=-2, dim2=-1).add_(10.0 ** -max(cond, 2))
    return (0.5 * (out + out.transpose(-1, -2))).contiguous()


def bench_one(custom_kernel, spec, warmup=2, iters=5):
    data = generate_input(spec["batch"], spec["n"], spec["cond"], spec["seed"])

    for _ in range(warmup):
        custom_kernel(data)
    if data.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if data.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        custom_kernel(data)
        if data.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    del data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return statistics.median(times)


def main(module_name: str):
    mod = importlib.import_module(module_name)
    custom_kernel = mod.custom_kernel

    times = []
    print(f"{'batch':>6} {'n':>6} {'cond':>5} {'seed':>8}   time (s)")
    for spec in BENCHMARKS:
        t = bench_one(custom_kernel, spec)
        times.append(t)
        print(f"{spec['batch']:>6} {spec['n']:>6} {spec['cond']:>5} {spec['seed']:>8}   {t:.6f}")

    geomean = math.exp(sum(math.log(t) for t in times) / len(times))
    print(f"\ngeomean over {len(times)} cases: {geomean:.6f} s")


if __name__ == "__main__":
    module_name = sys.argv[1] if len(sys.argv) > 1 else "starter"
    main(module_name)
