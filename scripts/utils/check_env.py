#!/usr/bin/env python3

import os
import sys
import socket
import subprocess
from datetime import datetime


def run(cmd: str):
    print(f"\n$ {cmd}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.stdout:
            print(result.stdout, flush=True)

        if result.stderr:
            print(result.stderr, flush=True)

        if result.returncode != 0:
            print(
                f"WARNING: command exited with code {result.returncode}",
                flush=True,
            )

    except Exception as e:
        print(f"ERROR running command: {repr(e)}", flush=True)


print("=" * 80, flush=True)
print("ENVIRONMENT CHECK", flush=True)
print("=" * 80, flush=True)

print("DATE:", datetime.now(), flush=True)
print("HOSTNAME:", socket.gethostname(), flush=True)
print("USER:", os.getenv("USER"), flush=True)
print("PWD:", os.getcwd(), flush=True)
print("PYTHON:", sys.executable, flush=True)
print("PYTHON VERSION:", sys.version, flush=True)
print("VIRTUAL_ENV:", os.getenv("VIRTUAL_ENV"), flush=True)
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"), flush=True)
print("LD_LIBRARY_PATH:", os.getenv("LD_LIBRARY_PATH"), flush=True)

run("which python")
run("python -m pip --version")
run("python -m pip show torch torchvision torchaudio")

print("\n" + "=" * 80, flush=True)
print("SLURM VARIABLES", flush=True)
print("=" * 80, flush=True)

for key in sorted(os.environ):
    if key.startswith("SLURM"):
        print(f"{key}={os.environ[key]}", flush=True)

print("\n" + "=" * 80, flush=True)
print("LOADED MODULES", flush=True)
print("=" * 80, flush=True)

run("module list")

print("\n" + "=" * 80, flush=True)
print("GPU CHECK", flush=True)
print("=" * 80, flush=True)

run("which nvidia-smi")
run("nvidia-smi -L")
run("nvidia-smi")
run(
    "nvidia-smi --query-gpu=index,name,uuid,driver_version,"
    "memory.total,memory.used,memory.free,temperature.gpu,"
    "utilization.gpu,power.draw --format=csv"
)

print("\n" + "=" * 80, flush=True)
print("PYTORCH / CUDA CHECK", flush=True)
print("=" * 80, flush=True)

try:
    import torch
    import torchvision

    print("torch version:", torch.__version__, flush=True)
    print("torchvision version:", torchvision.__version__, flush=True)
    print("torch installation:", torch.__file__, flush=True)

    print("PyTorch compiled CUDA version:", torch.version.cuda, flush=True)
    print("cuDNN version:", torch.backends.cudnn.version(), flush=True)
    print("cuDNN enabled:", torch.backends.cudnn.enabled, flush=True)

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("CUDA device count:", torch.cuda.device_count(), flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch does not see the GPU")

    for i in range(torch.cuda.device_count()):
        print("\n" + "-" * 40, flush=True)
        print(f"GPU {i}", flush=True)
        print("-" * 40, flush=True)

        props = torch.cuda.get_device_properties(i)

        print("name:", torch.cuda.get_device_name(i), flush=True)
        print("compute capability:", f"{props.major}.{props.minor}", flush=True)
        print(
            "total memory GB:",
            round(props.total_memory / 1024**3, 2),
            flush=True,
        )

        torch.cuda.set_device(i)

        x = torch.randn(4096, 4096, device=f"cuda:{i}")
        y = torch.matmul(x, x)

        torch.cuda.synchronize(i)

        print("tensor device:", y.device, flush=True)
        print("tensor shape:", tuple(y.shape), flush=True)
        print("tensor test: OK", flush=True)

        del x
        del y
        torch.cuda.empty_cache()

    print("\nPYTORCH CUDA TEST: SUCCESS", flush=True)

except Exception as e:
    print("\nPYTORCH CUDA TEST: FAILED", flush=True)
    print("ERROR:", repr(e), flush=True)
    raise

print("\n" + "=" * 80, flush=True)
print("CPU CHECK", flush=True)
print("=" * 80, flush=True)

run("lscpu | head -40")

print("\n" + "=" * 80, flush=True)
print("MEMORY CHECK", flush=True)
print("=" * 80, flush=True)

run("free -h")

print("\n" + "=" * 80, flush=True)
print("FILESYSTEM CHECK", flush=True)
print("=" * 80, flush=True)

run("df -h /work")
run("df -h $HOME")

print("\nCHECK COMPLETED", flush=True)
print("END DATE:", datetime.now(), flush=True)