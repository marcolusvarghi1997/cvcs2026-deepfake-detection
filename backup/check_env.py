#!/usr/bin/env python3

import os
import time
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
            print(f"WARNING: command exited with code {result.returncode}", flush=True)

    except Exception as e:
        print(f"ERROR running command: {repr(e)}", flush=True)


print("=" * 80, flush=True)
print("ENVIRONMENT CHECK", flush=True)
print("=" * 80, flush=True)

print("DATE:", datetime.now(), flush=True)
print("HOSTNAME:", socket.gethostname(), flush=True)
print("USER:", os.getenv("USER"), flush=True)
print("PWD:", os.getcwd(), flush=True)
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"), flush=True)

print("\n" + "=" * 80, flush=True)
print("SLURM VARIABLES", flush=True)
print("=" * 80, flush=True)

for key in sorted(os.environ):
    if key.startswith("SLURM"):
        print(f"{key}={os.environ[key]}", flush=True)

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

    print("torch version:", torch.__version__, flush=True)
    print("torch cuda available:", torch.cuda.is_available(), flush=True)
    print("torch cuda device count:", torch.cuda.device_count(), flush=True)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}", flush=True)
            print("name:", torch.cuda.get_device_name(i), flush=True)

            props = torch.cuda.get_device_properties(i)
            print("total memory GB:", round(props.total_memory / 1024**3, 2), flush=True)

            x = torch.randn(4096, 4096, device=f"cuda:{i}")
            y = torch.matmul(x, x)
            torch.cuda.synchronize(i)

            print("tensor test: OK", flush=True)
    else:
        print("PROBLEM: PyTorch does not see CUDA", flush=True)

except Exception as e:
    print("TORCH ERROR:", repr(e), flush=True)

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

print("\n" + "=" * 80, flush=True)
print("KEEP JOB ALIVE FOR 60 SECONDS", flush=True)
print("=" * 80, flush=True)

time.sleep(60)

print("\nCHECK COMPLETED", flush=True)
print("END DATE:", datetime.now(), flush=True)