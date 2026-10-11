#!/usr/bin/env bash
# Prints what this project needs to know about the machine. Read-only; changes nothing.
echo "== GPU ==";    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not found (no NVIDIA driver?)"
echo "== CUDA (driver max) =="; nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*"
echo "== Python ==";  for p in python3.12 python3.11 python3.10 python3; do command -v $p >/dev/null && echo "$p -> $($p --version 2>&1)"; done
echo "== conda ==";   command -v conda >/dev/null && conda --version || echo "conda not found (venv works too)"
echo "== tmux/screen =="; command -v tmux || command -v screen || echo "neither found: use nohup"
echo "== Disk free here =="; df -h . | tail -1
echo "== Internet =="; curl -sI --max-time 10 https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip | head -1 || echo "dataset host unreachable"
curl -sI --max-time 10 https://pypi.org/simple/ | head -1 || echo "pypi unreachable"
