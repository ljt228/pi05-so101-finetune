#!/usr/bin/env bash
# Fine-tune pi0.5 on the SO-101 demonstration dataset (dual-expert LoRA).
#
# Requirements:
#   - openpi codebase installed (uv sync)
#   - the TrainConfig block from ../config/pi05_so101_lora_finetune.py
#     registered in src/openpi/training/config.py
#   - LeRobot v2.1 dataset available locally
#     (~/.cache/huggingface/lerobot/local/record-breakfast_49src_v21)
#
# Hardware: single NVIDIA GPU (24 GB+). The run shown in assets/train.log
# used an NVIDIA L20.

uv run scripts/train.py pi05_so101_lora_finetune \
    --exp so101_lora_v1 \
    --save-interval 5000 \
    --log-interval 100
