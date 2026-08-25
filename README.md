# Fine-tuning π₀.₅ for Real-World Robot Manipulation on SO-101

![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=150&section=header&text=%CF%80%E2%82%80.%E2%82%85%20%C3%97%20SO-101&fontSize=34&fontColor=ffffff&fontAlignY=36&desc=Data%20Collection%20%C2%B7%20LoRA%20Fine-tuning%20%C2%B7%20Real-Robot%20Deployment&descAlignY=58&descSize=15)

![JAX](https://img.shields.io/badge/JAX-00539F?style=flat) ![openpi](https://img.shields.io/badge/openpi-Physical_Intelligence-2563EB?style=flat) ![LeRobot](https://img.shields.io/badge/LeRobot-v2.1-C2410C?style=flat) ![SO-101](https://img.shields.io/badge/SO--101-Real_Robot-059669?style=flat) ![LoRA](https://img.shields.io/badge/%CF%80%E2%82%80.%E2%82%85-LoRA_Fine--tune-8A2BE2?style=flat)

End-to-end pipeline for adapting **π₀.₅** (Physical Intelligence's vision-language-action
model) to a **real SO-101 manipulator**: designing manipulation tasks, collecting
teleoperated demonstrations, packaging them as LeRobot datasets, LoRA fine-tuning the
full VLA on a single GPU, and deploying the policy back on the real arm.

> Done during a research internship. <!-- 在这里补公司名（如允许）：at xcdl -->

## Demo

| Global view (data collection) | Wrist view |
|---|---|
| ![global](assets/frame_global.jpg) | ![wrist](assets/frame_wrist.jpg) |

▶ Raw demonstration clips: [`assets/sample_global.mp4`](assets/sample_global.mp4) ·
[`assets/sample_wrist.mp4`](assets/sample_wrist.mp4) ·
[`assets/sample_breakfast_front.mp4`](assets/sample_breakfast_front.mp4)

<!-- TODO: 真机部署视频 —— 把你的 rollout 视频放到 assets/deploy.mp4，然后取消下行注释：
<video src="assets/deploy.mp4" controls width="480"></video>
-->

## Pipeline

```
 task design ──► teleoperation ──► LeRobot v2.1 ──► π₀.₅ LoRA fine-tune ──► real-arm rollout
 (2 tasks)       (JoyCon, 30 fps)    (25 + 14 eps)     (dual-expert LoRA)     (SO-101)
```

### 1 · Task design

Two manipulation tasks of increasing difficulty, designed from scratch:

| Task | Instruction (prompt) | Episodes | Cameras |
|---|---|---|---|
| **yellow-block-plate** | *"Pick up the yellow block and place it into the plate"* | 25 | global + wrist |
| **breakfast table-setting** | *"First put the block onto the plate, move the plate to the center of the table, place the spoon on the right side of the plate, and place the cup on the left side of the plate."* | 14 | front + side |

The breakfast task is **multi-step and long-horizon** — four sequential object
rearrangements in one language-conditioned rollout — which is exactly the regime
where VLA pretraining should pay off.

### 2 · Data collection

- **Hardware:** SO-101 leader–follower arm pair, **JoyCon-based teleoperation**
- **Recording:** `lerobot-record`, 30 fps, joint-space actions (6-DoF: shoulder_pan /
  shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper)
- **Observations:** one global scene camera (640×480) + one wrist camera, stored as
  mp4 video streams alongside parquet state/action tracks (LeRobot v2.1)

### 3 · Fine-tuning π₀.₅

Rather than training a policy from scratch, I adapted the pretrained **π₀.₅** VLA
(PaliGemma 3B VLM + 300M action expert, flow-matching action decoder) with **LoRA
adapters on both experts**, starting from the public `pi05_base` checkpoint:

- framework: [openpi](https://github.com/Physical-Intelligence/openpi) (JAX)
- config: [`config/pi05_so101_lora_finetune.py`](config/pi05_so101_lora_finetune.py)
- 30k steps planned, checkpoints every 5k; trained on a single NVIDIA L20
- full config in [`config/`](config/), launch script in [`scripts/train.sh`](scripts/train.sh)

### 4 · Results

Training converges smoothly — flow-matching loss drops from **4.7e-2 to 8e-4**
within ~13k steps and stays flat:

![loss curve](assets/loss_curve.png)

The fine-tuned policy was deployed back on the physical SO-101 arm and executes the
task in the real world. <!-- rollout video 见上方 TODO -->

## Dataset card

Both datasets follow the [LeRobot v2.1](https://github.com/huggingface/lerobot) schema
(see [`docs/dataset.md`](docs/dataset.md) for the full feature spec):

| Field | Value |
|---|---|
| Format | LeRobot v2.1 (parquet + mp4 video streams) |
| FPS | 30 |
| Action | `float32[6]` — joint positions |
| State | `float32[6]` — joint positions |
| Cameras | `observation.images.global` 640×480 · `observation.images.wrist` |
| Size | 335 MB (yellow-block-plate, 25 episodes) |

## Repo structure

```
├── assets/                  # sample videos, frames, loss curve, raw training log
├── config/
│   └── pi05_so101_lora_finetune.py   # the TrainConfig used (openpi)
├── scripts/
│   └── train.sh             # training launch command
├── docs/
│   └── dataset.md           # LeRobot v2.1 feature specification
└── README.md
```

## Reproducing

```bash
git clone https://github.com/Physical-Intelligence/openpi
cd openpi && uv sync                        # installs jax + deps

# place your LeRobot dataset under ~/.cache/huggingface/lerobot/<repo_id>
# drop the config block into src/openpi/training/config.py

uv run scripts/train.py pi05_so101_lora_finetune --exp so101_lora_v1
```

Raw training log (129 loss/grad-norm records): [`assets/train.log`](assets/train.log).

## Acknowledgements

- [openpi](https://github.com/Physical-Intelligence/openpi) & π₀.₅ —
  Physical Intelligence ([paper](https://arxiv.org/abs/2504.16054))
- [LeRobot](https://github.com/huggingface/lerobot) & the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm — Hugging Face + TheRobotStudio

---

*Built by [ljt228](https://github.com/ljt228) — M.S. student in mechanical
engineering, working on VLA / world-action models.*
