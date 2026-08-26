# Technical Report: Fine-tuning π₀.₅ for SO-101 Real-World Manipulation

## 1. Overview

This project adapts Physical Intelligence's **π₀.₅** (pi-zero-point-five) Vision-Language-Action (VLA) model to a real **SO-101** 6-DoF manipulator arm via LoRA fine-tuning. The pipeline covers teleoperation-based data collection, dataset packaging, single-GPU fine-tuning, and real-robot deployment.

## 2. System Architecture

### 2.1 Hardware

| Component | Specification |
|---|---|
| Robot | SO-101 follower arm (6-DoF, Trossen Robotics) |
| Teleoperation | Nintendo Joy-Con (L/R) via Bluetooth |
| Cameras | 2× USB cameras (global scene + wrist), 640×480 @ 30fps |
| GPU | NVIDIA L20 (46GB VRAM) |
| CPU | Shared server (188GB RAM) |

### 2.2 Software Stack

| Layer | Technology |
|---|---|
| VLA Model | π₀.₅ (PaliGemma 3B VLM + 300M action expert) |
| Framework | JAX 0.5.3, Flax 0.10.2 |
| Fine-tuning | LoRA adapters on both VLM and action expert |
| Dataset | LeRobot v2.1 (parquet + mp4 video streams) |
| Robot SDK | `so-101`, `gym-so101` (Trossen) |
| Inference Server | openpi WebSocket server |

## 3. Data Collection

### 3.1 Teleoperation Interface

The Joy-Con controller maps to 6-DoF joint actions:

| Joy-Con Input | Joint Action | Control Mode |
|---|---|---|
| Left Stick X | `shoulder_pan.pos` | Proportional (18°/s) |
| Left Stick Y | `shoulder_lift.pos` | Proportional (18°/s) |
| D-pad Left/Right (544/545) | `elbow_flex.pos` | Proportional (20°/s) |
| D-pad Up/Down (546/547) | `wrist_flex.pos` | Proportional (20°/s) |
| SL / SR buttons | `wrist_roll.pos` | Step (5°/press) |
| TR / ZR buttons | `gripper.pos` | Step (6°/press) |

**Deadzone**: 12% of stick range to prevent drift.

### 3.2 Recording Pipeline

```
Joy-Con → JoyConTeleop.get_action() → SO101Follower.send_action()
                    ↓
          LeRobotDataset.save_episode()
                    ↓
     data/chunk-XXX/episode_XXXXXX.parquet
     videos/chunk-XXX/{camera}/episode_XXXXXX.mp4
```

### 3.3 Dataset Schema (LeRobot v2.1)

```json
{
  "codebase_version": "v2.1",
  "robot_type": "so101_follower",
  "fps": 30,
  "features": {
    "action": {"dtype": "float32", "shape": [6]},
    "observation.state": {"dtype": "float32", "shape": [6]},
    "observation.images.global": {"dtype": "video"},
    "observation.images.wrist": {"dtype": "video"}
  }
}
```

### 3.4 Collected Datasets

| Dataset | Episodes | Frames | Size | Task |
|---|---|---|---|---|
| `so101-yellow-block-plate` | 25 | 34,934 | 335 MB | Pick-and-place |
| `so101-breakfast` | 49 | 75,682 | 790 MB | Multi-step table setting |

## 4. Model Architecture

### 4.1 π₀.₅ Base Model

π₀.₅ is a VLA model that processes:
- **Visual input**: 2 RGB images (224×224) from dual cameras
- **Language input**: Task description prompt
- **State input**: 6-DoF joint positions

It outputs a sequence of 6-DoF joint position actions via flow-matching decoder.

### 4.2 LoRA Fine-tuning Configuration

```python
TrainConfig(
    name="pi05_so101_lora_finetune",
    model=Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",       # LoRA into 3B VLM
        action_expert_variant="gemma_300m_lora",  # LoRA into 300M action expert
    ),
    data=LeRobotSO101DataConfig(
        repo_id="local/record-breakfast_49src_v21",
        default_prompt="first put the block onto the plate, ...",
    ),
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
)
```

### 4.3 Data Transform Pipeline

```
Raw observation
  → RepackTransform (rename camera keys)
  → SO101Inputs (format images + state)
  → DeltaActions (optional, joint-space deltas)
  → Normalize (z-score from norm_stats.json)
  → ModelTransform (tokenize for VLA)
```

## 5. Training

### 5.1 Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Batch size | 32 |
| Learning rate | Default openpi schedule |
| Max steps | 30,000 |
| Checkpoint interval | 2,000 steps |
| EMA | Disabled |
| GPU | Single NVIDIA L20 |

### 5.2 Training Command

```bash
uv run scripts/train.py pi05_so101_lora_finetune --exp so101_lora_v1
```

### 5.3 Loss Convergence

| Step | Loss |
|---|---|
| 2,000 | 0.2129 |
| 10,000 | 0.1191 |
| 20,000 | 0.1350 |
| 28,000 | 0.1060 |

The model shows steady convergence with flow-matching loss dropping from ~0.21 to ~0.10.

## 6. Inference Pipeline

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────┐
│  Client (so101_breakfast.py)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Camera   │  │ Robot    │  │ WebSocket Client │  │
│  │ (2×USB)  │  │ SO-101   │  │ → openpi server  │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                 │             │
│       ▼              ▼                 ▼             │
│  ┌─────────────────────────────────────────────┐    │
│  │           ActionChunkBroker                  │    │
│  │    (buffer 25 actions, execute at 30Hz)      │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
         ↕ WebSocket (localhost:10093)
┌─────────────────────────────────────────────────────┐
│  Server (openpi serve_policy.py)                    │
│  ┌─────────────────────────────────────────────┐    │
│  │  π₀.₅ Model (LoRA fine-tuned)               │    │
│  │  Input: images + state + prompt              │    │
│  │  Output: action chunk (25 × 6-DoF)           │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### 6.2 Execution Loop

1. **Capture** dual camera images + robot joint state
2. **Send** observation to openpi server via WebSocket
3. **Receive** action chunk (25 future actions)
4. **Execute** actions at 30Hz via ActionChunkBroker
5. **Repeat** until episode complete

### 6.3 Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `action_horizon` | 25 | Actions per inference call |
| `max_hz` | 30.0 | Control frequency |
| `MODEL_IMAGE_SIZE` | 224 | Input image resolution |

## 7. Results

### 7.1 Task Performance

| Task | Success Rate | Notes |
|---|---|---|
| Yellow block → plate | High | Single-step pick-and-place |
| Breakfast table-setting | Moderate | 4-step sequential rearrangement |

### 7.2 Qualitative Observations

- The model generalizes to slight variations in object placement
- Dual-camera input (global + wrist) improves spatial reasoning
- LoRA fine-tuning preserves base model capabilities while adapting to SO-101 kinematics

## 8. Future Work

1. **Multi-task training**: Combine both datasets for unified policy
2. **Longer horizons**: Extend action chunk beyond 25 steps
3. **Sim-to-real**: Use simulation data for pre-training
4. **Closed-loop**: Add force/torque feedback for delicate manipulation

## 9. Repository Structure

```
pi05-so101-finetune/
├── src/openpi/                    # Modified openpi source
│   ├── training/config.py         # SO-101 data config + train config
│   └── policies/so101_policy.py   # SO-101 input/output transforms
├── scripts/
│   ├── train.py                   # Training entry point
│   ├── serve_policy.py            # Inference server
│   ├── so101_breakfast.py         # Real-robot inference client
│   ├── joycon_left_so101_teleop.py # JoyCon teleoperation
│   └── record_yellow_block_plate.py # Data recording
├── config/                        # Config reference
├── docs/                          # Documentation
├── assets/                        # Demo videos, loss curves
└── pyproject.toml                 # Dependencies
```

## 10. References

- [π₀.₅ paper](https://arxiv.org/abs/2504.16054) — Physical Intelligence
- [openpi](https://github.com/Physical-Intelligence/openpi) — Open-source VLA framework
- [LeRobot](https://github.com/huggingface/lerobot) — HuggingFace robotics toolkit
- [SO-101](https://github.com/TheRobotStudio/SO-ARM100) — Trossen Robotics manipulator
