# Dataset specification (LeRobot v2.1)

Both demonstration datasets use the [LeRobot v2.1](https://github.com/huggingface/lerobot)
format: parquet files for state/action tracks, mp4 video streams per camera,
one directory per episode chunk.

## Datasets

| | yellow-block-plate | breakfast table-setting |
|---|---|---|
| Episodes | 25 | 14 |
| Cameras | global, wrist | front, side |
| Task | pick the yellow block → place into plate | 4-step table rearrangement (block → plate → center → spoon → cup) |
| Teleop | JoyCon | JoyCon |

## Features

```json
{
  "action": {
    "dtype": "float32",
    "shape": [6],
    "names": [
      "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
      "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
    ],
    "fps": 30.0
  },
  "observation.state": { "dtype": "float32", "shape": [6], "fps": 30.0 },
  "observation.images.global": { "dtype": "video", "shape": [480, 640, 3], "fps": 30.0 },
  "observation.images.wrist":  { "dtype": "video", "shape": [480, 640, 3], "fps": 30.0 },
  "timestamp":       { "dtype": "float32" },
  "frame_index":     { "dtype": "int64" },
  "episode_index":   { "dtype": "int64" },
  "index":           { "dtype": "int64" },
  "task_index":      { "dtype": "int64" }
}
```

## Directory layout

```
<dataset>/
├── meta/
│   ├── info.json          # schema, fps, shapes, totals
│   ├── episodes.jsonl     # per-episode length & task index
│   ├── tasks.jsonl        # natural-language task prompts
│   └── stats.json         # normalization statistics (consumed by openpi)
├── data/chunk-000/        # episode_0000NN.parquet — state/action tracks
└── videos/chunk-000/
    ├── observation.images.global/episode_0000NN.mp4
    └── observation.images.wrist/episode_0000NN.mp4
```

## Normalization

`norm_stats.json` (mean/std per feature) is computed at dataset build time and
consumed by openpi's `AssetsConfig` during training — the same stats must be
served at inference time.
