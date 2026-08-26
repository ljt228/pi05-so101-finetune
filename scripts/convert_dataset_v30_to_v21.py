from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import jsonlines
import pyarrow.parquet as pq
import imageio_ffmpeg


V21 = "v2.1"
V30 = "v3.0"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_STATS_PATH = "meta/stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def load_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta/info.json").read_text(encoding="utf-8"))


def write_info(info: dict[str, Any], root: Path) -> None:
    path = root / "meta/info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def load_episode_records(root: Path) -> list[dict[str, Any]]:
    episodes_dir = root / "meta/episodes"
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}")

    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())
    records.sort(key=lambda r: int(r["episode_index"]))
    return records


def flatten_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature, values in stats.items():
        if not isinstance(values, dict):
            continue
        out[feature] = {k: v for k, v in values.items() if k in {"mean", "std", "min", "max", "count"}}
    return out


def convert_info(root: Path, new_root: Path, episode_records: list[dict[str, Any]], video_keys: list[str]) -> None:
    info = load_info(root)
    total_episodes = int(info.get("total_episodes") or len(episode_records))
    chunks_size = int(info.get("chunks_size") or DEFAULT_CHUNK_SIZE)

    info["codebase_version"] = V21
    info["data_path"] = LEGACY_DATA_PATH
    info["video_path"] = LEGACY_VIDEO_PATH if video_keys else None
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    for feat in info.get("features", {}).values():
        if isinstance(feat, dict) and feat.get("dtype") != "video":
            feat.pop("fps", None)

    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes else 0
    info["total_videos"] = total_episodes * len(video_keys)
    write_info(info, new_root)


def convert_tasks(root: Path, new_root: Path) -> None:
    table = pq.read_table(root / "meta/tasks.parquet")
    df = table.to_pandas().reset_index()
    task_col = df.columns[0]
    out = new_root / LEGACY_TASKS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out, mode="w") as writer:
        for _, row in df.iterrows():
            writer.write(
                {
                    "task_index": int(row["task_index"]),
                    "task": to_serializable(row[task_col]),
                }
            )


def convert_data(root: Path, new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        grouped[(int(record["data/chunk_index"]), int(record["data/file_index"]))].append(record)

    for (chunk_idx, file_idx), records in grouped.items():
        source_path = root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        table = pq.read_table(source_path)
        records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])

        for record in records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start
            if length <= 0:
                raise ValueError(f"Invalid episode length for episode_index={episode_index}: {length}")

            episode_table = table.slice(start, length)
            dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
            dest_path = new_root / LEGACY_DATA_PATH.format(
                episode_chunk=dest_chunk,
                episode_index=episode_index,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, dest_path)


def convert_videos(root: Path, new_root: Path, episode_records: list[dict[str, Any]], video_keys: list[str], fps: float) -> None:
    if not video_keys:
        return

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    for video_key in video_keys:
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        for record in episode_records:
            if chunk_col in record and file_col in record and record[chunk_col] is not None and record[file_col] is not None:
                grouped[(int(record[chunk_col]), int(record[file_col]))].append(record)

        for (chunk_idx, file_idx), records in grouped.items():
            source_path = root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            records = sorted(records, key=lambda r: float(r[f"videos/{video_key}/from_timestamp"]))
            for record in records:
                episode_index = int(record["episode_index"])
                start_t = float(record[f"videos/{video_key}/from_timestamp"])
                end_t = float(record[f"videos/{video_key}/to_timestamp"])
                duration = max(end_t - start_t, 1e-6)

                dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
                dest_path = new_root / LEGACY_VIDEO_PATH.format(
                    episode_chunk=dest_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_t:.6f}",
                    "-i",
                    str(source_path),
                    "-t",
                    f"{duration:.6f}",
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "1",
                    "-y",
                    str(dest_path),
                ]
                subprocess.run(cmd, check=True)


def convert_episodes_metadata(new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    episodes_path = new_root / LEGACY_EPISODES_PATH
    stats_path = new_root / LEGACY_STATS_PATH
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(episodes_path, mode="w") as episodes_writer, jsonlines.open(stats_path, mode="w") as stats_writer:
        for record in sorted(episode_records, key=lambda r: int(r["episode_index"])):
            legacy_episode = {
                k: v
                for k, v in record.items()
                if not k.startswith("data/")
                and not k.startswith("videos/")
                and not k.startswith("stats/")
                and not k.startswith("meta/")
                and k not in {"dataset_from_index", "dataset_to_index"}
            }
            episodes_writer.write({k: to_serializable(v) for k, v in legacy_episode.items()})

            stats_flat = {k: v for k, v in record.items() if k.startswith("stats/")}
            nested: dict[str, Any] = {}
            for key, value in stats_flat.items():
                parts = key.split("/")
                node = nested
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
            stats_writer.write(
                {
                    "episode_index": int(record["episode_index"]),
                    "stats": to_serializable(flatten_stats(nested.get("stats", {}))),
                }
            )


def convert_dataset(root: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    episode_records = load_episode_records(root)
    info = load_info(root)
    video_keys = [k for k, v in info.get("features", {}).items() if isinstance(v, dict) and v.get("dtype") == "video"]
    fps = float(info.get("fps") or 30.0)

    convert_info(root, output_root, episode_records, video_keys)
    convert_tasks(root, output_root)
    convert_data(root, output_root, episode_records)
    convert_videos(root, output_root, episode_records, video_keys, fps)
    convert_episodes_metadata(output_root, episode_records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="v3.0 dataset root")
    parser.add_argument(
        "--output-root",
        default=None,
        help="output root for v2.1 dataset; defaults to <root>_v21",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else root.with_name(root.name + "_v21")
    convert_dataset(root, output_root)
    print(f"Converted {root} -> {output_root}")


if __name__ == "__main__":
    main()
