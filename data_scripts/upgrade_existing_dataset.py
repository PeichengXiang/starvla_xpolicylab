#!/usr/bin/env python3
"""Upgrade the existing StarVLA LeRobot datasets without re-encoding real video.

The published dataset is built in a sibling staging directory and atomically
renamed into place. Existing parquet/action data and real videos are reused.
For EgoVLA only, two 224x224 constant-black wrist streams are added.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd


CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
EGO_INDICES = {
    "left_arm_joint_states": (4, 8, 12, 16, 20, 22, 24),
    "left_ee_joint_states": (26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48),
    "right_arm_joint_states": (5, 9, 13, 17, 21, 23, 25),
    "right_ee_joint_states": (31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49),
}
SPARK_COMPONENTS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.contract-v2.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_parquet(path: Path, value: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.contract-v2.tmp")
    value.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            link_or_copy(path, target)
        else:
            raise ValueError(f"Unsupported dataset entry: {path}")


def only_parquet(root: Path) -> Path:
    paths = sorted(root.glob("chunk-*/*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one parquet below {root}, found {paths}")
    return paths[0]


def stack_column(rows: pd.DataFrame, column: str) -> np.ndarray:
    values = rows[column].to_numpy()
    if not len(values):
        raise ValueError(f"No values in {column}")
    return np.stack(values).astype(np.float32, copy=False)


def verify_raw_data(dataset: Path, benchmark: str, manifest: dict) -> dict:
    episode_path = only_parquet(dataset / "meta" / "episodes")
    data_path = only_parquet(dataset / "data")
    episodes = pd.read_parquet(episode_path)
    data = pd.read_parquet(
        data_path,
        columns=["episode_index", "frame_index", "index", "observation.state", "action"],
    )
    source_paths = manifest.get("episodes", [])
    if len(source_paths) != len(episodes):
        raise ValueError(
            f"Manifest has {len(source_paths)} episodes but metadata has {len(episodes)}"
        )

    action_digest = hashlib.sha256()
    state_digest = hashlib.sha256()
    total_frames = 0
    expected_global_index = 0

    for episode_index, (source_path, episode) in enumerate(
        zip(source_paths, episodes.to_dict("records"), strict=True)
    ):
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        length = int(episode["length"])
        if stop - start != length:
            raise ValueError(f"Episode {episode_index}: inconsistent metadata range")
        rows = data.iloc[start:stop]
        if len(rows) != length:
            raise ValueError(f"Episode {episode_index}: missing parquet rows")
        if not np.array_equal(rows["episode_index"].to_numpy(), np.full(length, episode_index)):
            raise ValueError(f"Episode {episode_index}: episode_index is not contiguous")
        if not np.array_equal(rows["frame_index"].to_numpy(), np.arange(length)):
            raise ValueError(f"Episode {episode_index}: frame_index is not contiguous")
        if not np.array_equal(
            rows["index"].to_numpy(), np.arange(expected_global_index, expected_global_index + length)
        ):
            raise ValueError(f"Episode {episode_index}: global index is not contiguous")

        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with h5py.File(source, "r") as raw:
            if benchmark == "egovla":
                raw_action_native = raw["action"][()]
                raw_state_native = raw["observations/qpos"][()]
                raw_action = np.concatenate(
                    [raw_action_native[:, list(indices)] for indices in EGO_INDICES.values()], axis=1
                ).astype(np.float32)
                raw_state = np.concatenate(
                    [raw_state_native[:, list(indices)] for indices in EGO_INDICES.values()], axis=1
                ).astype(np.float32)
                image_length = int(raw["observations/images/main"].shape[0])
            else:
                raw_action = np.concatenate(
                    [raw[f"action/{name}"][()] for name in SPARK_COMPONENTS], axis=1
                ).astype(np.float32)
                raw_state = np.concatenate(
                    [raw[f"state/{name}"][()] for name in SPARK_COMPONENTS], axis=1
                ).astype(np.float32)
                image_length = int(raw["vision/cam_head/colors"].shape[0])

        parquet_action = stack_column(rows, "action")
        parquet_state = stack_column(rows, "observation.state")
        if len(raw_action) != length or len(raw_state) != length or image_length != length:
            raise ValueError(
                f"Episode {episode_index}: raw lengths do not match metadata "
                f"(action={len(raw_action)}, state={len(raw_state)}, image={image_length}, meta={length})"
            )
        if not np.array_equal(parquet_action, raw_action):
            difference = float(np.max(np.abs(parquet_action - raw_action)))
            raise ValueError(
                f"Episode {episode_index}: parquet action is not raw action[t], max_abs={difference}"
            )
        if not np.array_equal(parquet_state, raw_state):
            difference = float(np.max(np.abs(parquet_state - raw_state)))
            raise ValueError(
                f"Episode {episode_index}: parquet state does not match raw state[t], max_abs={difference}"
            )

        action_digest.update(np.ascontiguousarray(parquet_action).tobytes())
        state_digest.update(np.ascontiguousarray(parquet_state).tobytes())
        total_frames += length
        expected_global_index += length
        if (episode_index + 1) % 100 == 0 or episode_index + 1 == len(episodes):
            print(
                f"[verify] {benchmark}: {episode_index + 1}/{len(episodes)} episodes, "
                f"{total_frames} frames",
                flush=True,
            )

    if total_frames != len(data):
        raise ValueError(f"Verified {total_frames} frames but parquet contains {len(data)}")
    return {
        "scope": "all_episodes_all_frames",
        "episodes": int(len(episodes)),
        "frames": int(total_frames),
        "action_values": int(total_frames * (38 if benchmark == "egovla" else 54)),
        "action_max_abs_error": 0.0,
        "action_sha256": action_digest.hexdigest(),
        "state_max_abs_error": 0.0,
        "state_sha256": state_digest.hexdigest(),
    }


def check_existing_videos(dataset: Path, manifest: dict, cameras: list[str]) -> None:
    episodes = pd.read_parquet(only_parquet(dataset / "meta" / "episodes"))
    info = load_json(dataset / "meta" / "info.json")
    pattern = info["video_path"]
    for episode_index, episode in enumerate(episodes.to_dict("records")):
        for camera in cameras:
            prefix = f"videos/{camera}"
            chunk = int(episode[f"{prefix}/chunk_index"])
            file_index = int(episode[f"{prefix}/file_index"])
            relative = pattern.format(
                video_key=camera, chunk_index=chunk, file_index=file_index
            )
            path = dataset / relative
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"Episode {episode_index}: missing video {path}")


def write_black_template(path: Path, frame_count: int, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (224, 224)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        for _ in range(frame_count):
            writer.write(frame)
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(path))
    try:
        decoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, decoded = capture.read()
    finally:
        capture.release()
    if decoded_count != frame_count or (width, height) != (224, 224):
        raise ValueError(
            f"Black template validation failed for {path}: "
            f"frames={decoded_count}/{frame_count}, size={width}x{height}"
        )
    if not ok or int(decoded.max()) != 0:
        raise ValueError(f"Black template is not decoded as exact zero: {path}")


def prepare_stage(input_path: Path, stage: Path, benchmark: str) -> None:
    (stage / "videos").mkdir(parents=True)
    link_tree(input_path / "data", stage / "data")
    shutil.copytree(input_path / "meta", stage / "meta", copy_function=shutil.copy2)
    existing_cameras = [CAMERA_KEYS[0]] if benchmark == "egovla" else CAMERA_KEYS
    for camera in existing_cameras:
        link_tree(input_path / "videos" / camera, stage / "videos" / camera)


def add_ego_black_wrists(stage: Path) -> None:
    episode_path = only_parquet(stage / "meta" / "episodes")
    episodes = pd.read_parquet(episode_path)
    info_path = stage / "meta" / "info.json"
    modality_path = stage / "meta" / "modality.json"
    info = load_json(info_path)
    modality = load_json(modality_path)
    fps = int(info["fps"])
    if fps != 30:
        raise ValueError(f"EgoVLA FPS must be 30, got {fps}")
    head_key = CAMERA_KEYS[0]
    head_prefix = f"videos/{head_key}"
    template_dir = stage / ".black-video-templates"
    templates: dict[int, Path] = {}

    for length in sorted({int(value) for value in episodes["length"]}):
        template = template_dir / f"black-{length:04d}.mp4"
        write_black_template(template, length, fps)
        templates[length] = template
        print(f"[black-video] encoded template for {length} frames", flush=True)

    for episode_index, episode in enumerate(episodes.to_dict("records")):
        length = int(episode["length"])
        chunk_index = int(episode[f"{head_prefix}/chunk_index"])
        file_index = int(episode[f"{head_prefix}/file_index"])
        from_timestamp = float(episode[f"{head_prefix}/from_timestamp"])
        for camera in CAMERA_KEYS[1:]:
            prefix = f"videos/{camera}"
            relative = info["video_path"].format(
                video_key=camera, chunk_index=chunk_index, file_index=file_index
            )
            link_or_copy(templates[length], stage / relative)
            episodes.loc[episode_index, f"{prefix}/from_timestamp"] = from_timestamp
            episodes.loc[episode_index, f"{prefix}/chunk_index"] = chunk_index
            episodes.loc[episode_index, f"{prefix}/file_index"] = file_index
        if (episode_index + 1) % 200 == 0 or episode_index + 1 == len(episodes):
            print(
                f"[black-video] linked {episode_index + 1}/{len(episodes)} episode pairs",
                flush=True,
            )
    shutil.rmtree(template_dir)

    # New columns initially pass through pandas' NaN-capable float dtype while
    # they are filled row by row. Restore the exact dtypes of the corresponding
    # head-camera columns before serializing the LeRobot episode metadata.
    for camera in CAMERA_KEYS[1:]:
        prefix = f"videos/{camera}"
        for suffix in ("from_timestamp", "chunk_index", "file_index"):
            head_column = f"{head_prefix}/{suffix}"
            episodes[f"{prefix}/{suffix}"] = episodes[f"{prefix}/{suffix}"].astype(
                episodes[head_column].dtype
            )

    head_feature = info["features"][head_key]
    if head_feature.get("shape") != [3, 224, 224]:
        raise ValueError(f"Unexpected head feature: {head_feature}")
    for camera in CAMERA_KEYS[1:]:
        info["features"][camera] = copy.deepcopy(head_feature)
    modality["video"] = {
        "cam_high": {"original_key": CAMERA_KEYS[0]},
        "cam_left_wrist": {"original_key": CAMERA_KEYS[1]},
        "cam_right_wrist": {"original_key": CAMERA_KEYS[2]},
    }
    atomic_parquet(episode_path, episodes)
    atomic_json(info_path, info)
    atomic_json(modality_path, modality)


def build_manifest(
    old_manifest: dict,
    benchmark: str,
    input_path: Path,
    verification: dict,
) -> dict:
    if benchmark == "egovla":
        kind = "ego"
        action_dim = 38
        source_keys = {
            "video": {
                CAMERA_KEYS[0]: "observations/images/main",
                CAMERA_KEYS[1]: "constant_black",
                CAMERA_KEYS[2]: "constant_black",
            },
            "state": "observations/qpos (38 selected indices)",
            "action": "action",
            "language": "task directory name",
        }
        selected_indices = {name: list(indices) for name, indices in EGO_INDICES.items()}
        black_camera_keys = CAMERA_KEYS[1:]
        image_decode = (
            "existing cam_high RGB MP4 retained; wrist streams are constant-black uint8 RGB"
        )
    else:
        kind = "spark"
        action_dim = 54
        source_keys = {
            "video": {
                CAMERA_KEYS[0]: "vision/cam_head/colors",
                CAMERA_KEYS[1]: "vision/cam_left_wrist/colors",
                CAMERA_KEYS[2]: "vision/cam_right_wrist/colors",
            },
            "state": [f"state/{name}" for name in SPARK_COMPONENTS],
            "action": [f"action/{name}" for name in SPARK_COMPONENTS],
            "language": "instruction",
        }
        selected_indices = None
        black_camera_keys = []
        image_decode = (
            "existing legacy SPark videos retained; decoded RGB order verified against raw HDF5"
        )

    manifest = copy.deepcopy(old_manifest)
    manifest.update(
        {
            "contract_version": 2,
            "kind": kind,
            "action_dim": action_dim,
            "camera_keys": CAMERA_KEYS,
            "source_keys": source_keys,
            "action_contract": {
                "source": "raw_hdf5_action",
                "selected_indices": selected_indices,
                "temporal_offset": 0,
                "derived_from_state": False,
            },
            "image_contract": {
                "height": 224,
                "width": 224,
                "dtype": "uint8",
                "color_order": "rgb",
                "black_camera_keys": black_camera_keys,
            },
            "image_decode": image_decode,
            "source_is_external": True,
            "upgrade_provenance": {
                "method": "reuse_existing_lerobot_dataset",
                "source_dataset": str(input_path),
                "existing_action_parquet_reused": True,
                "existing_real_videos_reencoded": False,
                "synthetic_black_wrist_videos_added": benchmark == "egovla",
                "verification": verification,
                "upgraded_at": datetime.now(timezone.utc).isoformat(),
            },
        }
    )
    return manifest


def validate_stage(stage: Path, benchmark: str) -> None:
    manifest = load_json(stage / "conversion_manifest.json")
    info = load_json(stage / "meta" / "info.json")
    modality = load_json(stage / "meta" / "modality.json")
    episodes = pd.read_parquet(only_parquet(stage / "meta" / "episodes"))
    if manifest.get("contract_version") != 2:
        raise ValueError("Staged manifest is not contract v2")
    if manifest.get("camera_keys") != CAMERA_KEYS:
        raise ValueError("Staged camera order is incorrect")
    if list(modality.get("video", {})) != ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        raise ValueError("Staged modality camera order is incorrect")
    for camera in CAMERA_KEYS:
        if info.get("features", {}).get(camera, {}).get("shape") != [3, 224, 224]:
            raise ValueError(f"Staged feature is invalid: {camera}")
        prefix = f"videos/{camera}"
        for suffix in ("from_timestamp", "chunk_index", "file_index"):
            if f"{prefix}/{suffix}" not in episodes:
                raise ValueError(f"Staged episode metadata lacks {prefix}/{suffix}")
    check_existing_videos(stage, manifest, CAMERA_KEYS)
    if benchmark == "egovla":
        for camera in CAMERA_KEYS[1:]:
            files = sorted((stage / "videos" / camera).glob("chunk-*/*.mp4"))
            if len(files) != len(episodes):
                raise ValueError(f"Expected {len(episodes)} {camera} videos, found {len(files)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("egovla", "spark"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if input_path == output_path:
        raise SystemExit("Refusing an in-place upgrade; choose a sibling output path")
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        manifest_path = output_path / "conversion_manifest.json"
        if manifest_path.is_file() and load_json(manifest_path).get("contract_version") == 2:
            print(f"[upgrade] output already contains contract v2: {output_path}")
            return
        raise FileExistsError(f"Refusing to overwrite incomplete output: {output_path}")

    old_manifest = load_json(input_path / "conversion_manifest.json")
    expected_kind = "ego" if args.benchmark == "egovla" else "spark"
    if old_manifest.get("kind") != expected_kind:
        raise ValueError(
            f"Dataset kind is {old_manifest.get('kind')!r}, expected {expected_kind!r}"
        )
    expected_cameras = [CAMERA_KEYS[0]] if args.benchmark == "egovla" else CAMERA_KEYS
    check_existing_videos(input_path, old_manifest, expected_cameras)
    verification = verify_raw_data(input_path, args.benchmark, old_manifest)
    print(f"[verify] raw action contract passed: {verification}", flush=True)

    stage = output_path.parent / f".{output_path.name}.contract-v2-stage-{os.getpid()}"
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    try:
        prepare_stage(input_path, stage, args.benchmark)
        if args.benchmark == "egovla":
            add_ego_black_wrists(stage)
        manifest = build_manifest(old_manifest, args.benchmark, input_path, verification)
        atomic_json(stage / "conversion_manifest.json", manifest)
        validate_stage(stage, args.benchmark)
        os.replace(stage, output_path)
    except Exception:
        print(f"[upgrade] failed; staging retained for inspection: {stage}", file=sys.stderr)
        raise
    print(f"[upgrade] published contract-v2 dataset: {output_path}")


if __name__ == "__main__":
    main()
