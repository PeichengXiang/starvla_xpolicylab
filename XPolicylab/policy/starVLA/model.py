from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)
from .runtime_config import (
    parse_bool,
    resolve_checkpoint_framework,
    resolve_include_state,
    validate_server_runtime_contract,
)


_CUR_DIR = Path(__file__).resolve().parent


def _egovla_task_instruction_contract() -> tuple[dict[str, str], str]:
    path = _CUR_DIR.parents[2] / "data_scripts" / "egovla_task_instructions.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(
        mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return mapping, hashlib.sha256(payload).hexdigest()


def _expected_xpolicylab_schema(env_cfg_type: str) -> dict[str, Any]:
    if env_cfg_type == "ego_h1_inspire":
        robot_type = "xpolicylab_egovla"
        dims = [7, 12, 7, 12]
        source_action_keys: Any = "action"
        source_indices = [
            [4, 8, 12, 16, 20, 22, 24],
            [26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48],
            [5, 9, 13, 17, 21, 23, 25],
            [31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49],
        ]
    elif env_cfg_type == "tianji_marvin_wuji":
        robot_type = "xpolicylab_sparkarena"
        dims = [7, 20, 7, 20]
        source_action_keys = [
            "action/left_arm_joint_states",
            "action/left_ee_joint_states",
            "action/right_arm_joint_states",
            "action/right_ee_joint_states",
        ]
        source_indices = None
    else:
        raise ValueError(f"No XPolicy data schema for env_cfg_type={env_cfg_type!r}.")

    names = ["left_arm", "left_ee", "right_arm", "right_ee"]
    state_entries = [
        {"key": f"state.{name}", "dim": dim}
        for name, dim in zip(names, dims)
    ]
    action_entries = [
        {"key": f"action.{name}", "dim": dim}
        for name, dim in zip(names, dims)
    ]
    component_indices = (
        None
        if source_indices is None
        else [
            {"key": entry["key"], "indices": indices}
            for entry, indices in zip(action_entries, source_indices)
        ]
    )
    return {
        "xpolicylab_schema": {
            "version": 1,
            "robot_type": robot_type,
            "normalization_mode": "q99",
            "state_dtype": "float16",
            "state": state_entries,
            "action": action_entries,
            "source_action_component_indices": component_indices,
            "source_action_keys": source_action_keys,
        },
    }


def _optional_path(value: str | None, *base_dirs: Path) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    for base_dir in base_dirs:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return base_dirs[0] / path


def _decode_image(image: Any) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _extract_camera(observation: dict[str, Any], camera_names: list[str]) -> np.ndarray:
    vision = observation.get("vision", {})
    for camera_name in camera_names:
        if camera_name not in vision:
            continue
        camera_obs = vision[camera_name]
        if isinstance(camera_obs, dict):
            for image_key in ("color", "rgb", "colors"):
                if image_key in camera_obs:
                    return _decode_image(camera_obs[image_key])
        else:
            return _decode_image(camera_obs)
    raise KeyError(f"Missing camera from candidates: {camera_names}")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("starVLA currently supports action_type='joint' first.")

        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.env_cfg_type is None:
            raise ValueError("starVLA requires env_cfg_type.")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )

        starvla_root = _optional_path(
            self.model_cfg.get("starvla_root"),
            _CUR_DIR,
        ) or (_CUR_DIR / "source_starvla")
        if str(starvla_root) not in sys.path:
            sys.path.insert(0, str(starvla_root))

        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(
            self.model_cfg.get("starvla_server_host", "127.0.0.1"),
            int(self.model_cfg.get("starvla_server_port", 5694)),
        )
        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = int(server_meta["action_chunk_size"])
        execute_horizon = self.model_cfg.get("execute_horizon", self.action_chunk_size)
        if execute_horizon in (None, "", "null", "None"):
            execute_horizon = self.action_chunk_size
        self.execute_horizon = int(execute_horizon)
        if self.execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {self.execute_horizon}.")
        if self.execute_horizon > self.action_chunk_size:
            raise ValueError(
                f"execute_horizon={self.execute_horizon} exceeds action_chunk_size={self.action_chunk_size}."
            )
        self.unnorm_key = self.model_cfg.get("unnorm_key", "arx_x5")
        if self.unnorm_key in (None, "", "null", "None", "auto"):
            self.unnorm_key = None
        self.use_ddim = bool(self.model_cfg.get("use_ddim", True))
        self.num_ddim_steps = int(self.model_cfg.get("num_ddim_steps", 10))
        self.image_size = tuple(int(value) for value in self.model_cfg.get("image_size", [224, 224]))
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError(f"image_size must be [height, width], got {self.image_size!r}.")
        self.camera_names = list(
            self.model_cfg.get(
                "camera_names",
                ["cam_head", "cam_left_wrist", "cam_right_wrist"],
            )
        )
        default_black_cameras = (
            ["cam_left_wrist", "cam_right_wrist"]
            if self.env_cfg_type == "ego_h1_inspire" else []
        )
        self.black_camera_names = list(
            self.model_cfg.get("black_camera_names", default_black_cameras)
        )
        unknown_black_cameras = set(self.black_camera_names) - set(self.camera_names)
        if unknown_black_cameras:
            raise ValueError(
                f"black_camera_names must be a subset of camera_names; got {sorted(unknown_black_cameras)}."
            )
        self.egovla_task_instructions = None
        self.instruction_mapping_sha256 = None
        if self.env_cfg_type == "ego_h1_inspire":
            (
                self.egovla_task_instructions,
                self.instruction_mapping_sha256,
            ) = _egovla_task_instruction_contract()
        self.include_state = resolve_include_state(
            self.model_cfg.get("include_state", "auto"),
            self.model_cfg.get("checkpoint_path"),
        )
        self.require_runtime_contract = parse_bool(
            self.model_cfg.get("require_runtime_contract", True)
        )
        if self.require_runtime_contract:
            expected_framework = resolve_checkpoint_framework(
                self.model_cfg.get("checkpoint_path")
            )
            expected_pi_v3_forward = self.model_cfg.get("required_pi_v3_forward")
            if expected_pi_v3_forward in (None, "", "auto", "none", "null", "None"):
                expected_pi_v3_forward = None
            else:
                expected_pi_v3_forward = str(expected_pi_v3_forward)
            expected_data_contract = None
            if self.env_cfg_type in {"ego_h1_inspire", "tianji_marvin_wuji"}:
                expected_data_contract = {
                    "action_mode": "abs",
                    "action_source": "raw_hdf5_action",
                    "action_temporal_offset": 0,
                    "action_derived_from_state": False,
                    "camera_names": self.camera_names,
                    "black_camera_names": self.black_camera_names,
                    "image_size": list(self.image_size),
                    "include_state": self.include_state,
                    "instruction_mapping_sha256": self.instruction_mapping_sha256,
                    **_expected_xpolicylab_schema(self.env_cfg_type),
                }
            validate_server_runtime_contract(
                server_meta,
                include_state=self.include_state,
                action_dim=self.action_dim,
                unnorm_key=self.unnorm_key,
                expected_framework=expected_framework,
                expected_pi_v3_forward=expected_pi_v3_forward,
                expected_data_contract=expected_data_contract,
            )

        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.action_chunks_by_env: dict[int, np.ndarray] = {}
        self.step_by_env: dict[int, int] = {}
        self._latest_env_idx_list = [0]

        print(
            f"[starVLA] connected to StarVLA server, action_dim={self.action_dim}, "
            f"chunk={self.action_chunk_size}, execute_horizon={self.execute_horizon}, "
            f"include_state={self.include_state}, cameras={self.camera_names}, "
            f"black_cameras={self.black_camera_names}, image_size={self.image_size}, "
            f"action_order=xpolicy, metadata={server_meta}"
        )

    def _convert_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        aliases = {"cam_head": ["cam_head", "head_camera"], "cam_left_wrist": ["cam_left_wrist", "left_camera"], "cam_right_wrist": ["cam_right_wrist", "right_camera"]}
        height, width = self.image_size
        images = []
        for camera_name in self.camera_names:
            if camera_name in self.black_camera_names:
                image = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                image = _extract_camera(observation, aliases.get(camera_name, [camera_name]))
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            images.append(image)

        instruction = observation.get("instruction") or observation.get("instructions")
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        if instruction in (None, ""):
            instruction = self.model_cfg.get("task_name", "")
        if self.egovla_task_instructions is not None:
            expected_instructions = set(self.egovla_task_instructions.values())
            if str(instruction) not in expected_instructions:
                raise ValueError(
                    "EgoVLA instruction does not exactly match the official registry: "
                    f"{instruction!r}"
                )

        converted_obs = {
            "lang": str(instruction),
            "image": images,
        }
        if self.include_state:
            state = pack_robot_state(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            ).astype(np.float32)
            if state.ndim == 1:
                state = state[None, :]
            if state.ndim != 2 or state.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected state shape (T, {self.action_dim}), got {state.shape}."
                )
            converted_obs["state"] = state

        return converted_obs

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._latest_env_idx_list.append(env_idx)
            self.obs_by_env[env_idx] = self._convert_obs(obs)

    def _infer_chunk(self, env_idx: int) -> np.ndarray:
        if env_idx not in self.obs_by_env:
            raise AssertionError("update_obs must be called before get_action.")

        vla_input = {
            "examples": [self.obs_by_env[env_idx]],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        if self.unnorm_key is not None:
            vla_input["unnorm_key"] = self.unnorm_key
        response = self.client.predict_action(vla_input)
        if not response.get("ok", False):
            raise RuntimeError(f"StarVLA inference failed: {response.get('error', response)}")
        return np.asarray(response["data"]["actions"][0], dtype=np.float32)

    def _next_action_vector(self, env_idx: int) -> np.ndarray:
        step = self.step_by_env.get(env_idx, 0)
        chunk = self.action_chunks_by_env.get(env_idx)
        if chunk is None or step % self.execute_horizon == 0:
            chunk = self._infer_chunk(env_idx)
            self.action_chunks_by_env[env_idx] = chunk

        action_idx = min(step % self.execute_horizon, len(chunk) - 1)
        self.step_by_env[env_idx] = step + 1
        action = np.asarray(chunk[action_idx], dtype=np.float32)
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[-1]}.")
        return action

    def get_action(self):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        return [
            [
                unpack_robot_state(
                    self._next_action_vector(int(env_idx)),
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            ]
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.obs_by_env.clear()
        self.action_chunks_by_env.clear()
        self.step_by_env.clear()
        self._latest_env_idx_list = [0]
