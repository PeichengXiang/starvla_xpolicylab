from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
XPL_ROOT = THIS_DIR.parents[2]
WORKSPACE_ROOT = XPL_ROOT.parent
if str(XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(XPL_ROOT))


def _load_converter():
    path = WORKSPACE_ROOT / "data_scripts" / "convert_dataset.py"
    spec = importlib.util.spec_from_file_location("xpolicy_starvla_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load converter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_upgrader():
    path = WORKSPACE_ROOT / "data_scripts" / "upgrade_existing_dataset.py"
    spec = importlib.util.spec_from_file_location("xpolicy_starvla_upgrader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dataset upgrader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EgoVLAConversionContractTest(unittest.TestCase):
    def test_raw_action_and_black_wrist_contract(self):
        converter = _load_converter()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "raw" / "Close-Drawer"
            task_dir.mkdir(parents=True)
            source_path = task_dir / "episode_0.hdf5"
            (task_dir.parent / "DATASET_MANIFEST.json").write_text(
                '{"synthetic": true}\n', encoding="utf-8"
            )

            steps = 4
            qpos = np.arange(steps * 50, dtype=np.float32).reshape(steps, 50)
            action = qpos + 10000.0
            images = np.full((steps, 384, 384, 3), 127, dtype=np.uint8)
            with h5py.File(source_path, "w") as stream:
                stream.create_dataset("observations/qpos", data=qpos)
                stream.create_dataset("observations/images/main", data=images)
                stream.create_dataset("action", data=action)

            output = root / "converted"
            converter.convert_ego(task_dir.parent, output)

            frame_table = pd.read_parquet(output / "data/chunk-000/file-000.parquet")
            flat_indices = [
                index
                for indices in converter.EGO_INDICES.values()
                for index in indices
            ]
            expected_action = action[:, flat_indices]
            converted_action = np.stack(frame_table["action"].to_numpy())
            np.testing.assert_array_equal(converted_action, expected_action)
            self.assertFalse(np.array_equal(converted_action[:-1], qpos[1:, flat_indices]))

            manifest = json.loads((output / "conversion_manifest.json").read_text())
            self.assertEqual(manifest["source_keys"]["action"], "action")
            self.assertEqual(manifest["action_contract"]["source"], "raw_hdf5_action")
            self.assertEqual(manifest["action_contract"]["temporal_offset"], 0)
            self.assertIs(manifest["action_contract"]["derived_from_state"], False)
            self.assertEqual(
                manifest["image_contract"]["black_camera_keys"],
                converter.CAMERA_KEYS[1:],
            )
            instruction_contract = manifest["instruction_contract"]
            self.assertEqual(instruction_contract["mapping"], converter.EGO_TASK_INSTRUCTIONS)
            self.assertEqual(
                instruction_contract["mapping_sha256"],
                converter.mapping_sha256(converter.EGO_TASK_INSTRUCTIONS),
            )

            for camera_key in converter.CAMERA_KEYS:
                video_path = output / "videos" / camera_key / "chunk-000/file-000.mp4"
                capture = cv2.VideoCapture(str(video_path))
                ok, frame = capture.read()
                capture.release()
                self.assertTrue(ok, camera_key)
                self.assertEqual(frame.shape, (224, 224, 3))
                if camera_key != converter.CAMERA_KEYS[0]:
                    self.assertEqual(int(frame.max()), 0)

    def test_upgrade_manifest_uses_the_same_canonical_prompt_hash(self):
        upgrader = _load_upgrader()
        manifest = upgrader.build_manifest(
            {}, "egovla", Path("/source/dataset"), {"scope": "test"}
        )
        instruction_contract = upgrader.ego_instruction_contract()
        self.assertEqual(manifest["instruction_contract"], instruction_contract)

        from XPolicyLab.policy.starVLA.model import _egovla_task_instruction_contract

        runtime_mapping, runtime_hash = _egovla_task_instruction_contract()
        self.assertEqual(instruction_contract["mapping"], runtime_mapping)
        self.assertEqual(instruction_contract["mapping_sha256"], runtime_hash)


class EgoVLAInferenceContractTest(unittest.TestCase):
    def test_live_wrist_images_are_replaced_with_black(self):
        from XPolicyLab.policy.starVLA.model import Model

        model = Model.__new__(Model)
        model.camera_names = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
        model.black_camera_names = ["cam_left_wrist", "cam_right_wrist"]
        model.image_size = (224, 224)
        model.model_cfg = {}
        model.include_state = False

        observation = {
            "instruction": "close the drawer",
            "vision": {
                "cam_head": {"color": np.full((384, 384, 3), 127, dtype=np.uint8)},
                "cam_left_wrist": {"color": np.full((384, 384, 3), 255, dtype=np.uint8)},
            },
        }
        converted = model._convert_obs(observation)
        self.assertEqual([image.shape for image in converted["image"]], [(224, 224, 3)] * 3)
        self.assertGreater(int(converted["image"][0].max()), 0)
        self.assertEqual(int(converted["image"][1].max()), 0)
        self.assertEqual(int(converted["image"][2].max()), 0)

    def test_server_metadata_must_match_training_data_contract(self):
        from XPolicyLab.policy.starVLA.model import (
            _egovla_task_instruction_contract,
            _expected_xpolicylab_schema,
        )
        from XPolicyLab.policy.starVLA.runtime_config import (
            validate_server_runtime_contract,
        )

        _, instruction_mapping_sha256 = _egovla_task_instruction_contract()
        data_contract = {
            "action_mode": "abs",
            "action_source": "raw_hdf5_action",
            "action_temporal_offset": 0,
            "action_derived_from_state": False,
            "camera_names": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
            "black_camera_names": ["cam_left_wrist", "cam_right_wrist"],
            "image_size": [224, 224],
            "include_state": True,
            "instruction_mapping_sha256": instruction_mapping_sha256,
            **_expected_xpolicylab_schema("ego_h1_inspire"),
        }
        metadata = {
            "runtime_contract": {
                "version": 1,
                "image_color_order": "rgb",
                "state_input": "raw_env",
                "state_normalization": "training_transform",
                "action_output": "unnormalized_env",
                "action_dim": 38,
            },
            "training_data_contract": dict(data_contract),
            "available_unnorm_keys": ["new_embodiment"],
        }
        validate_server_runtime_contract(
            metadata,
            include_state=True,
            action_dim=38,
            unnorm_key=None,
            expected_data_contract=data_contract,
        )

        metadata["training_data_contract"]["image_size"] = [256, 256]
        with self.assertRaisesRegex(ValueError, "image_size"):
            validate_server_runtime_contract(
                metadata,
                include_state=True,
                action_dim=38,
                unnorm_key=None,
                expected_data_contract=data_contract,
            )

        metadata["training_data_contract"] = dict(data_contract)
        metadata["training_data_contract"]["xpolicylab_schema"] = dict(
            data_contract["xpolicylab_schema"]
        )
        metadata["training_data_contract"]["xpolicylab_schema"]["action"] = list(
            reversed(data_contract["xpolicylab_schema"]["action"])
        )
        with self.assertRaisesRegex(ValueError, "xpolicylab_schema"):
            validate_server_runtime_contract(
                metadata,
                include_state=True,
                action_dim=38,
                unnorm_key=None,
                expected_data_contract=data_contract,
            )

        metadata["training_data_contract"] = dict(data_contract)
        metadata["training_data_contract"]["instruction_mapping_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "instruction_mapping_sha256"):
            validate_server_runtime_contract(
                metadata,
                include_state=True,
                action_dim=38,
                unnorm_key=None,
                expected_data_contract=data_contract,
            )


if __name__ == "__main__":
    unittest.main()
