"""Runtime data registry for the two XPolicyLab StarVLA datasets."""
from __future__ import annotations
import os
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform

class RuntimeDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    normalization_mode = "q99"
    observation_indices = [0]
    action_indices = list(range(16))
    def __init__(
        self,
        dim,
        cameras,
        groups,
        source_action_keys,
        source_action_component_indices=None,
    ):
        self.video_keys = [f"video.{x}" for x in cameras]
        self.groups = groups
        self.state_keys = [f"state.{name}" for name, _, _ in groups]
        self.action_keys = [f"action.{name}" for name, _, _ in groups]
        self.state_key_dims = {f"state.{name}": e-s for name,s,e in groups}
        self.action_key_dims = {f"action.{name}": e-s for name,s,e in groups}
        self.modality_key_ranges = {"state": {name:(s,e) for name,s,e in groups}, "action": {name:(s,e) for name,s,e in groups}}
        self.language_keys = ["annotation.human.action.task_description"]
        self.source_action_keys = source_action_keys
        self.source_action_component_indices = (
            None
            if source_action_component_indices is None
            else [
                {"key": key, "indices": list(indices)}
                for key, indices in zip(
                    self.action_keys, source_action_component_indices, strict=True
                )
            ]
        )
    def modality_config(self):
        return {"video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys), "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys), "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys), "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys)}
    def transform(self):
        transforms=[]
        for root, keys in (("state",self.state_keys),("action",self.action_keys)):
            transforms += [StateActionToTensor(apply_to=keys), StateActionTransform(apply_to=keys, normalization_modes={k:self.normalization_mode for k in keys})]
        return ComposedModalityTransform(transforms=transforms)

SPARK = RuntimeDataConfig(
    54,
    ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    [("left_arm",0,7),("left_ee",7,27),("right_arm",27,34),("right_ee",34,54)],
    [
        "action/left_arm_joint_states", "action/left_ee_joint_states",
        "action/right_arm_joint_states", "action/right_ee_joint_states",
    ],
)
# EgoVLA is physically single-view.  Keep the model's fixed three-slot image
# contract by reading the two all-black wrist streams created by the converter.
EGO = RuntimeDataConfig(
    38,
    ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    [("left_arm",0,7),("left_ee",7,19),("right_arm",19,26),("right_ee",26,38)],
    "action",
    [
        [4,8,12,16,20,22,24],
        [26,36,27,37,28,38,29,39,30,40,46,48],
        [5,9,13,17,21,23,25],
        [31,41,32,42,33,43,34,44,35,45,47,49],
    ],
)
ROBOT_TYPE_CONFIG_MAP = {"xpolicylab_sparkarena": SPARK, "xpolicylab_egovla": EGO}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
_dataset = os.environ.get("STARVLA_XPOLICY_DATASET_NAME")
_robot = os.environ.get("STARVLA_XPOLICY_ROBOT_TYPE", "xpolicylab_sparkarena")
if _dataset:
    DATASET_NAMED_MIXTURES = {os.environ.get("STARVLA_XPOLICY_DATA_MIX", "xpolicylab_runtime"): [(_dataset, 1.0, _robot)]}
else:
    DATASET_NAMED_MIXTURES = {}
