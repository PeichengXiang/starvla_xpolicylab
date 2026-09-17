#!/bin/bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]"
    echo "Example: bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0,1,2,3"
    exit 1
fi

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}
shift 6

for extra_arg in "$@"; do
    normalized_arg=${extra_arg}
    while [[ "${normalized_arg}" == -* ]]; do
        normalized_arg=${normalized_arg#-}
    done
    arg_key=${normalized_arg%%=*}
    if [[ -n "${arg_key}" && "config_yaml" == "${arg_key}"* ]]; then
        echo "[starVLA][ERROR] The generated config cannot be replaced: ${extra_arg}" >&2
        exit 2
    fi
    case "${normalized_arg}" in
        config-yaml|config-yaml=*|datasets|datasets=*|datasets.vla_data|datasets.vla_data=*|datasets.vla_data.data_root_dir*|datasets.vla_data.data_mix*|datasets.vla_data.dataset_py*|datasets.vla_data.action_*|datasets.vla_data.camera_names*|datasets.vla_data.black_camera_names*|datasets.vla_data.obs_image_size*|datasets.vla_data.image_size*|datasets.vla_data.default_image_resolution*|datasets.vla_data.include_state*|datasets.vla_data.xpolicylab_*|framework|framework=*|framework.action_model|framework.action_model=*|framework.action_model.action_dim*|framework.action_model.state_dim*|framework.action_model.action_horizon*|framework.action_model.future_action_window_size*|framework.action_model.past_action_window_size*)
            echo "[starVLA][ERROR] Dataset action/camera contracts cannot be overridden: ${extra_arg}" >&2
            exit 2
            ;;
    esac
done

case "${bench_name}:${env_cfg_type}" in
    SParkArena:tianji_marvin_wuji|spark:tianji_marvin_wuji|SparkArena:tianji_marvin_wuji|EgoVLA:ego_h1_inspire|egovla:ego_h1_inspire) ;;
    *) echo "[starVLA][ERROR] Use SParkArena/tianji_marvin_wuji or EgoVLA/ego_h1_inspire" >&2; exit 2;;
esac
if [[ "${action_type}" != "joint" ]]; then
    echo "[starVLA][ERROR] This HDF5 conversion uses joint-state action keys; action_type must be joint" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"
POLICY_PYTHON="${STARVLA_PYTHON:-${SCRIPT_DIR}/../../../.venv-starvla/bin/python}"
if [[ ! -x "${POLICY_PYTHON}" ]]; then POLICY_PYTHON="${STARVLA_PYTHON:-python}"; fi

base_config_yaml="${SCRIPT_DIR}/qwen_pi_v3.yaml"
data_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
run_id="${data_dir_name}-${seed}"
num_processes=$(awk -F',' '{print NF}' <<< "${gpu_id}")
data_root_dir="${STARVLA_DATA_ROOT:-${SCRIPT_DIR}/../../../data}"
base_vlm="${STARVLA_BASE_VLM:-${SCRIPT_DIR}/../../../pretrain_model/Qwen3-VL-4B-Instruct}"
if [[ ! -d "${data_root_dir}" ]]; then
    echo "[starVLA][ERROR] Data root does not exist: ${data_root_dir}" >&2
    exit 1
fi
# Training later changes cwd into source_starvla.  Resolve once so preflight
# and the loader cannot interpret the same relative data root differently.
data_root_dir="$(cd "${data_root_dir}" && pwd -P)"
if [[ ! -f "${base_vlm}/config.json" ]]; then
    echo "[starVLA][ERROR] Qwen3-VL-4B-Instruct is incomplete: ${base_vlm}" >&2
    echo "[starVLA][ERROR] Set STARVLA_BASE_VLM to a complete local checkpoint directory." >&2
    exit 1
fi
base_vlm="$(cd "${base_vlm}" && pwd -P)"
data_mix="${STARVLA_DATA_MIX:-xpolicylab_runtime}"
dataset_name="${STARVLA_XPOLICY_DATASET_NAME:-${data_dir_name}}"
config_yaml="${SCRIPT_DIR}/.generated/qwen_pi_v3_${run_id}.yaml"
dataset_path="${data_root_dir}/${dataset_name}"
task_instruction_path="${SCRIPT_DIR}/../../../data_scripts/egovla_task_instructions.json"
robot_type="xpolicylab_sparkarena"
if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then robot_type="xpolicylab_egovla"; fi

if [[ ! -f "${dataset_path}/meta/modality.json" && -f "${dataset_path}/${env_cfg_type}/meta/modality.json" ]]; then
    dataset_name="${dataset_name}/${env_cfg_type}"
    dataset_path="${data_root_dir}/${dataset_name}"
fi

if [[ ! -f "${dataset_path}/meta/modality.json" ]]; then
    echo "[starVLA][ERROR] LeRobot dataset not found or incomplete: ${dataset_path}" >&2
    echo "[starVLA][ERROR] expected ${dataset_path}/meta/modality.json" >&2
    echo "[starVLA][ERROR] Run process_data.sh first, or set STARVLA_DATA_ROOT/STARVLA_XPOLICY_DATASET_NAME." >&2
    exit 1
fi

"${POLICY_PYTHON}" - "${dataset_path}" "${robot_type}" "${task_instruction_path}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
robot_type = sys.argv[2]
task_instruction_path = Path(sys.argv[3])
manifest_path = dataset_path / "conversion_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"[starVLA][ERROR] Dataset has no source manifest: {manifest_path}. Re-run process_data.sh.")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
modality = json.loads((dataset_path / "meta/modality.json").read_text(encoding="utf-8"))
info = json.loads((dataset_path / "meta/info.json").read_text(encoding="utf-8"))
camera_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
if robot_type == "xpolicylab_egovla":
    expected_dim = 38
    expected_kind = "ego"
    expected_video = {
        camera_keys[0]: "observations/images/main",
        camera_keys[1]: "constant_black",
        camera_keys[2]: "constant_black",
    }
    expected_action = "action"
    expected_black = camera_keys[1:]
    expected_groups = [
        ("left_arm", 0, 7), ("left_ee", 7, 19),
        ("right_arm", 19, 26), ("right_ee", 26, 38),
    ]
    expected_indices = {
        "left_arm_joint_states": [4,8,12,16,20,22,24],
        "left_ee_joint_states": [26,36,27,37,28,38,29,39,30,40,46,48],
        "right_arm_joint_states": [5,9,13,17,21,23,25],
        "right_ee_joint_states": [31,41,32,42,33,43,34,44,35,45,47,49],
    }
    expected_instruction_mapping = json.loads(
        task_instruction_path.read_text(encoding="utf-8")
    )
    instruction_payload = json.dumps(
        expected_instruction_mapping,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    expected_instruction_hash = hashlib.sha256(instruction_payload).hexdigest()
else:
    expected_dim = 54
    expected_kind = "spark"
    expected_video = {
        camera_keys[0]: "vision/cam_head/colors",
        camera_keys[1]: "vision/cam_left_wrist/colors",
        camera_keys[2]: "vision/cam_right_wrist/colors",
    }
    expected_action = [
        "action/left_arm_joint_states", "action/left_ee_joint_states",
        "action/right_arm_joint_states", "action/right_ee_joint_states",
    ]
    expected_black = []
    expected_indices = None
    expected_instruction_mapping = None
    expected_instruction_hash = None
    expected_groups = [
        ("left_arm", 0, 7), ("left_ee", 7, 27),
        ("right_arm", 27, 34), ("right_ee", 34, 54),
    ]

errors = []
def require(condition, message):
    if not condition:
        errors.append(message)

require(manifest.get("contract_version") == 2, "contract_version must be 2")
require(manifest.get("kind") == expected_kind, f"kind must be {expected_kind!r}")
require(manifest.get("action_dim") == expected_dim, f"action_dim must be {expected_dim}")
require(manifest.get("camera_keys") == camera_keys, f"camera_keys must be {camera_keys!r}")
require(manifest.get("source_keys", {}).get("video") == expected_video, "camera provenance is stale")
require(manifest.get("source_keys", {}).get("action") == expected_action, "labels are not sourced from raw HDF5 action")
require(manifest.get("action_contract") == {
    "source": "raw_hdf5_action",
    "selected_indices": expected_indices,
    "temporal_offset": 0,
    "derived_from_state": False,
}, "action contract must use same-timestep raw action and must not derive labels from state")
if expected_indices is not None:
    actual_indices = manifest.get("action_contract", {}).get("selected_indices", {})
    require(list(actual_indices.items()) == list(expected_indices.items()),
            "selected action component order is stale")
image_contract = manifest.get("image_contract", {})
require(image_contract.get("height") == 224 and image_contract.get("width") == 224,
        "converted image resolution must be 224x224")
require(image_contract.get("dtype") == "uint8" and image_contract.get("color_order") == "rgb",
        "converted images must be uint8 RGB")
require(image_contract.get("black_camera_keys") == expected_black,
        f"black camera slots must be {expected_black!r}")
if expected_instruction_mapping is not None:
    instruction_contract = manifest.get("instruction_contract", {})
    require(instruction_contract.get("mapping") == expected_instruction_mapping,
            "task instructions differ from the official EgoVLA mapping")
    require(instruction_contract.get("mapping_sha256") == expected_instruction_hash,
            "task instruction mapping hash is stale")

expected_short_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
require(list(modality.get("video", {})) == expected_short_names,
        f"modality video order must be {expected_short_names!r}")
expected_group_names = [name for name, _, _ in expected_groups]
for root, original_key in (("state", "observation.state"), ("action", "action")):
    actual_groups = modality.get(root, {})
    require(list(actual_groups) == expected_group_names,
            f"{root} group order must be {expected_group_names!r}")
    for name, start, end in expected_groups:
        item = actual_groups.get(name, {})
        require(
            item.get("start") == start
            and item.get("end") == end
            and item.get("absolute") is True
            and item.get("dtype") == "float32"
            and item.get("original_key") == original_key,
            f"{root}.{name} must cover [{start}, {end}) from {original_key}",
        )
features = info.get("features", {})
require(features.get("action", {}).get("shape") == [expected_dim],
        f"parquet action feature must be {expected_dim}D")
for key in camera_keys:
    require(features.get(key, {}).get("shape") == [3, 224, 224],
            f"{key} must have shape [3, 224, 224]")

if errors:
    details = "\n  - ".join(errors)
    raise SystemExit(
        f"[starVLA][ERROR] Stale or incompatible converted dataset: {dataset_path}\n"
        f"  - {details}\n"
        "Re-run process_data.sh before training."
    )
print(f"[starVLA] dataset contract verified: raw action, cameras={camera_keys}, 224x224 RGB")
PY

mkdir -p "$(dirname "${config_yaml}")"
"${POLICY_PYTHON}" - "${base_config_yaml}" "${config_yaml}" "${data_root_dir}" "${data_mix}" "${run_id}" "${seed}" "${robot_type}" "${dataset_path}" "${base_vlm}" <<'PY'
import hashlib
import json
import os
import sys
import yaml

src, dst, data_root_dir, data_mix, run_id, seed, robot_type, dataset_path, base_vlm = sys.argv[1:10]
with open(src, "r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)
dataset_path = os.path.realpath(dataset_path)
with open(os.path.join(dataset_path, "conversion_manifest.json"), "r", encoding="utf-8") as fp:
    manifest = json.load(fp)
with open(os.path.join(dataset_path, "conversion_manifest.json"), "rb") as fp:
    conversion_manifest_sha256 = hashlib.sha256(fp.read()).hexdigest()
with open(os.path.join(dataset_path, "meta", "modality.json"), "r", encoding="utf-8") as fp:
    modality = json.load(fp)

cfg["run_id"] = run_id
cfg["seed"] = int(seed)
cfg["wandb_entity"] = os.environ.get("WANDB_ENTITY", cfg.get("wandb_entity"))
cfg["wandb_project"] = os.environ.get("WANDB_PROJECT", cfg.get("wandb_project"))
cfg["framework"]["qwenvl"]["base_vlm"] = base_vlm
vla_data_cfg = cfg.setdefault("datasets", {}).setdefault("vla_data", {})
vla_data_cfg["data_root_dir"] = data_root_dir
vla_data_cfg["data_mix"] = data_mix
vla_data_cfg["dataset_path"] = dataset_path
vla_data_cfg["conversion_manifest_sha256"] = conversion_manifest_sha256
vla_data_cfg["raw_dataset_manifest"] = manifest.get("raw_dataset_manifest")
vla_data_cfg["action_mode"] = "abs"
vla_data_cfg["action_source"] = "raw_hdf5_action"
vla_data_cfg["action_temporal_offset"] = 0
vla_data_cfg["action_derived_from_state"] = False
vla_data_cfg["include_state"] = True
vla_data_cfg["obs_image_size"] = [224, 224]
vla_data_cfg["camera_names"] = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
vla_data_cfg["black_camera_names"] = (
    ["cam_left_wrist", "cam_right_wrist"]
    if robot_type == "xpolicylab_egovla" else []
)
vla_data_cfg["instruction_mapping_sha256"] = manifest.get(
    "instruction_contract", {}
).get("mapping_sha256")
normalization_mode = vla_data_cfg.get("xpolicylab_normalization_mode")
if normalization_mode != "q99":
    raise SystemExit(f"[starVLA][ERROR] normalization_mode must be q99, got {normalization_mode!r}")

def ordered_entries(root):
    return [
        {
            "key": f"{root}.{name}",
            "dim": int(item["end"]) - int(item["start"]),
        }
        for name, item in modality[root].items()
    ]

state_entries = ordered_entries("state")
action_entries = ordered_entries("action")
selected_indices = manifest["action_contract"].get("selected_indices")
if selected_indices is None:
    source_action_component_indices = None
else:
    source_action_component_indices = [
        {"key": entry["key"], "indices": list(indices)}
        for entry, indices in zip(action_entries, selected_indices.values(), strict=True)
    ]
vla_data_cfg["xpolicylab_schema"] = {
    "version": 1,
    "robot_type": robot_type,
    "normalization_mode": normalization_mode,
    "state_dtype": "float16",
    "state": state_entries,
    "action": action_entries,
    "source_action_component_indices": source_action_component_indices,
    "source_action_keys": manifest["source_keys"]["action"],
}
dim = 38 if robot_type == "xpolicylab_egovla" else 54
cfg["framework"]["action_model"]["action_dim"] = dim
cfg["framework"]["action_model"]["state_dim"] = dim

with open(dst, "w", encoding="utf-8") as fp:
    yaml.safe_dump(cfg, fp, sort_keys=False)
PY

"${POLICY_PYTHON}" - \
    "${config_yaml}" \
    "${num_processes}" \
    "${ACCELERATE_GRADIENT_ACCUMULATION_STEPS:-1}" \
    "${STARVLA_EXPECTED_GLOBAL_BATCH_SIZE:-}" \
    "$@" <<'PY'
import sys

import yaml


config_path = sys.argv[1]
num_processes_raw = sys.argv[2]
accelerate_accumulation_raw = sys.argv[3]
expected_global_batch_raw = sys.argv[4]
extra_args = sys.argv[5:]


def positive_int(value, name):
    if isinstance(value, bool):
        raise SystemExit(f"[starVLA][ERROR] {name} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise SystemExit(
            f"[starVLA][ERROR] {name} must be a positive integer, got {value!r}"
        ) from None
    if parsed <= 0:
        raise SystemExit(f"[starVLA][ERROR] {name} must be positive, got {parsed}")
    return parsed


def normalize_dotlist_args(args):
    normalized = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif index + 1 < len(args) and not args[index + 1].startswith("--"):
                normalized.append(f"{key}={args[index + 1]}")
                index += 1
            else:
                normalized.append(f"{key}=true")
        index += 1
    return normalized


with open(config_path, "r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

per_device_batch = config["datasets"]["vla_data"]["per_device_batch_size"]
trainer_accumulation = config["trainer"]["gradient_accumulation_steps"]
for item in normalize_dotlist_args(extra_args):
    key, raw_value = item.split("=", 1)
    value = yaml.safe_load(raw_value)
    if key == "datasets.vla_data.per_device_batch_size":
        per_device_batch = value
    elif key == "trainer.gradient_accumulation_steps":
        trainer_accumulation = value

per_device_batch = positive_int(per_device_batch, "per-device batch size")
num_processes = positive_int(num_processes_raw, "number of training processes")
trainer_accumulation = positive_int(
    trainer_accumulation, "trainer.gradient_accumulation_steps"
)
accelerate_accumulation = positive_int(
    accelerate_accumulation_raw, "ACCELERATE_GRADIENT_ACCUMULATION_STEPS"
)
if trainer_accumulation != accelerate_accumulation:
    raise SystemExit(
        "[starVLA][ERROR] Gradient-accumulation contract failed: "
        f"trainer={trainer_accumulation}, Accelerate/DeepSpeed={accelerate_accumulation}. "
        "Set trainer.gradient_accumulation_steps and "
        "ACCELERATE_GRADIENT_ACCUMULATION_STEPS to the same value."
    )

global_batch = per_device_batch * num_processes * accelerate_accumulation
if expected_global_batch_raw:
    expected_global_batch = positive_int(
        expected_global_batch_raw, "STARVLA_EXPECTED_GLOBAL_BATCH_SIZE"
    )
    if global_batch != expected_global_batch:
        raise SystemExit(
            "[starVLA][ERROR] Global batch-size contract failed: "
            f"per_device={per_device_batch}, processes={num_processes}, "
            f"gradient_accumulation={accelerate_accumulation}, "
            f"resolved={global_batch}, expected={expected_global_batch}"
        )

print(
    "[starVLA] global batch contract verified: "
    f"{per_device_batch} x {num_processes} x {accelerate_accumulation} = {global_batch}"
)
PY

echo "[starVLA] config_yaml=${config_yaml}"
echo "[starVLA] run_id=${run_id}"
echo "[starVLA] seed=${seed}"
echo "[starVLA] data_root_dir=${data_root_dir}"
echo "[starVLA] data_mix=${data_mix}, dataset=${dataset_name}, dataset_path=${dataset_path}"
echo "[starVLA] base_vlm=${base_vlm}"
echo "[starVLA] train_entry=starVLA/training/train_starvla.py"
echo "[starVLA] num_processes=${num_processes}, mixed_precision=bf16"

if [[ "${STARVLA_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[starVLA] preflight-only validation complete; training was not started"
    exit 0
fi

cd "${STARVLA_ROOT}"
PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
STARVLA_XPOLICY_DATASET_NAME="${dataset_name}" \
STARVLA_XPOLICY_DATA_MIX="${data_mix}" \
STARVLA_XPOLICY_ROBOT_TYPE="${robot_type}" \
WANDB_MODE="${WANDB_MODE:-online}" \
NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" \
CUDA_VISIBLE_DEVICES="${gpu_id}" "${POLICY_PYTHON}" -m accelerate.commands.accelerate_cli launch \
    --num_processes "${num_processes}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --run_id "${run_id}" \
    --seed "${seed}" \
    "$@"
