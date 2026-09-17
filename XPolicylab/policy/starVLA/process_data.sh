#!/bin/bash
set -euo pipefail
if [[ $# -lt 4 ]]; then echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2; exit 1; fi
bench_name=$1; ckpt_name=$2; env_cfg_type=$3; action_type=$4; limit=${5:-}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
POLICY_PYTHON="${STARVLA_PYTHON:-${ROOT_DIR}/.venv-starvla/bin/python}"
if [[ ! -x "${POLICY_PYTHON}" ]]; then POLICY_PYTHON="${STARVLA_PYTHON:-python}"; fi
case "$bench_name" in
  SParkArena|spark|SparkArena)
    [[ "$env_cfg_type" == "tianji_marvin_wuji" ]] || { echo "SParkArena requires env_cfg_type=tianji_marvin_wuji" >&2; exit 2; }
    mode=spark; source="$ROOT_DIR/data/raw_sources/SParkArena";;
  EgoVLA|egovla)
    [[ "$env_cfg_type" == "ego_h1_inspire" ]] || { echo "EgoVLA requires env_cfg_type=ego_h1_inspire" >&2; exit 2; }
    mode=egovla; source="${STARVLA_EGOVLA_RAW_ROOT:-$ROOT_DIR/data/raw_sources/EgoVLA}";;
  *) echo "Unsupported benchmark: $bench_name (use SParkArena or EgoVLA)" >&2; exit 2;;
esac
[[ "$action_type" == "joint" ]] || { echo "This converter reads HDF5 joint-state action keys; use action_type=joint" >&2; exit 2; }
out="$ROOT_DIR/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
args=("$mode" --source "$source" --output "$out")
[[ -n "$limit" ]] && args+=(--limit "$limit")
exec "${POLICY_PYTHON}" "$ROOT_DIR/data_scripts/convert_dataset.py" "${args[@]}"
