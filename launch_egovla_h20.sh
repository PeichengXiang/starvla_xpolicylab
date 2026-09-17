#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="${STARVLA_CACHE_ROOT:-${ROOT_DIR}/.cache}"

mkdir -p \
  "${CACHE_ROOT}/huggingface" \
  "${CACHE_ROOT}/torch_extensions" \
  "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/wandb" \
  "${CACHE_ROOT}/tmp"

export STARVLA_PYTHON="${STARVLA_PYTHON:-${ROOT_DIR}/.venv-starvla/bin/python}"
export STARVLA_DATA_ROOT="${STARVLA_DATA_ROOT:-${ROOT_DIR}/data}"
export STARVLA_BASE_VLM="${STARVLA_BASE_VLM:-${ROOT_DIR}/pretrain_model/Qwen3-VL-4B-Instruct}"
export CUDA_HOME="${CUDA_HOME:-/vepfs-cnbje63de6fae220/wenwei/deps/cuda-12.1}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export WANDB_MODE=online
export WANDB_REQUIRE_ONLINE=1
export WANDB_ENTITY="${WANDB_ENTITY:-peichengxiang773-hkust}"
export WANDB_PROJECT="${WANDB_PROJECT:-starvla-xpolicylab}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=4
export ACCELERATE_DEEPSPEED_ZERO_STAGE=2
unset ACCELERATE_DEEPSPEED_ZERO3_INIT ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL
export STARVLA_EXPECTED_GLOBAL_BATCH_SIZE=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

exec bash "${ROOT_DIR}/XPolicylab/policy/starVLA/train.sh" \
  EgoVLA full_v2 ego_h1_inspire joint 0 \
  0,1,2,3 \
  --datasets.vla_data.per_device_batch_size=4 \
  --trainer.max_train_steps=80000 \
  --trainer.save_interval=10000 \
  --trainer.eval_interval=10000 \
  --trainer.gradient_accumulation_steps=4
