# starvla_xpolicylab

StarVLA adapted to XPolicyLab for SParkArena and EgoVLA training and inference.

## EgoVLA contract

- Labels are the same-timestep values from the raw HDF5 `action[t]`; they are never derived from `qpos[t+1]`.
- H1 Inspire state and action are 38D in the order left arm (7), left hand (12), right arm (7), right hand (12).
- EgoVLA uses the real `observations/images/main` RGB view plus two constant-black wrist slots.
- Training and inference both use three 224×224 uint8 RGB slots in the order head, left wrist, right wrist.
- The canonical task instruction is preserved per task. Training uses absolute actions, q99 normalization, and a 16-step horizon.

## Repository layout

- `XPolicylab/policy/starVLA/`: policy adapter, trainer entry points, inference contract, and vendored StarVLA source.
- `data_scripts/`: raw HDF5 to LeRobot v3 conversion and contract validation.
- `launch_egovla_h20.sh`: eight-GPU H20 launcher with global batch size fixed to 64.

Large artifacts are intentionally excluded from Git: raw/converted data, pretrained weights, checkpoints, virtual environments, logs, and W&B credentials.

## H20 preparation

```bash
ln -s /path/to/python-env .venv-starvla
mkdir -p data/raw_sources pretrain_model
ln -s /path/to/EgoVLA_raw_remove_deprecated data/raw_sources/EgoVLA

STARVLA_EGOVLA_RAW_ROOT=/path/to/EgoVLA_raw_remove_deprecated \
  bash XPolicylab/policy/starVLA/process_data.sh \
  EgoVLA full_v2 ego_h1_inspire joint

./launch_egovla_h20.sh
```

The launcher requires online W&B logging but never stores the API key in this repository.
