#!/bin/bash
set -e

ROOT_DIR="$1"
env_cfg_type="$2"

python3 -c '
import sys, os, json

root_dir = sys.argv[1]
env_cfg_type = sys.argv[2]

robot_info_path = None
for dirname in ("XPolicyLab", "XPolicylab"):
    candidate = os.path.join(root_dir, dirname, "utils", "robot", "_robot_info.json")
    if os.path.isfile(candidate):
        robot_info_path = candidate
        break
if robot_info_path is None:
    raise FileNotFoundError(
        "robot info not found under XPolicyLab or XPolicylab: " + root_dir
    )
robot_action_dim_info = json.load(
    open(robot_info_path, "r", encoding="utf-8")
)[env_cfg_type]

print(sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"]))
' "${ROOT_DIR}" "${env_cfg_type}"