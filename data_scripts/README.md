# Dataset preparation

`convert_dataset.py` converts the two raw HDF5 layouts into LeRobot v3 datasets used by StarVLA.
The raw datasets stay on shared storage and are never copied. SParkArena reads the seven
`tianji_marvin_wuji` task directories and preserves three real camera streams. EgoVLA follows
the benchmark's single-view contract: it reads only `observations/images/main`, maps it to
`cam_high`, and creates constant-black left/right wrist streams. State uses `qpos[t]` and labels
use raw `action[t]` at the same timestep, with the canonical 38-DoF bimanual H1+Inspire split
(the remaining 12 lower-body joints are intentionally excluded).

Smoke checks (one episode, no training):

```bash
python data_scripts/convert_dataset.py spark --source data/raw_sources/SParkArena --output data/SParkArena/smoke --limit 1
python data_scripts/convert_dataset.py egovla --source data/raw_sources/EgoVLA --output data/EgoVLA/smoke --limit 1
```

Full conversion materializes compressed 224×224 videos and records action, camera, RGB, resolution,
and exact official instruction provenance in `conversion_manifest.json`. Use the same commands
without `--limit` for the final run.
