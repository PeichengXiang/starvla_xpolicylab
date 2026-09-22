from __future__ import annotations
import argparse, hashlib, json, multiprocessing, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, h5py, numpy as np, pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
XPL_DIR = ROOT_DIR / 'XPolicylab'
if str(XPL_DIR) not in sys.path:
    sys.path.insert(0, str(XPL_DIR))

from XPolicyLab.utils.process_data import decode_image_bit

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CAMERA_KEYS = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
]

EGO_TASK_INSTRUCTIONS = json.loads(
    Path(__file__).with_name('egovla_task_instructions.json').read_text(encoding='utf-8')
)
SPARK_REAL_TASK_INSTRUCTIONS = json.loads(
    Path(__file__).with_name('spark_real_bench_v5_task_instructions.json').read_text(
        encoding='utf-8'
    )
)
SPARK_KINDS = {'spark', 'spark_real_bench_v5'}

def mapping_sha256(mapping):
    payload=json.dumps(mapping,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def path_lexists(path):
    return os.path.lexists(path)

def stage_path(out):
    return out.parent / f'.{out.name}.staging'

def file_sha256(path, chunk_size=16 * 1024 * 1024):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def raw_dataset_manifest_provenance(source):
    path=Path(source) / 'DATASET_MANIFEST.json'
    if not path.is_file():
        raise FileNotFoundError(f'EgoVLA source manifest not found: {path}')
    before=path.stat()
    sha256=file_sha256(path)
    after=path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'EgoVLA source manifest changed while hashing: {path}')
    return {
        'path': str(path.resolve()),
        'size_bytes': int(after.st_size),
        'sha256': sha256,
    }

def spark_real_manifest_provenance(source, episode_count, limited):
    path=Path(source) / 'conversion_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'SParkRealBenchV5 source manifest not found: {path}')
    before=path.stat(); raw=path.read_bytes(); after=path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'SParkRealBenchV5 source manifest changed while reading: {path}')
    payload=json.loads(raw)
    expected_policy=(
        'hand_master_aligned_causal_then_camera_keep;',
        'arm_future_state:1_after_camera_keep_hold_last',
    )
    action_policy=str(payload.get('action_policy',''))
    errors=[]
    if payload.get('schema') != 'spark0_raw_directory_canonical_manifest_v1':
        errors.append('unexpected source manifest schema')
    if payload.get('task_instructions') != SPARK_REAL_TASK_INSTRUCTIONS:
        errors.append('task instruction mapping differs from SParkRealBenchV5 registry')
    if any(token not in action_policy for token in expected_policy):
        errors.append('action_policy does not describe master-hand actions plus arm next-state labels')
    if not limited and int(payload.get('episode_count',-1)) != episode_count:
        errors.append(
            f"episode_count={payload.get('episode_count')!r} does not match discovered {episode_count}"
        )
    if errors:
        raise ValueError(f'{path}: ' + '; '.join(errors))
    return {
        'path': str(path.resolve()),
        'size_bytes': int(after.st_size),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'schema': payload['schema'],
        'action_policy': action_policy,
        'episode_count': int(payload['episode_count']),
    }

def h5_text(handle, key):
    value=handle[key][()]
    if isinstance(value,(bytes,np.bytes_)):
        return bytes(value).decode(errors='replace')
    return str(value)

def update_max_abs(current, lhs, rhs, label):
    delta=np.asarray(lhs,dtype=np.float64)-np.asarray(rhs,dtype=np.float64)
    if delta.size == 0:
        return current
    if not np.isfinite(delta).all():
        raise ValueError(f'{label}: non-finite action/state comparison')
    return max(current,float(np.max(np.abs(delta))))

def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path_lexists(destination):
        raise FileExistsError(f'Refusing to overwrite existing file: {destination}')
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        if path_lexists(destination):
            raise FileExistsError(f'Refusing to overwrite existing file: {destination}')
        shutil.copy2(source, destination)
EGO_INDICES = {
    'left_arm_joint_states': (4,8,12,16,20,22,24),
    'left_ee_joint_states': (26,36,27,37,28,38,29,39,30,40,46,48),
    'right_arm_joint_states': (5,9,13,17,21,23,25),
    'right_ee_joint_states': (31,41,32,42,33,43,34,44,35,45,47,49),
}

def task_text(name):
    return EGO_TASK_INSTRUCTIONS.get(name, name.replace('_',' '))

def feature_image(key, h=IMAGE_HEIGHT, w=IMAGE_WIDTH, fps=30):
    return {'dtype':'video','shape':[3,h,w],'names':['channels','height','width'],'info':{'video.height':h,'video.width':w,'video.codec':'mp4v','video.pix_fmt':'yuv420p','video.is_depth_map':False,'video.fps':fps,'video.channels':3,'has_audio':False}}

def feature_vec(n): return {'dtype':'float32','shape':[n],'names':[[f'joint_{i}' for i in range(n)]]}

def write_meta(out, dim, camera_keys, episodes, frames, tasks, fps):
    names=[f'joint_{i}' for i in range(dim)]
    half = None
    modality={'state':{},'action':{},'video':{},'annotation':{'human.action.task_description':{'original_key':'task_index'}}}
    if dim==54:
        groups=[('left_arm',0,7),('left_ee',7,27),('right_arm',27,34),('right_ee',34,54)]
    elif dim==38:
        groups=[('left_arm',0,7),('left_ee',7,19),('right_arm',19,26),('right_ee',26,38)]
    else: groups=[('all',0,dim)]
    for root in ('state','action'):
        for name,s,e in groups: modality[root][name]={'start':s,'end':e,'absolute':True,'dtype':'float32','original_key':'observation.state' if root=='state' else 'action'}
    for key in camera_keys:
        short=key.split('.')[-1]; modality['video'][short]={'original_key':key}
    (out/'meta').mkdir(parents=True,exist_ok=True)
    (out/'meta/modality.json').write_text(json.dumps(modality,indent=2))
    features={'observation.state':feature_vec(dim),'action':feature_vec(dim),'timestamp':{'dtype':'float32','shape':[1],'names':None},'frame_index':{'dtype':'int64','shape':[1],'names':None},'episode_index':{'dtype':'int64','shape':[1],'names':None},'index':{'dtype':'int64','shape':[1],'names':None},'task_index':{'dtype':'int64','shape':[1],'names':None}}
    for key in camera_keys: features[key]=feature_image(key,fps=fps)
    info={'codebase_version':'v3.0','robot_type':'tianji_marvin_wuji' if dim==54 else 'ego_h1_inspire','total_episodes':episodes,'total_frames':frames,'total_tasks':tasks,'chunks_size':1000,'data_files_size_in_mb':100,'video_files_size_in_mb':200,'fps':fps,'splits':{'train':f'0:{episodes}'},'data_path':'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet','video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4','features':features}
    (out/'meta/info.json').write_text(json.dumps(info,indent=2))

def write_video(frames, path, fps):
    path.parent.mkdir(parents=True,exist_ok=True); h,w=IMAGE_HEIGHT,IMAGE_WIDTH
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),float(fps),(w,h))
    if not writer.isOpened(): raise RuntimeError(f'VideoWriter failed: {path}')
    try:
        for image in frames:
            image=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
            # VideoWriter expects BGR; this conversion preserves the decoded
            # HDF5 pixels when the training video backend returns RGB.
            writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    finally: writer.release()

def write_spark_episode_videos(source_path, output_root, episode_index, fps, frame_count):
    """Stream one Spark episode into its three videos without retaining RGB frames."""
    cv2.setNumThreads(1)
    source_path=Path(source_path); output_root=Path(output_root)
    camera_pairs=zip(
        CAMERA_KEYS,
        ('cam_head','cam_left_wrist','cam_right_wrist'),
    )
    with h5py.File(source_path,'r') as handle:
        for key,camera in camera_pairs:
            dataset=handle[f'vision/{camera}/colors']
            if len(dataset) != frame_count:
                raise ValueError(
                    f'{source_path}: vision/{camera}/colors has {len(dataset)} rows, '
                    f'expected {frame_count}'
                )
            final=(
                output_root/'videos'/key/'chunk-000'/f'file-{episode_index:03d}.mp4'
            )
            temporary=final.with_name(
                f'.{final.stem}.{os.getpid()}.tmp.mp4'
            )
            if path_lexists(final) or path_lexists(temporary):
                raise FileExistsError(f'Refusing to overwrite video: {final}')
            try:
                write_video(
                    (decode_image_bit(dataset[index]) for index in range(frame_count)),
                    temporary,
                    fps,
                )
                if not temporary.is_file() or temporary.stat().st_size <= 0:
                    raise RuntimeError(f'VideoWriter produced an empty file: {temporary}')
                os.replace(temporary,final)
            finally:
                if path_lexists(temporary):
                    temporary.unlink()

def write_black_video(path, frame_count, fps):
    black=np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype=np.uint8)
    write_video((black for _ in range(frame_count)),path,fps)

def convert_spark(source,out,limit=None,keep=False,workers=1):
    paths=sorted(Path(source).glob('*/tianji_marvin_wuji/data/episode_*.hdf5'))
    return convert(paths,out,'spark',limit,keep,source=source,workers=workers)

def convert_spark_real(source,out,limit=None,keep=False,workers=1):
    paths=[]
    for task in sorted(Path(source).glob('*')):
        if task.is_dir(): paths += sorted(task.glob('episode_*.hdf5'))
    return convert(
        paths,out,'spark_real_bench_v5',limit,keep,source=source,workers=workers
    )

def convert_ego(source,out,limit=None,keep=False,workers=1):
    paths=[]
    for task in sorted(Path(source).glob('*')):
        if task.is_dir(): paths += sorted(task.glob('episode_*.hdf5'))
    return convert(paths,out,'ego',limit,keep,source=source,workers=workers)

def convert(paths,out,kind,limit,keep,source=None,workers=1):
    if keep:
        raise ValueError('--keep-existing is incompatible with fail-closed conversion')
    if limit: paths=paths[:limit]
    if not paths: raise FileNotFoundError('no episodes found')
    final=Path(out)
    if path_lexists(final):
        raise FileExistsError(f'Refusing to overwrite existing output: {final}')

    raw_manifest=None
    if kind == 'ego':
        raw_manifest=raw_dataset_manifest_provenance(source)
    elif kind == 'spark_real_bench_v5':
        raw_manifest=spark_real_manifest_provenance(source,len(paths),limit is not None)
    final.parent.mkdir(parents=True,exist_ok=True)
    atomic=kind in {'ego','spark_real_bench_v5'}
    if atomic:
        stage=stage_path(final)
        if path_lexists(stage):
            raise FileExistsError(f'Refusing to overwrite existing staging directory: {stage}')
        stage.mkdir()
        out=stage
    else:
        final.mkdir()
        out=final

    try:
        episodes,frames,dim=_convert_into(paths,out,kind,raw_manifest,workers)
        if atomic:
            if path_lexists(final):
                raise FileExistsError(f'Output appeared during conversion; staging retained: {final}')
            os.replace(out,final)
    except Exception:
        retained=out if atomic else final
        print(f'conversion failed; partial output retained for inspection: {retained}',file=sys.stderr)
        raise
    print(f'wrote {final}: episodes={episodes} frames={frames} action_dim={dim}')

def _convert_into(paths,out,kind,raw_manifest,workers):
    (out/'data/chunk-000').mkdir(parents=True,exist_ok=True); (out/'meta/episodes/chunk-000').mkdir(parents=True,exist_ok=True)
    rows=[]; erows=[]; task_map={}; total=0; fps=30; dataset_fps=None
    dim=54 if kind in SPARK_KINDS else 38
    cams=CAMERA_KEYS
    black_templates={}
    black_template_dir=out/'.black-video-templates'
    upstream_stats={
        'nonterminal_rows': 0,
        'arm_next_state_max_abs': {'left': 0.0, 'right': 0.0},
        'arm_terminal_hold_max_abs': {'left': 0.0, 'right': 0.0},
        'hand_next_state_max_abs': {'left': 0.0, 'right': 0.0},
    }
    video_executor=None
    video_futures=[]
    if kind in SPARK_KINDS and workers > 1:
        video_executor=ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
        )
    try:
      for ei,p in enumerate(tqdm(paths,desc=f'{kind} episodes')):
        with h5py.File(p,'r') as f:
            if kind in SPARK_KINDS:
                fps=int(np.asarray(f['additional_info/frequency']).item()) if 'additional_info/frequency' in f else 30
                if dataset_fps is None:
                    dataset_fps=fps
                elif fps != dataset_fps:
                    raise ValueError(
                        f'{p}: fps={fps} differs from dataset fps={dataset_fps}'
                    )
                instruction=f['instruction'][()].decode(errors='replace') if 'instruction' in f and isinstance(f['instruction'][()],bytes) else (str(f['instruction'][()]) if 'instruction' in f else p.parent.parent.name.replace('_',' '))
                state_parts={name:f[f'state/{name}'][()] for name in ('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')}
                action_parts={name:f[f'action/{name}'][()] for name in ('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')}
                state=np.concatenate(list(state_parts.values()),axis=1).astype('float32')
                # Intentionally use canonical action[t] at the same row.  Arm labels
                # are upstream next-state targets; hand labels are upstream aligned
                # master commands.  This converter never constructs or shifts either.
                action=np.concatenate(list(action_parts.values()),axis=1).astype('float32')
                frame_lengths={
                    key:len(f[f'vision/{camera}/colors'])
                    for key,camera in zip(
                        cams,('cam_head','cam_left_wrist','cam_right_wrist')
                    )
                }
                if kind == 'spark_real_bench_v5':
                    task=p.parent.name
                    expected_instruction=SPARK_REAL_TASK_INSTRUCTIONS.get(task)
                    if expected_instruction is None or instruction != expected_instruction:
                        raise ValueError(
                            f'{p}: instruction {instruction!r} does not match task {task!r}'
                        )
                    expected_metadata={
                        'metadata/action_source': 'master_hand_aligned_causal_with_zero_state_policy',
                        'metadata/arm_state_action_source': 'source_robot_joints',
                        'metadata/ee_action_source': 'master_hand_aligned_causal_with_zero_state_policy',
                    }
                    for key,expected in expected_metadata.items():
                        if key not in f or h5_text(f,key) != expected:
                            actual=h5_text(f,key) if key in f else None
                            raise ValueError(f'{p}: {key}={actual!r}, expected {expected!r}')
                    for camera in ('cam_head','cam_left_wrist','cam_right_wrist'):
                        group=f[f'vision/{camera}']
                        if group.attrs.get('encoding') != 'jpeg' or group.attrs.get('color_order') != 'true_rgb':
                            raise ValueError(
                                f'{p}: vision/{camera} must declare jpeg true_rgb source pixels'
                            )
                    length=len(state)
                    upstream_stats['nonterminal_rows'] += max(0,length-1)
                    for side in ('left','right'):
                        arm_state=state_parts[f'{side}_arm_joint_states']
                        arm_action=action_parts[f'{side}_arm_joint_states']
                        hand_state=state_parts[f'{side}_ee_joint_states']
                        hand_action=action_parts[f'{side}_ee_joint_states']
                        if length > 1:
                            upstream_stats['arm_next_state_max_abs'][side]=update_max_abs(
                                upstream_stats['arm_next_state_max_abs'][side],
                                arm_action[:-1],arm_state[1:],f'{p}: {side} arm next-state',
                            )
                            upstream_stats['hand_next_state_max_abs'][side]=update_max_abs(
                                upstream_stats['hand_next_state_max_abs'][side],
                                hand_action[:-1],hand_state[1:],f'{p}: {side} hand next-state',
                            )
                        upstream_stats['arm_terminal_hold_max_abs'][side]=update_max_abs(
                            upstream_stats['arm_terminal_hold_max_abs'][side],
                            arm_action[-1],arm_state[-1],f'{p}: {side} arm terminal hold',
                        )
            else:
                task=p.parent.name; instruction=task_text(task); fps=30
                q=f['observations/qpos'][()]; a=f['action'][()]
                state=np.concatenate([q[:,list(i)] for i in EGO_INDICES.values()],axis=1).astype('float32')
                action=np.concatenate([a[:,list(i)] for i in EGO_INDICES.values()],axis=1).astype('float32')
                images=f['observations/images/main'][:]
                frame_lists={
                    'observation.images.cam_high':[im for im in images],
                }
                frame_lengths={key:len(frames) for key,frames in frame_lists.items()}
            lengths={'state':len(state),'action':len(action),**frame_lengths}
            if kind == 'ego':
                lengths.update({key:len(images) for key in CAMERA_KEYS[1:]})
            if len(set(lengths.values())) != 1:
                raise ValueError(f'{p}: unaligned source lengths would change action timing: {lengths}')
            length=len(state); task_idx=task_map.setdefault(instruction,len(task_map))
            video_meta={}
            if kind in SPARK_KINDS:
                video_args=(str(p),str(out),ei,fps,length)
                if video_executor is None:
                    write_spark_episode_videos(*video_args)
                else:
                    video_futures.append(
                        video_executor.submit(write_spark_episode_videos,*video_args)
                    )
            for key in cams:
                if kind not in SPARK_KINDS:
                    vp=out/'videos'/key/'chunk-000'/f'file-{ei:03d}.mp4'
                    if key in frame_lists:
                        write_video(frame_lists[key][:length],vp,fps)
                    else:
                        template=black_templates.get(length)
                        if template is None:
                            template=black_template_dir/f'black-{length:06d}.mp4'
                            write_black_video(template,length,fps)
                            black_templates[length]=template
                        link_or_copy(template,vp)
                video_meta[f'videos/{key}/from_timestamp']=0.0; video_meta[f'videos/{key}/chunk_index']=0; video_meta[f'videos/{key}/file_index']=ei
            for fi in range(length):
                rows.append({'episode_index':ei,'frame_index':fi,'timestamp':fi/float(fps),'task_index':task_idx,'index':total+fi,'observation.state':state[fi],'action':action[fi]})
            erow={'episode_index':ei,'length':length,'tasks':[instruction],'data/chunk_index':0,'data/file_index':0,'data/file_from_index':total,'data/file_to_index':total+length,'dataset_from_index':total,'dataset_to_index':total+length}; erow.update(video_meta); erows.append(erow); total += length
      for future in tqdm(video_futures,desc='spark videos'):
          future.result()
    except Exception:
      for future in video_futures:
          future.cancel()
      raise
    finally:
      if video_executor is not None:
          video_executor.shutdown(wait=True,cancel_futures=True)
    if kind == 'spark_real_bench_v5':
        if upstream_stats['nonterminal_rows'] <= 0:
            raise ValueError('SParkRealBenchV5 has no non-terminal rows to verify')
        for side in ('left','right'):
            if upstream_stats['arm_next_state_max_abs'][side] != 0.0:
                raise ValueError(
                    f'SParkRealBenchV5 {side} arm action is not the declared next-state target: '
                    f"max_abs={upstream_stats['arm_next_state_max_abs'][side]}"
                )
            if upstream_stats['arm_terminal_hold_max_abs'][side] != 0.0:
                raise ValueError(
                    f'SParkRealBenchV5 {side} arm terminal action is not hold-last state'
                )
            if upstream_stats['hand_next_state_max_abs'][side] == 0.0:
                raise ValueError(
                    f'SParkRealBenchV5 {side} hand action was replaced by state[t+1]; '
                    'expected the canonical aligned master-hand action'
                )
    for template in black_templates.values():
        template.unlink()
    if black_template_dir.exists():
        black_template_dir.rmdir()
    pd.DataFrame(rows).to_parquet(out/'data/chunk-000/file-000.parquet',index=False)
    pd.DataFrame(erows).to_parquet(out/'meta/episodes/chunk-000/file-000.parquet',index=False)
    pd.DataFrame({'task_index':list(task_map.values())},index=list(task_map.keys())).to_parquet(out/'meta/tasks.parquet')
    write_meta(out,dim,cams,len(erows),total,len(task_map),dataset_fps or fps)
    source_keys = ({
        'video': {
            'observation.images.cam_high': 'vision/cam_head/colors',
            'observation.images.cam_left_wrist': 'vision/cam_left_wrist/colors',
            'observation.images.cam_right_wrist': 'vision/cam_right_wrist/colors',
        },
        'state': ['state/left_arm_joint_states', 'state/left_ee_joint_states',
                  'state/right_arm_joint_states', 'state/right_ee_joint_states'],
        'action': ['action/left_arm_joint_states', 'action/left_ee_joint_states',
                   'action/right_arm_joint_states', 'action/right_ee_joint_states'],
        'language': 'instruction',
    } if kind in SPARK_KINDS else {
        'video': {
            'observation.images.cam_high': 'observations/images/main',
            'observation.images.cam_left_wrist': 'constant_black',
            'observation.images.cam_right_wrist': 'constant_black',
        },
        'state': 'observations/qpos (38 selected indices)',
        'action': 'action',
        'language': 'task directory name',
    })
    conversion_manifest={
        'contract_version': 2,
        'kind': kind, 'episodes': [str(p) for p in paths], 'action_dim': dim,
        'camera_keys': cams, 'source_keys': source_keys,
        'action_contract': {
            'source': 'raw_hdf5_action',
            'selected_indices': (
                {name: list(indices) for name, indices in EGO_INDICES.items()}
                if kind == 'ego' else None
            ),
            'temporal_offset': 0,
            'derived_from_state': False,
        },
        'image_contract': {
            'height': IMAGE_HEIGHT,
            'width': IMAGE_WIDTH,
            'dtype': 'uint8',
            'color_order': 'rgb',
            'black_camera_keys': [] if kind in SPARK_KINDS else CAMERA_KEYS[1:],
        },
        'image_decode': 'XPolicyLab.utils.process_data.decode_image_bit (RGB)' if kind in SPARK_KINDS else 'decoded uint8 RGB array',
        'instruction_contract': ({
            'source': 'EgoVLA official LANGUAGE_MAPPING',
            'mapping': EGO_TASK_INSTRUCTIONS,
            'mapping_sha256': mapping_sha256(EGO_TASK_INSTRUCTIONS),
        } if kind == 'ego' else {
            'source': 'SParkRealBenchV5 canonical task registry',
            'mapping': SPARK_REAL_TASK_INSTRUCTIONS,
            'mapping_sha256': mapping_sha256(SPARK_REAL_TASK_INSTRUCTIONS),
        } if kind == 'spark_real_bench_v5' else {
            'source': 'raw_hdf5_instruction',
        }),
        'source_is_external': True,
    }
    if kind == 'spark_real_bench_v5':
        conversion_manifest['upstream_action_contract']={
            'arm': {
                'source': 'canonical action/{left,right}_arm_joint_states',
                'semantics': 'state[t+1], terminal hold-last',
                'verified_max_abs_error': upstream_stats['arm_next_state_max_abs'],
                'terminal_hold_max_abs_error': upstream_stats['arm_terminal_hold_max_abs'],
            },
            'hand': {
                'source': 'canonical aligned master-hand action',
                'semantics': 'use action[t] unchanged; never substitute state[t+1]',
                'verified_not_next_state_max_abs': upstream_stats['hand_next_state_max_abs'],
            },
        }
    if raw_manifest is not None:
        conversion_manifest['raw_dataset_manifest']=raw_manifest
    (out/'conversion_manifest.json').write_text(
        json.dumps(conversion_manifest,indent=2)+'\n',encoding='utf-8'
    )
    return len(erows),total,dim

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('benchmark',choices=['spark','spark_real_bench_v5','egovla']); ap.add_argument('--source',required=True); ap.add_argument('--output',required=True); ap.add_argument('--limit',type=int); ap.add_argument('--keep-existing',action='store_true'); ap.add_argument('--workers',type=int,default=1); a=ap.parse_args()
    if a.workers <= 0:
        ap.error('--workers must be positive')
    converter={'spark':convert_spark,'spark_real_bench_v5':convert_spark_real,'egovla':convert_ego}[a.benchmark]
    converter(a.source,a.output,a.limit,a.keep_existing,a.workers)
if __name__=='__main__': main()
