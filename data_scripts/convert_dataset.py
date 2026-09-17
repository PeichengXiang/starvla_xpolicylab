from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
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

def mapping_sha256(mapping):
    payload=json.dumps(mapping,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def path_lexists(path):
    return os.path.lexists(path)

def ego_stage_path(out):
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

def write_black_video(path, frame_count, fps):
    black=np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype=np.uint8)
    write_video((black for _ in range(frame_count)),path,fps)

def convert_spark(source,out,limit=None,keep=False):
    paths=sorted(Path(source).glob('*/tianji_marvin_wuji/data/episode_*.hdf5'))
    return convert(paths,out,'spark',limit,keep,source=source)

def convert_ego(source,out,limit=None,keep=False):
    paths=[]
    for task in sorted(Path(source).glob('*')):
        if task.is_dir(): paths += sorted(task.glob('episode_*.hdf5'))
    return convert(paths,out,'ego',limit,keep,source=source)

def convert(paths,out,kind,limit,keep,source=None):
    if keep:
        raise ValueError('--keep-existing is incompatible with fail-closed conversion')
    if limit: paths=paths[:limit]
    if not paths: raise FileNotFoundError('no episodes found')
    final=Path(out)
    if path_lexists(final):
        raise FileExistsError(f'Refusing to overwrite existing output: {final}')

    raw_manifest=(
        raw_dataset_manifest_provenance(source)
        if kind == 'ego'
        else None
    )
    final.parent.mkdir(parents=True,exist_ok=True)
    if kind == 'ego':
        stage=ego_stage_path(final)
        if path_lexists(stage):
            raise FileExistsError(f'Refusing to overwrite existing staging directory: {stage}')
        stage.mkdir()
        out=stage
    else:
        final.mkdir()
        out=final

    try:
        episodes,frames,dim=_convert_into(paths,out,kind,raw_manifest)
        if kind == 'ego':
            if path_lexists(final):
                raise FileExistsError(f'Output appeared during conversion; staging retained: {final}')
            os.replace(out,final)
    except Exception:
        retained=out if kind == 'ego' else final
        print(f'conversion failed; partial output retained for inspection: {retained}',file=sys.stderr)
        raise
    print(f'wrote {final}: episodes={episodes} frames={frames} action_dim={dim}')

def _convert_into(paths,out,kind,raw_manifest):
    (out/'data/chunk-000').mkdir(parents=True,exist_ok=True); (out/'meta/episodes/chunk-000').mkdir(parents=True,exist_ok=True)
    rows=[]; erows=[]; task_map={}; total=0; fps=30
    dim=54 if kind=='spark' else 38
    cams=CAMERA_KEYS
    black_templates={}
    black_template_dir=out/'.black-video-templates'
    for ei,p in enumerate(tqdm(paths,desc=f'{kind} episodes')):
        with h5py.File(p,'r') as f:
            if kind=='spark':
                fps=int(np.asarray(f['additional_info/frequency']).item()) if 'additional_info/frequency' in f else 30
                instruction=f['instruction'][()].decode(errors='replace') if 'instruction' in f and isinstance(f['instruction'][()],bytes) else (str(f['instruction'][()]) if 'instruction' in f else p.parent.parent.name.replace('_',' '))
                state=np.concatenate([f['state/left_arm_joint_states'][()],f['state/left_ee_joint_states'][()],f['state/right_arm_joint_states'][()],f['state/right_ee_joint_states'][()]],axis=1).astype('float32')
                action=np.concatenate([f['action/left_arm_joint_states'][()],f['action/left_ee_joint_states'][()],f['action/right_arm_joint_states'][()],f['action/right_ee_joint_states'][()]],axis=1).astype('float32')
                frame_lists={k:[decode_image_bit(x) for x in f['vision'][cam]['colors'][:]] for k,cam in zip(cams,['cam_head','cam_left_wrist','cam_right_wrist'])}
            else:
                task=p.parent.name; instruction=task_text(task); fps=30
                q=f['observations/qpos'][()]; a=f['action'][()]
                state=np.concatenate([q[:,list(i)] for i in EGO_INDICES.values()],axis=1).astype('float32')
                action=np.concatenate([a[:,list(i)] for i in EGO_INDICES.values()],axis=1).astype('float32')
                images=f['observations/images/main'][:]
                frame_lists={
                    'observation.images.cam_high':[im for im in images],
                }
            lengths={'state':len(state),'action':len(action),**{key:len(frames) for key,frames in frame_lists.items()}}
            if kind == 'ego':
                lengths.update({key:len(images) for key in CAMERA_KEYS[1:]})
            if len(set(lengths.values())) != 1:
                raise ValueError(f'{p}: unaligned source lengths would change action timing: {lengths}')
            length=len(state); task_idx=task_map.setdefault(instruction,len(task_map))
            video_meta={}
            for key in cams:
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
    for template in black_templates.values():
        template.unlink()
    if black_template_dir.exists():
        black_template_dir.rmdir()
    pd.DataFrame(rows).to_parquet(out/'data/chunk-000/file-000.parquet',index=False)
    pd.DataFrame(erows).to_parquet(out/'meta/episodes/chunk-000/file-000.parquet',index=False)
    pd.DataFrame({'task_index':list(task_map.values())},index=list(task_map.keys())).to_parquet(out/'meta/tasks.parquet')
    write_meta(out,dim,cams,len(erows),total,len(task_map),fps)
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
    } if kind == 'spark' else {
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
            'black_camera_keys': [] if kind == 'spark' else CAMERA_KEYS[1:],
        },
        'image_decode': 'XPolicyLab.utils.process_data.decode_image_bit (RGB)' if kind == 'spark' else 'decoded uint8 RGB array',
        'instruction_contract': ({
            'source': 'EgoVLA official LANGUAGE_MAPPING',
            'mapping': EGO_TASK_INSTRUCTIONS,
            'mapping_sha256': mapping_sha256(EGO_TASK_INSTRUCTIONS),
        } if kind == 'ego' else {
            'source': 'raw_hdf5_instruction',
        }),
        'source_is_external': True,
    }
    if raw_manifest is not None:
        conversion_manifest['raw_dataset_manifest']=raw_manifest
    (out/'conversion_manifest.json').write_text(
        json.dumps(conversion_manifest,indent=2)+'\n',encoding='utf-8'
    )
    return len(erows),total,dim

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('benchmark',choices=['spark','egovla']); ap.add_argument('--source',required=True); ap.add_argument('--output',required=True); ap.add_argument('--limit',type=int); ap.add_argument('--keep-existing',action='store_true'); a=ap.parse_args()
    (convert_spark if a.benchmark=='spark' else convert_ego)(a.source,a.output,a.limit,a.keep_existing)
if __name__=='__main__': main()
