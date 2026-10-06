#!/usr/bin/env python3
"""Replay a recorded 18-D command sequence without perception/policy feedback."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import shlex
import fcntl
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config, resolve_package_path
from sock_dressing_simulation.demo import run_demo, _open_video, _write_video_frame
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.perception import PerceptionResult, assess_masks
from sock_dressing_simulation.scenario import scenario_from_config

BASELINE = ROOT / 'artifacts/phase4/opening-short-edge-widen/final100-w95/data_sock_sim_smoke/train/phase4_20261005T011108Z'
DEFAULT_OUTPUT = ROOT / 'artifacts/phase4/foot-nonpenetration-fixed-motion'



def renderer_measurements(observation, masks):
    """Use the existing simulator camera contract, without learned perception."""
    if masks is None:
        raise RuntimeError('renderer masks unavailable during fixed replay')
    sock=np.asarray(masks['sock'],dtype=bool)
    leg=np.asarray(masks['leg'],dtype=bool)
    depth=np.asarray(observation['camera']['camera_depth'],dtype=np.uint8)
    quality=assess_masks(sock,leg,min_fraction=1e-7,max_fraction=.95,max_area_change=1000)
    quality['source']='renderer_measurements_for_fixed_motion_replay'
    return PerceptionResult(depth,sock,leg,depth*sock,depth*leg,quality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--substeps', type=int, choices=(8,16), default=8)
    parser.add_argument('--iterations', type=int, choices=(20,40), default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--remaining-grid', action='store_true')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.remaining_grid:
        for substeps, iterations in [(8,40),(16,20),(16,40)]:
            args.output_root.mkdir(parents=True,exist_ok=True)
            log=args.output_root/f's{substeps}_i{iterations}.log'
            with log.open('w') as stream:
                result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--substeps',str(substeps),'--iterations',str(iterations),
                    '--output-root',str(args.output_root)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            print(f's{substeps}_i{iterations} exit={result.returncode}',flush=True)
            if result.returncode: raise RuntimeError(f'candidate failed: see {log}')
        return
    config = load_config(ROOT / 'config/opening_widen_w95.yaml')
    actions = BASELINE / 'applied_action.csv'
    values = np.loadtxt(actions, delimiter=',', ndmin=2)
    if values.shape != (100,18): raise ValueError('baseline must contain exactly 100 commands')
    config['rcareworld']['graphics'] = True
    config['rcareworld']['port'] = 5100 + (2 if args.substeps == 16 else 0) + (1 if args.iterations == 40 else 0)
    config['obi']['substeps'] = config['obi']['expected']['substeps'] = args.substeps
    config['obi']['solver_iterations'] = config['obi']['expected']['solver_iterations'] = args.iterations
    config['inference'].update(reference_actions=str(actions),reference_action_blend=1.0,
        reference_action_hold_steps=1,reference_action_interpolation='hold',device='cpu')
    directory = args.output_root / f's{args.substeps}_i{args.iterations}'
    directory.mkdir(parents=True,exist_ok=True)
    # A grid controller and a parallel invocation can request the same condition.
    # Hold the condition lock through metadata publication, then reuse only a
    # completed recording produced by these exact sources/configuration.
    condition_lock=(directory/'replay.lock').open('a')
    fcntl.flock(condition_lock.fileno(),fcntl.LOCK_EX)
    digest=hashlib.sha256(yaml.safe_dump(config,sort_keys=True).encode())
    for source in [Path(__file__).resolve(),ROOT/'sock_dressing_simulation/demo.py',
            ROOT/'sock_dressing_simulation/environment.py',ROOT/'sock_dressing_simulation/sock_cloth.py',
            ROOT/'Build/SockDressingPlayer/Development/Player_Data/Managed/RCareWorld.dll']:
        digest.update(source.read_bytes())
    engine_hash=digest.hexdigest()
    cache=directory/'result.json'
    if cache.exists():
        try:
            cached=json.loads(Path(json.loads(cache.read_text())['metadata']).read_text())
            if cached.get('frames')==100 and cached.get('stop_reason')=='max_steps' and cached.get('replay_engine_sha256')==engine_hash:
                print('REUSED '+str(cache),flush=True)
                return
        except (OSError,ValueError,KeyError):
            pass
    # Separate native logs make concurrent fixed-command experiments reviewable.
    executable=resolve_package_path(config['rcareworld']['executable']).resolve()
    wrapper=directory/'player.sh'
    wrapper.write_text('#!/bin/bash\nexec '+shlex.quote(str(executable))+
        ' -logFile '+shlex.quote(str((directory/'player.log').resolve()))+' "$@"\n')
    wrapper.chmod(0o755)
    config['rcareworld']['executable']=str(wrapper.resolve())
    (directory/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    last = {}; side_video = [None]; track_frame = [0]; errors = []

    class ReplayEnvironment(SockDressingEnv):
        def observe(self):
            value = super().observe(); last.clear(); last.update(value); return value
        def command(self, command, previous=None):
            # The baseline commands already passed the position/rate limiter.
            # Re-limiting against changed contact feedback would change this experiment.
            result = super().command(command, None)
            errors.append(float(np.max(np.abs(result-command))))
            return result

    class ReplayPolicy:
        def __init__(self,config,**kwargs):
            self.checkpoint=actions; self.checkpoint_sha256=hashlib.sha256(actions.read_bytes()).hexdigest()
        def step(self,**kwargs):
            return {'action':values[min(int(kwargs.get('step_index',0)),len(values)-1)].copy()}

    class RendererMeasurements:
        def __init__(self, config): pass
        def initialize(self,rgb,**kwargs): return self.measure(rgb,kwargs.get('renderer_masks'))
        def track(self,rgb,renderer_masks=None):
            side = last.get('side_recording_camera')
            if side is not None:
                if side_video[0] is None:
                    side_video[0]=_open_video(directory/'side_demo.mp4',side['rgb'].shape,5.0)
                _write_video_frame(side_video[0],side['rgb'],track_frame[0]);track_frame[0]+=1
            result=self.measure(rgb,renderer_masks)
            print(f"frame {track_frame[0]} safety="+json.dumps(last.get('dressing_qa',{}).get('foot_collision_safety',{})),flush=True)
            return result
        def measure(self,rgb,masks):
            return renderer_measurements(last, masks)

    try:
        result=run_demo(config,prepared=_prepare(config,scenario_from_config(config,seed=0)),
            output_root=directory,sock_points=None,leg_points=None,max_steps=args.steps,seed=0,
            environment_factory=ReplayEnvironment,policy_factory=ReplayPolicy,
            perception_factory=RendererMeasurements)
    finally:
        if side_video[0] is not None: side_video[0].release()
    metadata_paths=list(directory.glob('data_sock_sim_smoke/train/phase4_*/metadata.json'))
    metadata_path=max(metadata_paths,key=lambda p:p.stat().st_mtime)
    metadata=json.loads(metadata_path.read_text())
    metadata.update(mode='fixed_motion_replay',replay_engine_sha256=engine_hash,
        frames_inferred=0,frames_replayed=metadata['frames'],
        replay_source=str(actions),
        replay_source_sha256=hashlib.sha256(actions.read_bytes()).hexdigest(),
        replay_command_maximum_error=max(errors,default=0),
        perception='renderer RGB/depth/masks (fixed command replay)',
        command_contract='recorded applied 18-D commands -> position bounds/mimic -> RCareWorld',
        original_policy_checkpoint=config['inference']['checkpoint'],
        perception_source='renderer_measurements',policy_source='recorded_applied_commands',
        side_video=str(directory/'side_demo.mp4'))
    metadata_path.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    (directory/'result.json').write_text(json.dumps({'result':result,'metadata':str(metadata_path)},indent=2)+'\n')
    print('METADATA '+str(metadata_path),flush=True)

if __name__=='__main__': main()
