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
import copy
import re
import shutil
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



def prepared_replay_assets(config, directory):
    """Generate immutable shared assets once, before any native reader loads them."""
    scenario=scenario_from_config(config,seed=0)
    asset_directory=directory.parent/'prepared_assets'
    asset_directory.mkdir(exist_ok=True)
    identity={'assets':{k:v for k,v in config['assets'].items() if k!='output_dir'},
              'sock_mesh':scenario.to_metadata()['sock_mesh'],
              'generator_sha256':hashlib.sha256((ROOT/'sock_dressing_simulation/assets.py').read_bytes()).hexdigest()}
    manifest=asset_directory/'manifest.json'
    with (asset_directory/'generation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if manifest.exists():
            record=json.loads(manifest.read_text())
            if record['identity']!=identity:
                raise RuntimeError('Shared replay assets belong to another robot or sock; use another comparison directory')
            for name,digest in record['files'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Immutable replay asset changed: '+name)
            return record['prepared']
        asset_config=copy.deepcopy(config)
        asset_config['assets']['output_dir']=str(asset_directory.resolve())
        prepared=_prepare(asset_config,scenario)
        files={prepared[k]:hashlib.sha256(Path(prepared[k]).read_bytes()).hexdigest()
               for k in ['urdf','runtime_urdf','sock_obj']}
        record={'identity':identity,'prepared':prepared,'files':files}
        temporary=manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2)+'\n')
        temporary.replace(manifest)
        return prepared


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
    parser.add_argument('--candidate-name')
    parser.add_argument('--material-overrides', type=Path)
    parser.add_argument('--surface-mode', choices=('legacy','local'),default='legacy')
    parser.add_argument('--foot-shape', choices=('boxes','skin-hull'),default='boxes')
    parser.add_argument('--transport', choices=('bulk','local'),default='bulk')
    parser.add_argument('--fixed-grasp-offsets',action='store_true')
    parser.add_argument('--post-preparation-material',action='store_true')
    parser.add_argument('--prepared-state',type=Path,
                        help='Restore a validated common preparation without moving targets or advancing physics')
    parser.add_argument('--diagnostic-continue-failed-preparation',action='store_true',
                        help='Retain failed preparation QA and continue only for investigation; excluded from best selection')
    parser.add_argument('--port',type=int)
    parser.add_argument('--player-executable',type=Path,help='Use an immutable, separately built native player')
    parser.add_argument('--detach',action='store_true',
                        help='Keep a logged trial running independently of chat/tool interruptions')
    parser.add_argument('--terminal-projection-probe',type=int,default=0,
                        help='Diagnostic projections after recording, without simulated time; never part of candidate scoring')
    args = parser.parse_args()
    if args.detach:
        name=args.candidate_name or f's{args.substeps}_i{args.iterations}'
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name): raise ValueError('Unsafe candidate name')
        directory=args.output_root/name
        directory.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(Path(__file__).resolve())]+[v for v in sys.argv[1:] if v!='--detach']
        log_path=directory/'controller.log'
        with log_path.open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        report=dict(pid=process.pid,command=command,log=str(log_path),candidate=name)
        (directory/'background_job.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        return
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
    if args.player_executable:config['rcareworld']['executable']=str(args.player_executable.resolve())
    native_player=resolve_package_path(config['rcareworld']['executable']).resolve()
    native_kernel=native_player.parent/'Player_Data/Managed'
    preparation_parameters=copy.deepcopy(config['obi']['expected'])
    parameters=copy.deepcopy(preparation_parameters)
    if args.material_overrides:
        overrides=yaml.safe_load(args.material_overrides.read_text()) or {}
        allowed={'friction','damping','stretch_compliance','bend_compliance','particle_radius_m','collision_margin_m'}
        if set(overrides)-allowed: raise ValueError('Unknown or fixed material override: '+str(set(overrides)-allowed))
        parameters.update(overrides)
    parameters.update(substeps=args.substeps,solver_iterations=args.iterations)
    if not args.post_preparation_material:
        config['obi']['expected']=copy.deepcopy(parameters)
    config['obi']['requested']=copy.deepcopy(config['obi']['expected'])
    config['foot_conforming_rollout']={'parameters':parameters,'preparation_parameters':preparation_parameters,
        'surface_mode':args.surface_mode,'collider_shape':args.foot_shape,'transport':args.transport,'fixed_grasp_offsets':args.fixed_grasp_offsets,
        'post_preparation_material':args.post_preparation_material}
    config['foot_conforming_rollout']['diagnostic_continue_failed_preparation']=args.diagnostic_continue_failed_preparation
    if args.prepared_state:
        if not args.post_preparation_material: raise ValueError('Prepared state requires post-preparation material application')
        config['foot_conforming_rollout']['prepared_state_path']=str(args.prepared_state.resolve())
    actions = BASELINE / 'applied_action.csv'
    values = np.loadtxt(actions, delimiter=',', ndmin=2)
    if values.shape != (100,18): raise ValueError('baseline must contain exactly 100 commands')
    config['rcareworld']['graphics'] = True
    config['rcareworld']['port'] = args.port or 5100 + (2 if args.substeps == 16 else 0) + (1 if args.iterations == 40 else 0)
    if not args.post_preparation_material:
        config['obi']['substeps'] = args.substeps
        config['obi']['solver_iterations'] = args.iterations
    config['inference'].update(reference_actions=str(actions),reference_action_blend=1.0,
        reference_action_hold_steps=1,reference_action_interpolation='hold',device='cpu')
    name=args.candidate_name or f's{args.substeps}_i{args.iterations}'
    if not re.fullmatch(r'[A-Za-z0-9_-]+',name): raise ValueError('Unsafe candidate name')
    directory = args.output_root / name
    directory.mkdir(parents=True,exist_ok=True)
    # A grid controller and a parallel invocation can request the same condition.
    # Hold the condition lock through metadata publication, then reuse only a
    # completed recording produced by these exact sources/configuration.
    condition_lock=(directory/'replay.lock').open('a')
    fcntl.flock(condition_lock.fileno(),fcntl.LOCK_EX)
    native_source_archive = native_kernel.parents[1] / 'runtime_sources'
    native_source_manifest_path = native_source_archive / 'manifest.json'
    native_source_manifest = json.loads(native_source_manifest_path.read_text()) if native_source_manifest_path.exists() else {}
    digest=hashlib.sha256(yaml.safe_dump(config,sort_keys=True).encode())
    for source in [Path(__file__).resolve(),ROOT/'sock_dressing_simulation/demo.py',
            ROOT/'sock_dressing_simulation/quality.py',
            ROOT/'sock_dressing_simulation/environment.py',ROOT/'sock_dressing_simulation/sock_cloth.py',
            ROOT/'RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs',
            ROOT/'RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockPreparedState.cs',
            ROOT/'RCareUnity/Assets/RCareCommon/Scripts/Main/PlayerMain.cs',
            ROOT/'RCareUnity/Assets/Paid Dependencies/Obi/Scripts/Common/Solver/ObiSolver.cs',
            native_kernel/'Obi.dll',
            native_kernel/'RCareWorld.dll']:
        try:
            relative = str(source.relative_to(ROOT))
        except ValueError:
            relative = ''
        digest.update((native_source_archive/source.name).read_bytes() if relative in native_source_manifest else source.read_bytes())
    if args.prepared_state:digest.update(args.prepared_state.read_bytes())
    if native_source_manifest_path.exists():digest.update(native_source_manifest_path.read_bytes())
    engine_hash=digest.hexdigest()
    cache=directory/'result.json'
    if cache.exists():
        try:
            cached=json.loads(Path(json.loads(cache.read_text())['metadata']).read_text())
            if cached.get('frames')==args.steps and cached.get('stop_reason')=='max_steps' and cached.get('replay_engine_sha256')==engine_hash:
                print('REUSED '+str(cache),flush=True)
                return
        except (OSError,ValueError,KeyError):
            pass
    # Retain the exact native source as well as its compiled identity. Unity's
    # source tree is shared between experiments and can change during later
    # fixes; a hash alone does not make an old trial reproducible or reviewable.
    sources = directory / 'sources'
    sources.mkdir(exist_ok=True)
    source_manifest = {}
    immutable_sources = native_kernel.parents[1] / 'runtime_sources'
    immutable_manifest_path = immutable_sources / 'manifest.json'
    immutable_manifest = json.loads(immutable_manifest_path.read_text()) if immutable_manifest_path.exists() else {}
    for library in ('RCareWorld.dll', 'Obi.dll'):
        expected_library = immutable_manifest.get('compiled_' + library)
        if expected_library is not None and hashlib.sha256((native_kernel/library).read_bytes()).hexdigest() != expected_library:
            raise RuntimeError('Archived runtime source does not match compiled player: ' + library)
    for source in [Path(__file__).resolve(),
            ROOT/'sock_dressing_simulation/demo.py',
            ROOT/'sock_dressing_simulation/quality.py',
            ROOT/'sock_dressing_simulation/environment.py',
            ROOT/'sock_dressing_simulation/sock_cloth.py',
            ROOT/'RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs',
            ROOT/'RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockPreparedState.cs',
            ROOT/'RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/FootContactGeometry.cs',
            ROOT/'RCareUnity/Assets/RCareCommon/Scripts/Main/PlayerMain.cs',
            ROOT/'RCareUnity/Assets/Paid Dependencies/Obi/Scripts/Common/Solver/ObiSolver.cs']:
        relative = str(source.relative_to(ROOT))
        archived = immutable_sources / source.name if relative in immutable_manifest else source
        data = archived.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if relative in immutable_manifest and digest != immutable_manifest[relative]:
            raise RuntimeError('Archived native source changed: ' + relative)
        shutil.copyfile(archived, sources/source.name)
        source_manifest[relative] = digest
    for relative, expected_digest in immutable_manifest.items():
        if relative.startswith('compiled_') or relative in source_manifest:
            continue
        archived = immutable_sources / Path(relative).name
        data = archived.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_digest:
            raise RuntimeError('Archived native source changed: ' + relative)
        shutil.copyfile(archived, sources/archived.name)
        source_manifest[relative] = expected_digest
    kernel = native_kernel/'RCareWorld.dll'
    source_manifest['compiled_RCareWorld.dll'] = hashlib.sha256(kernel.read_bytes()).hexdigest()
    source_manifest['compiled_Obi.dll'] = hashlib.sha256((kernel.parent/'Obi.dll').read_bytes()).hexdigest()
    (directory/'replay_sources.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
    # Separate native logs make concurrent fixed-command experiments reviewable.
    (directory/'live_frame_diagnostics.jsonl').write_text('')
    executable=resolve_package_path(config['rcareworld']['executable']).resolve()
    wrapper=directory/'player.sh'
    wrapper.write_text('#!/bin/bash\nexport SOCK_CONTACT_PROGRESS_PATH='+shlex.quote(str((directory/'contact-progress.log').resolve()))+'\nexec '+shlex.quote(str(executable))+
        ' -logFile '+shlex.quote(str((directory/'player.log').resolve()))+' "$@"\n')
    wrapper.chmod(0o755)
    config['rcareworld']['executable']=str(wrapper.resolve())
    (directory/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    last = {}; side_video = [None]; track_frame = [0]; errors = []; initial_clock = [None]

    class ReplayEnvironment(SockDressingEnv):
        def close(self):
            try:
                attempt=getattr(self,'_preparation_attempt_state',None)
                if attempt:(directory/'preparation_attempt_state.json').write_text(attempt+'\n')
                if args.terminal_projection_probe and track_frame[0] == args.steps and not getattr(self,'_terminal_probe_done',False):
                    self._terminal_probe_done=True
                    probes=[]
                    for iteration in range(args.terminal_projection_probe):
                        self.sock_cloth.stabilize_constraints()
                        self.sock_cloth.request_particles()
                        self.sock_cloth.request_grasp_state()
                        self.sock_cloth.request_scene_geometry()
                        self.sock_cloth.request_dressing_qa()
                        self._env.step(simulate=False)
                        data=dict(self.sock_cloth.data)
                        qa=data.get('dressing_qa',{})
                        stretch=self.cloth_radius_qa(data)
                        report=dict(iteration=iteration+1,simulated_steps=0,
                                    dressing_qa=qa,stretch=stretch,scene_geometry=data.get('scene_geometry',{}),
                                    particles_world=self.sock_cloth.particles().tolist(),
                                    particle_edges=data.get('particle_edges',[]),
                                    particle_rest_edge_lengths=data.get('particle_rest_edge_lengths',[]),
                                    particle_surface_triangles=data.get('particle_surface_triangles',[]),
                                    opening_particle_indices=data.get('opening_particle_indices',[]))
                        probes.append(report)
                        safety=qa.get('foot_collision_safety',{})
                        clock = safety.get('cloth_physics_clock', {})
                        recorded_clock = last.get('dressing_qa',{}).get('foot_collision_safety',{}).get('cloth_physics_clock',{})
                        if not clock.get('externally_stepped') or clock != recorded_clock:
                            raise RuntimeError('Diagnostic projection advanced cloth physics or missing explicit cloth clock')
                        limits_ok=stretch.get('available',False) and all(
                            v.get('maximum_stretch') is None or v['maximum_stretch'] <=
                            (2 if k=='opening_rim' else 1.1 if k=='opening_body' else 1.5)*1.01
                            for k,v in stretch.get('edge_classes',{}).items())
                        print('terminal probe '+json.dumps(dict(iteration=iteration+1,
                              remaining_surfaces=safety.get('remaining_surface_intersections'),limits_ok=limits_ok)),flush=True)
                        if safety.get('remaining_surface_intersections')==0 and limits_ok:
                            break
                    (directory/'terminal_projection_probe.json').write_text(json.dumps(dict(
                        diagnostic_only=True,excluded_from_recording_and_selection=True,
                        simulated_steps=0,probes=probes),indent=2)+'\n')
            finally:
                super().close()
        def begin_cloth_contact_rollout(self):
            super().begin_cloth_contact_rollout()
            self.sock_cloth.request_dressing_qa()
            self.sock_cloth.request_particles()
            self._env.step(simulate=False)
            initial_clock[0] = self.sock_cloth.data.get('dressing_qa',{}).get('foot_collision_safety',{}).get('cloth_physics_clock',{})
            if not initial_clock[0].get('externally_stepped'):
                raise RuntimeError('Fixed-motion replay requires explicitly stepped Obi physics')
            if self._rollout_material_report is not None:
                self._rollout_material_report['initial_collision_corrected_particles_world']=self.sock_cloth.particles().tolist()
                self._rollout_material_report['initial_contact_qa']=self.sock_cloth.data.get('dressing_qa',{})
                snapshot=self._rollout_material_report.get('prepared_state_json')
                if snapshot:
                    (directory/'prepared_state.json').write_text(snapshot+'\n')
                    self._rollout_material_report['prepared_state_sha256']=hashlib.sha256(snapshot.encode()).hexdigest()
                (directory/'applied_material.json').write_text(json.dumps(self._rollout_material_report,indent=2)+'\n')
            (directory/'config.yaml').write_text(yaml.safe_dump(self.config,sort_keys=False))
        def observe(self):
            value = super().observe(); last.clear(); last.update(value); return value
        def command(self, command, previous=None):
            # The baseline commands already passed the position/rate limiter.
            # Re-limiting against changed contact feedback would change this experiment.
            replaying = self._cloth_contact_rollout_started
            result = super().command(command, None if replaying else previous)
            if replaying:
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
            qa=last.get('dressing_qa',{})
            stretch=SockDressingEnv.cloth_radius_qa(last.get('cloth',{}))
            with (directory/'live_frame_diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps({'frame':track_frame[0]-1,'dressing_qa':qa,'stretch':stretch,
                    'particles_world':last.get('cloth',{}).get('particles',[]),
                    'particle_velocities_world':last.get('cloth',{}).get('particle_velocities',[]),
                    'particle_edges':last.get('cloth',{}).get('particle_edges',[]),
                    'particle_rest_edge_lengths':last.get('cloth',{}).get('particle_rest_edge_lengths',[]),
                    'particle_surface_triangles':last.get('cloth',{}).get('particle_surface_triangles',[]),
                    'opening_particle_indices':last.get('cloth',{}).get('opening_particle_indices',[]),
                    'scene_geometry':last.get('cloth',{}).get('scene_geometry',{}),
                    'grasp_state':last.get('cloth',{}).get('grasp_state',[])})+'\n')
            print(f"frame {track_frame[0]} safety="+json.dumps(last.get('dressing_qa',{}).get('foot_collision_safety',{})),flush=True)
            print('metrics '+json.dumps({'surface':qa.get('surface_containment_ratio'),
                'sections':qa.get('sections'), 'stretch':stretch.get('edge_classes'),
                'obi_penetration_m':qa.get('obi_maximum_cloth_foot_penetration_m')}),flush=True)
            return result
        def measure(self,rgb,masks):
            return renderer_measurements(last, masks)

    try:
        result=run_demo(config,prepared=prepared_replay_assets(config,directory),
            output_root=directory,sock_points=None,leg_points=None,max_steps=args.steps,seed=0,
            environment_factory=ReplayEnvironment,policy_factory=ReplayPolicy,
            perception_factory=RendererMeasurements)
    finally:
        if side_video[0] is not None: side_video[0].release()
    metadata_paths=list(directory.glob('data_sock_sim_smoke/train/phase4_*/metadata.json'))
    metadata_path=max(metadata_paths,key=lambda p:p.stat().st_mtime)
    metadata=json.loads(metadata_path.read_text())
    (directory/'last_observed_cloth_state.json').write_text(json.dumps(dict(
        dressing_qa=last.get('dressing_qa',{}),cloth=last.get('cloth',{})),indent=2,
        default=lambda value: value.tolist() if isinstance(value,np.ndarray) else value.item())+'\n')
    metadata.update(mode='fixed_motion_replay',replay_engine_sha256=engine_hash,
        frames_inferred=0,frames_replayed=metadata['frames'],
        replay_source=str(actions),
        replay_source_sha256=hashlib.sha256(actions.read_bytes()).hexdigest(),
        replay_command_maximum_error=max(errors,default=0),
        rollout_initial_cloth_physics_clock=initial_clock[0],
        perception='renderer RGB/depth/masks (fixed command replay)',
        command_contract='recorded applied 18-D commands -> position bounds/mimic -> RCareWorld',
        original_policy_checkpoint=config['inference']['checkpoint'],
        perception_source='renderer_measurements',policy_source='recorded_applied_commands',
        side_video=str(directory/'side_demo.mp4'))
    metadata['native_player_executable']=str(executable)
    # Keep the baseline preparation values explicitly in the rollout profile,
    # while the saved active expected/requested values match the candidate.
    config['obi']['expected']=copy.deepcopy(parameters)
    config['obi']['requested']=copy.deepcopy(parameters)
    (directory/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    metadata_path.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    (directory/'result.json').write_text(json.dumps({'result':result,'metadata':str(metadata_path)},indent=2)+'\n')
    print('METADATA '+str(metadata_path),flush=True)

if __name__=='__main__': main()
