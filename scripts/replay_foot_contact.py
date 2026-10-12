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
from sock_dressing_simulation.perception import PerceptionResult, assess_masks, SAMDepthPerception
from sock_dressing_simulation.policy import SAMDAMSARNNPolicy
from sock_dressing_simulation.scenario import scenario_from_config

BASELINE = ROOT / 'artifacts/phase4/opening-short-edge-widen/final100-w95/data_sock_sim_smoke/train/phase4_20261005T011108Z'
DEFAULT_OUTPUT = ROOT / 'artifacts/phase4/foot-nonpenetration-fixed-motion'


def apply_total_cloth_mass(config, prepared_state):
    """Translate total cloth mass to the existing uniform native particle API."""
    total = config['obi'].get('total_mass_kg')
    if total is None:
        return
    if prepared_state is None:
        raise ValueError('Total cloth mass requires a prepared state with particle count')
    total = float(total)
    count = len(json.loads(Path(prepared_state).read_text())['positions'])
    if not np.isfinite(total) or total<=0 or count<=0:
        raise ValueError('Total cloth mass and particle count must be positive')
    config['obi']['expected']['particle_mass_kg'] = total/count
    config['obi']['total_mass_particle_count'] = count


def validate_total_cloth_mass(config, particle_count, actual):
    total = config['obi'].get('total_mass_kg')
    if total is None:
        return None
    measured = float(actual['particle_mass_kg'])*particle_count
    ok = particle_count==config['obi']['total_mass_particle_count'] and np.isclose(
        measured,float(total),rtol=1e-6,atol=1e-9)
    if not ok:
        raise RuntimeError(f'Native total cloth mass mismatch: {measured} kg / {particle_count} particles')
    return dict(requested_total_mass_kg=float(total),particle_count=particle_count,
                actual_particle_mass_kg=float(actual['particle_mass_kg']),
                actual_total_mass_kg=measured,ok=True,
                distribution='uniform finite mass on every cloth particle, including grasped particles')



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
    parser.add_argument('--config',type=Path,default=ROOT/'config/opening_widen_w95.yaml')
    parser.add_argument('--substeps', type=int, choices=(8,16), default=8)
    parser.add_argument('--iterations', type=int, choices=(20,40), default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--autonomous', action='store_true',
                        help='Generate every command with fresh closed-loop SAMDAMSARNN inference; never load recorded commands')
    parser.add_argument('--device', choices=('cpu','cuda'), default=None)
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
    parser.add_argument('--foot-views', action='store_true',
                        help='Record synchronized vertical top/bottom views without additional physics steps')
    parser.add_argument('--physics-video', action='store_true',
                        help='Record side RGB after every existing 0.02 s fixed step without advancing physics')
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
    config = load_config(args.config)
    speed_matched = bool(config.get('real_speed', {}).get('enabled', False))
    args.device = args.device or ('cuda' if speed_matched else 'cpu')
    if speed_matched and (not args.autonomous or not args.physics_video):
        raise ValueError('Real-speed profile requires autonomous inference and physics video')
    if config['obi'].get('total_mass_kg') is not None and not args.autonomous:
        raise ValueError('Total-mass override is scoped to the new autonomous run')
    apply_total_cloth_mass(config,args.prepared_state)
    if args.player_executable:config['rcareworld']['executable']=str(args.player_executable.resolve())
    native_player=resolve_package_path(config['rcareworld']['executable']).resolve()
    native_kernel=native_player.parent/'Player_Data/Managed'
    clock_source = native_player.parent/'runtime_sources/SockGraspMotion.cs'
    config['physics_video_non_destructive_clock'] = clock_source.exists() and 'GetClothPhysicsClock' in clock_source.read_text()
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
    actions = None
    values = None
    if not args.autonomous:
        actions = BASELINE / 'applied_action.csv'
        values = np.loadtxt(actions, delimiter=',', ndmin=2)
        if values.shape != (100,18): raise ValueError('baseline must contain exactly 100 commands')
    config['rcareworld']['graphics'] = True
    config['rcareworld']['port'] = args.port or 5100 + (2 if args.substeps == 16 else 0) + (1 if args.iterations == 40 else 0)
    if not args.post_preparation_material:
        config['obi']['substeps'] = args.substeps
        config['obi']['solver_iterations'] = args.iterations
    config['inference'].update(reference_actions=None if args.autonomous else str(actions),
        reference_action_blend=0.0 if args.autonomous else 1.0,
        reference_action_hold_steps=1,reference_action_interpolation='hold',device=args.device)
    if args.autonomous:
        config['inference']['reference_cartesian_pull_m']=0.0
        config['inference']['policy_catch_up']={'enabled':False}
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
    for source in [Path(__file__).resolve(),ROOT/'sock_dressing_simulation/real_speed.py',ROOT/'sock_dressing_simulation/demo.py',
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
    if args.foot_views:
        from foot_containment_demo import FootViewRecorder
        engine_hash=hashlib.sha256((engine_hash+'foot_views').encode()+
            (ROOT/'scripts/foot_containment_demo.py').read_bytes()).hexdigest()
    cache=directory/'result.json'
    if cache.exists() and not args.autonomous:
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
            ROOT/'sock_dressing_simulation/real_speed.py',
            ROOT/'RCareUnity/Assets/RCareCommon/Scripts/Attributes/ControllerAttr.cs',
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
    if args.foot_views:
        for filename in ('foot_containment_demo.py','report_natural_opening.py'):
            source = ROOT/'scripts'/filename
            shutil.copyfile(source,sources/filename)
            source_manifest[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
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
    if args.autonomous:
        if (directory/'live_frame_diagnostics.jsonl').exists():
            raise RuntimeError('Fresh autonomous run requires a new output directory')
        inference_sources={}
        for source in [ROOT/'sock_dressing_simulation/policy.py',
                       ROOT/'sock_dressing_simulation/perception.py',
                       resolve_package_path(config['inference']['checkpoint']),
                       resolve_package_path(config['inference']['stats'])]:
            shutil.copyfile(source,sources/source.name)
            inference_sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        (directory/'inference-sources.json').write_text(json.dumps(inference_sources,indent=2)+'\n')
    (directory/'live_frame_diagnostics.jsonl').write_text('')
    executable=resolve_package_path(config['rcareworld']['executable']).resolve()
    wrapper=directory/'player.sh'
    wrapper.write_text('#!/bin/bash\nexport SOCK_GRASP_MOTION_LOG='+shlex.quote(str((directory/'grasp-substeps.jsonl').resolve()))+'\nexport SOCK_CONTACT_PROGRESS_PATH='+shlex.quote(str((directory/'contact-progress.log').resolve()))+'\nexec '+shlex.quote(str(executable))+
        ' -logFile '+shlex.quote(str((directory/'player.log').resolve()))+' "$@"\n')
    wrapper.chmod(0o755)
    config['rcareworld']['executable']=str(wrapper.resolve())
    (directory/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    last = {}; side_video = [None]; track_frame = [0]; errors = []; initial_clock = [None]
    foot_views = [None]
    physics_video = [None]
    speed_rollout = [None]
    commanded_frame = [0]
    inferred_frames = [0]

    class ReplayEnvironment(SockDressingEnv):
        refresh_observation_after_contact_start = args.autonomous
        def observe_restored_contact_state(self):
            self._read_only_native_observation = True
            try:
                observation = self.observe()
                clock = observation['dressing_qa']['foot_collision_safety']['cloth_physics_clock']
                if clock != initial_clock[0]:
                    raise RuntimeError('Initial inference observation advanced restored physics')
                return observation
            finally:
                self._read_only_native_observation = False
        def close(self):
            try:
                if speed_rollout[0] is not None:
                    speed_rollout[0].close()
                    speed_rollout[0] = None
                if physics_video[0] is not None:
                    physics_video[0].close()
                    physics_video[0] = None
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
            settling=self.config.get('contact_settling',{})
            if settling.get('enabled',False):
                if not speed_matched:
                    raise ValueError('Contact settling requires the real-speed profile')
                from sock_dressing_simulation.real_speed import SpeedMatchedRollout
                from sock_dressing_simulation.physics_video import PhysicsStepRecorder
                settling_directory=directory/'settling'
                settling_directory.mkdir(exist_ok=True)
                holding=SpeedMatchedRollout(self,settling_directory,lambda:0,Path(prepared['runtime_urdf']))
                recording=PhysicsStepRecorder(self,settling_directory,lambda:0)
                stable=0
                report=dict(passed=False,policy_commands=0,settings=settling,samples=[])
                try:
                    for step in range(1,int(settling['maximum_steps'])+1):
                        holding.step()
                        self.sock_cloth.request_particle_velocities()
                        self._env.step(simulate=False)
                        velocities=self.sock_cloth.particle_velocities()
                        speeds=np.linalg.norm(velocities,axis=1)
                        sample=dict(step=step,speed_p95_m_s=float(np.quantile(speeds,.95)),
                            maximum_speed_m_s=float(speeds.max()),mean_vertical_speed_m_s=float(velocities[:,1].mean()))
                        report['samples'].append(sample)
                        quiet=(sample['speed_p95_m_s']<=settling['maximum_speed_p95_m_s'] and
                            sample['maximum_speed_m_s']<=settling['maximum_particle_speed_m_s'] and
                            sample['mean_vertical_speed_m_s']<=settling['maximum_mean_upward_speed_m_s'])
                        stable=stable+1 if quiet else 0
                        if step%10==0:
                            print('contact settling '+json.dumps(sample),flush=True)
                        if step>=settling['minimum_steps'] and stable>=settling['consecutive_stable_steps']:
                            report['passed']=True
                            break
                finally:
                    recording.close()
                    holding.close()
                    report['physical_steps']=len(report['samples'])
                    (settling_directory/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
                if not report['passed']:
                    raise RuntimeError('Grasped cloth did not physically settle; see settling/validation.json')
            self.sock_cloth.request_dressing_qa()
            self.sock_cloth.request_particles()
            if args.foot_views:
                self.sock_cloth.request_scene_geometry()
            self._env.step(simulate=False)
            initial_clock[0] = self.sock_cloth.data.get('dressing_qa',{}).get('foot_collision_safety',{}).get('cloth_physics_clock',{})
            if not initial_clock[0].get('externally_stepped'):
                raise RuntimeError('Fixed-motion replay requires explicitly stepped Obi physics')
            if args.foot_views:
                foot_views[0] = FootViewRecorder(self, directory)
                foot_views[0].initialize(dict(self.sock_cloth.data))
            if self._rollout_material_report is not None:
                mass_check = validate_total_cloth_mass(self.config,
                    len(self.sock_cloth.particles()),self._rollout_material_report['actual'])
                if mass_check is not None:
                    self._rollout_material_report['total_mass_validation']=mass_check
                    (directory/'cloth-mass-validation.json').write_text(json.dumps(mass_check,indent=2)+'\n')
                self._rollout_material_report['initial_collision_corrected_particles_world']=self.sock_cloth.particles().tolist()
                self._rollout_material_report['initial_contact_qa']=self.sock_cloth.data.get('dressing_qa',{})
                snapshot=self._rollout_material_report.get('prepared_state_json')
                if snapshot:
                    (directory/'prepared_state.json').write_text(snapshot+'\n')
                    self._rollout_material_report['prepared_state_sha256']=hashlib.sha256(snapshot.encode()).hexdigest()
                (directory/'applied_material.json').write_text(json.dumps(self._rollout_material_report,indent=2)+'\n')
            (directory/'config.yaml').write_text(yaml.safe_dump(self.config,sort_keys=False))
            if args.physics_video:
                from sock_dressing_simulation.physics_video import PhysicsStepRecorder
                physics_video[0] = PhysicsStepRecorder(self, directory, lambda: commanded_frame[0])
            if speed_matched:
                from sock_dressing_simulation.real_speed import SpeedMatchedRollout
                speed_rollout[0] = SpeedMatchedRollout(self, directory, lambda: commanded_frame[0],
                    Path(prepared['runtime_urdf']))
                self._read_only_native_observation = True
        def observe(self):
            if speed_matched and self._cloth_contact_rollout_started:
                self._read_only_native_observation = True
            value = super().observe()
            if foot_views[0] is not None:
                value['foot_views'] = foot_views[0].capture(value['cloth'])
            last.clear(); last.update(value); return value
        def command(self, command, previous=None):
            # The baseline commands already passed the position/rate limiter.
            # Re-limiting against changed contact feedback would change this experiment.
            replaying = self._cloth_contact_rollout_started
            if replaying:
                commanded_frame[0] += 1
            if speed_matched and replaying:
                return speed_rollout[0].command(command)
            result = super().command(command, None if replaying and not args.autonomous else previous)
            if replaying:
                errors.append(float(np.max(np.abs(result-command))))
            return result

        def advance_physics(self, steps):
            if speed_matched and self._cloth_contact_rollout_started:
                for _ in range(steps):
                    speed_rollout[0].step()
            else:
                super().advance_physics(steps)

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
            if foot_views[0] is not None:
                foot_views[0].record(last['foot_views'], last['cloth'], track_frame[0]-1)
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

    class AutonomousPerception(RendererMeasurements, SAMDepthPerception):
        def __init__(self, config):
            SAMDepthPerception.__init__(self, config)
        def initialize(self, rgb, **kwargs):
            return SAMDepthPerception.initialize(self, rgb, **kwargs)
        def measure(self, rgb, masks):
            return SAMDepthPerception.track(self, rgb, renderer_masks=masks)

    class AutonomousPolicy(SAMDAMSARNNPolicy):
        def step(self, **kwargs):
            result = super().step(**kwargs)
            inferred_frames[0] += 1
            return result

    prepared = prepared_replay_assets(config,directory)
    try:
        result=run_demo(config,prepared=prepared,
            output_root=directory,sock_points=None,leg_points=None,max_steps=args.steps,seed=0,
            environment_factory=ReplayEnvironment,
            policy_factory=AutonomousPolicy if args.autonomous else ReplayPolicy,
            perception_factory=AutonomousPerception if args.autonomous else RendererMeasurements)
    finally:
        if side_video[0] is not None: side_video[0].release()
        if foot_views[0] is not None: foot_views[0].close()
    metadata_paths=list(directory.glob('data_sock_sim_smoke/train/phase4_*/metadata.json'))
    metadata_path=max(metadata_paths,key=lambda p:p.stat().st_mtime)
    metadata=json.loads(metadata_path.read_text())
    (directory/'last_observed_cloth_state.json').write_text(json.dumps(dict(
        dressing_qa=last.get('dressing_qa',{}),cloth=last.get('cloth',{})),indent=2,
        default=lambda value: value.tolist() if isinstance(value,np.ndarray) else value.item())+'\n')
    metadata.update(mode='samdamsarnn_closed_loop' if args.autonomous else 'fixed_motion_replay',replay_engine_sha256=engine_hash,
        frames_inferred=inferred_frames[0] if args.autonomous else 0,
        frames_replayed=0 if args.autonomous else metadata['frames'],
        replay_source=None if args.autonomous else str(actions),
        replay_source_sha256=None if args.autonomous else hashlib.sha256(actions.read_bytes()).hexdigest(),
        replay_command_maximum_error=None if args.autonomous else max(errors,default=0),
        rollout_initial_cloth_physics_clock=initial_clock[0],
        perception='SAM2/Depth-Anything-V2 (existing renderer-mask configuration)' if args.autonomous else 'renderer RGB/depth/masks (fixed command replay)',
        command_contract='live model output -> existing joint bounds/rate/mimic -> RCareWorld' if args.autonomous else 'recorded applied 18-D commands -> position bounds/mimic -> RCareWorld',
        original_policy_checkpoint=config['inference']['checkpoint'],
        perception_source='SAMDepthPerception' if args.autonomous else 'renderer_measurements',
        policy_source='SAMDAMSARNNPolicy' if args.autonomous else 'recorded_applied_commands',
        side_video=str(directory/'side_demo.mp4'))
    metadata['native_player_executable']=str(executable)
    if speed_matched:
        metadata['real_speed'] = config['real_speed']
        metadata['command_contract'] = 'live model output -> joint bounds/mimic -> URDF velocity override and acceleration limits at 0.02 s -> PhysX drive'
    # Keep the baseline preparation values explicitly in the rollout profile,
    # while the saved active expected/requested values match the candidate.
    config['obi']['expected']=copy.deepcopy(parameters)
    config['obi']['requested']=copy.deepcopy(parameters)
    (directory/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    metadata_path.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    (directory/'result.json').write_text(json.dumps({'result':result,'metadata':str(metadata_path)},indent=2)+'\n')
    if args.foot_views and result['ok']:
        from foot_containment_demo import finalize
        finalize(directory, metadata_path, args.steps)
    elif args.foot_views:
        (directory/'recording_incomplete.json').write_text(json.dumps(dict(
            requested_commands=args.steps, completed_commands=metadata['frames'],
            inferred_commands=inferred_frames[0], stop_reason=metadata['stop_reason'],
            foot_view_frames=foot_views[0].frames if foot_views[0] is not None else 0,
            raw_partial_videos_preserved=True),indent=2)+'\n')
    print('METADATA '+str(metadata_path),flush=True)
    if not result['ok']:
        raise SystemExit(1)

if __name__=='__main__': main()
