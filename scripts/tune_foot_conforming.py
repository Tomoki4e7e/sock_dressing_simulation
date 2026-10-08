#!/usr/bin/env python3
"""Controlled, fixed-command contact ablations and material comparisons."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib
import json
import fcntl
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.config import load_config,resolve_package_path
from scripts.summarize_foot_contact_replay import BASELINE,export_diagnostics

OUTPUT=ROOT/'artifacts/phase4/foot-conforming-fixed-motion'
LEVELS={'friction':[.1,.2,.5],'damping':[.05,.2,.8],
        'stretch_compliance':[.00001,.0001,.0005],'bend_compliance':[.005,.02,.1],
        'particle_radius_m':[.002,.004,.008],'collision_margin_m':[.0005,.002]}


def recorded_transport_port(directory: Path, steps: int) -> int | None:
    """Preserve the recorded port when checking a complete replay's cache.

    replay_foot_contact.py still checks the full configuration and source
    identity before reusing anything. Its identity includes the TCP port;
    changing a port solely because of a different worker slot wastes a replay.
    """
    try:
        result = json.loads((directory/'result.json').read_text())
        metadata = json.loads(Path(result['metadata']).read_text())
        if metadata.get('frames') != steps or metadata.get('stop_reason') != 'max_steps':
            return None
        port = yaml.safe_load((directory/'config.yaml').read_text())['rcareworld']['port']
        return port if isinstance(port, int) and not isinstance(port, bool) and 0 < port < 65535 else None
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return None


def condition_identity(parameters, *, steps=40, mode='local', transport='local', fixed=True,
                       substeps=8, iterations=20, shape='skin-hull'):
    return dict(parameters=parameters, steps=steps, mode=mode, transport=transport,
                fixed=fixed, substeps=substeps, iterations=iterations, collider_shape=shape)


def condition_name(identity, label='material'):
    return label+'_'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]


def active_transport_port(directory: Path) -> int | None:
    """Wait for an existing recorder on its own port, without reusing partial data."""
    try:
        with (directory/'replay.lock').open('r') as lock:
            try:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                port=yaml.safe_load((directory/'config.yaml').read_text())['rcareworld']['port']
                return port if isinstance(port,int) and not isinstance(port,bool) and 0<port<65535 else None
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    except (OSError,ValueError,KeyError,TypeError,yaml.YAMLError):
        pass
    return None


def transport_groups(pending, base_port, workers):
    """Keep recorded ports and avoid collisions if a cache check rejects reuse."""
    pending=list(pending)
    while pending:
        reserved={port for _,port in pending if port is not None}
        group=[];claimed=set();deferred=[]
        for condition,port in pending:
            if len(group)>=workers or port is not None and port in claimed:
                deferred.append((condition,port));continue
            if port is None:
                port=base_port
                while port in reserved or port in claimed:port+=4
            claimed.add(port);group.append((condition,port))
        pending=deferred
        yield group


def assess(path:Path,steps:int)->dict:
    m=json.loads(path.read_text());rows=m.get('dressing_quality_by_frame',[])
    safety=[r.get('foot_collision_safety',{}) for r in rows]
    expected=np.loadtxt(BASELINE/'applied_action.csv',delimiter=',',ndmin=2)[:steps]
    action_path=path.parent/'applied_action.csv'
    actual=np.loadtxt(action_path,delimiter=',',ndmin=2) if action_path.stat().st_size else np.empty((0,18))
    error=float(np.max(np.abs(actual-expected))) if actual.size and actual.shape==expected.shape else None
    diagnostic=len(rows)==steps and all(s.get('contact_safety_armed') and 'maximum_original_grasp_anchor_error_m' in s for s in safety)
    config_path=path.parents[3]/'config.yaml'
    grasp_slip_threshold=.2
    physical_steps_per_frame=18
    requested_iterations=20
    requested_substeps=8
    if config_path.exists():
        config=yaml.safe_load(config_path.read_text())
        grasp_slip_threshold=float(config['obi']['expected']['slip_constraint_error_m'])
        # Preserve the baseline's existing observation schedule: one state
        # step, RGB/depth/ID, sock and leg amodal masks, front and side RGB.
        scene=config.get('scene',{})
        leg_ids=scene.get('human_leg_mask_ids') or scene.get('human_foot_collider_ids') or [scene.get('human_id')]
        observation_steps=1+3+1+len(leg_ids)+2
        physical_steps_per_frame=int(config.get('inference',{}).get('physics_steps_per_action',10))+observation_steps
        active_parameters=config.get('foot_conforming_rollout',{}).get('parameters',config['obi']['expected'])
        requested_iterations=int(active_parameters.get('solver_iterations',20))
        requested_substeps=int(active_parameters.get('substeps',8))
    material_path=path.parents[3]/'applied_material.json'
    material=json.loads(material_path.read_text()) if material_path.exists() else {}
    effective_iterations=material.get('actual',{}).get('effective_constraint_iterations',{})
    coupled_iterations=(set(effective_iterations)=={'distance','bending','collision','particle_collision','pin'} and
                        all(value==requested_iterations for value in effective_iterations.values()))
    clocks=[m.get('rollout_initial_cloth_physics_clock') or {}]+[s.get('cloth_physics_clock') or {} for s in safety]
    fixed_clock=all(c.get('externally_stepped') for c in clocks) and len(clocks)==steps+1
    if fixed_clock:
        fixed_clock=all(
            after.get('fixed_steps',-1)-before.get('fixed_steps',-1)==physical_steps_per_frame and
            np.isclose(after.get('simulated_time_s',0)-before.get('simulated_time_s',0),
                       physical_steps_per_frame*after.get('last_step_time_s',0),atol=1e-5,rtol=0)
            for before,after in zip(clocks,clocks[1:]))
    checks={'completed':m.get('frames')==steps and m.get('stop_reason')=='max_steps',
            'preparation_qa_not_overridden':not m.get('pre_inference_drape',{}).get('continued_for_diagnostics',False),
            'exact_commands':error is not None and error<=1e-12,'diagnostics':diagnostic,
            'fixed_cloth_physics_time':fixed_clock,
            'accepted_geometry_matches_render':diagnostic and all(
                s.get('maximum_render_position_error_m',float('inf'))<=1e-5 and
                s.get('maximum_end_position_error_m',float('inf'))<=1e-5 for s in safety),
            'validated_material_application':bool(material.get('validation',{}).get('ok')),
            'validated_preparation_state':bool(material.get('prepared_state_readback_validation',{}).get('ok')),
            'native_pins_reference_original_grippers':diagnostic and all(
                s.get('native_grasp_binding',{}).get('all_original_targets_match') is True and
                s.get('native_grasp_binding',{}).get('expected_pin_count')==10 and
                s.get('native_grasp_binding',{}).get('matched_pin_count')==10
                for s in safety),
            'particle_contact_shell_matches_native_radius':diagnostic and all(
                s.get('particle_contact_shell_policy')=='particle_radius_with_numerical_separation' and
                np.isfinite(s.get('particle_contact_numerical_separation_m',float('inf'))) and
                0<=s.get('particle_contact_numerical_separation_m',float('inf'))<=.00002 and
                np.isfinite(s.get('particle_contact_shell_m',float('inf'))) and
                np.isclose(s.get('particle_contact_shell_m',float('inf')),
                    material.get('actual',{}).get('particle_radius_m',float('inf'))+
                    s.get('particle_contact_numerical_separation_m',float('inf')),atol=1e-7,rtol=0)
                for s in safety),
            'coupled_solver_iterations_applied':coupled_iterations,
            'no_remaining_sweeps':diagnostic and all(s['remaining_swept_intersections']==0 for s in safety),
            'no_remaining_surfaces':diagnostic and all(s['remaining_surface_intersections']==0 for s in safety),
            'no_remaining_surface_motion':diagnostic and all(s.get('continuous_surface_motion_checked',False) and
                s.get('remaining_swept_surface_intersections',-1)==0 for s in safety),
            'local_surface_history_applied':diagnostic and all(s.get('surface_reference_policy')=='per_triangle_accepted_segments' for s in safety),
            'surface_bounds_match_collider_geometry':diagnostic and all(s.get('surface_bounds_policy')=='collider_geometry' for s in safety),
            'native_continuous_surface_contact_applied':diagnostic and all(s.get('native_particle_ccd',0)==1 and
                s.get('native_surface_collision_tolerance_m',float('inf'))<=.00010001 for s in safety),
            'penetration_within_2mm':diagnostic and all(r['geometric_maximum_cloth_foot_penetration_m']<=.002 for r in rows),
            'foot_skin_covered':diagnostic and all(s['foot_skin_vertex_count']>0 and s['uncovered_foot_skin_vertices']==0 for s in safety),
            'cloth_faces_checked':diagnostic and all(s.get('cloth_surface_triangle_count',0)>0 for s in safety),
            'native_surface_contact_applied':diagnostic and all(s.get('surface_collisions_enabled') and
                s.get('native_collision_triangles',0)>=s.get('cloth_surface_triangle_count',1) for s in safety),
            'continuous_grasp':bool(m.get('task_success',{}).get('continuous_grasp_ok')),
            'within_original_grasp_slip_threshold':diagnostic and all(s['maximum_original_grasp_anchor_error_m']<=grasp_slip_threshold for s in safety),
            'grasp_offset_drift_within_2mm':diagnostic and all(s['maximum_grasp_offset_drift_m']<=.002 for s in safety)}
    tail=rows[-10:]
    anatomical_available=bool(rows) and all(r.get('anatomical_foot_conformity',{}).get('valid') for r in rows)
    if any('anatomical_foot_conformity' in r for r in rows) or any(s.get('foot_collider_shape_mode')=='skin-hull' for s in safety):
        checks['complete_anatomical_surface_diagnostics']=anatomical_available
        if any(s.get('foot_collider_shape_mode')=='skin-hull' for s in safety):
            checks['native_foot_mesh_registration_verified']=all(s.get('native_foot_mesh_registration_verified',False) for s in safety)
            checks['accepted_substep_velocity_synchronized']=all(
                s.get('substep_velocity_projection_enabled',False) and
                s.get('velocity_sync_preserves_native_damping_and_friction',False) and
                s.get('native_substep_velocity_synchronizations',-1)==physical_steps_per_frame*requested_substeps and
                np.isfinite(s.get('maximum_velocity_position_error_m_s',float('inf'))) and
                s.get('maximum_velocity_position_error_m_s',float('inf'))<=1e-5
                for s in safety)
    def coverage_row(row):
        return row['anatomical_foot_conformity'] if anatomical_available else row
    if anatomical_available:
        checks['anatomical_toes_forefoot_samples_valid']=all(
            all(next((section.get('sample_source')=='skin_envelope' and section.get('total_samples')==8
                      for section in r['anatomical_foot_conformity']['sections'] if section['name']==name),False)
                for name in ('toes','forefoot')) for r in rows)
        checks['human_skin_faces_covered']=all(s.get('skin_face_coverage_verified',False) for s in safety)
    def section(name):
        return float(np.mean([next((s['containment_ratio'] for s in coverage_row(r).get('sections',[]) if s['name']==name),0) for r in tail])) if tail else 0
    all_cloth=m.get('cloth_quality_by_frame',[])
    checks['complete_cloth_diagnostics']=len(all_cloth)==steps
    cloth=all_cloth[-10:]
    violations=0;largest_stretch=0
    for r in cloth:
        edges=r.get('stretch',{}).get('edge_classes',{})
        bad=not r.get('stretch',{}).get('available',False) or not any(
            values.get('maximum_stretch') is not None for values in edges.values())
        for name,values in edges.items():
            value=values.get('maximum_stretch')
            if value is None:continue
            limit=2.0 if name=='opening_rim' else 1.1 if name=='opening_body' else 1.5
            largest_stretch=max(largest_stretch,float(value))
            bad=bad or value>limit*1.01
        violations+=int(bad)
    result={'metadata':str(path),'steps':steps,'checks':checks,'safe':all(checks.values()),
        'containment_metric':'anatomical_skin_envelope' if anatomical_available else 'legacy_box_surfaces',
        'mean_tail_surface_containment':float(np.mean([coverage_row(r)['surface_containment_ratio'] for r in tail])) if tail else 0,
        'legacy_mean_tail_surface_containment':float(np.mean([r['surface_containment_ratio'] for r in tail])) if tail else 0,
        'legacy_mean_tail_toes_containment':float(np.mean([next((s['containment_ratio'] for s in r.get('sections',[]) if s['name']=='toes'),0) for r in tail])) if tail else 0,
        'legacy_mean_tail_forefoot_containment':float(np.mean([next((s['containment_ratio'] for s in r.get('sections',[]) if s['name']=='forefoot'),0) for r in tail])) if tail else 0,
        'mean_tail_toes_containment':section('toes'),'mean_tail_forefoot_containment':section('forefoot'),
        'tail_stretch_violation_frames':violations,'tail_maximum_stretch':largest_stretch,
        'maximum_contact_correction_m':max((s.get('maximum_correction_m',0) for s in safety),default=0),
        'maximum_anchor_error_m':max((s.get('maximum_original_grasp_anchor_error_m',0) for s in safety),default=0),
        'configured_grasp_slip_threshold_m':grasp_slip_threshold,
        'maximum_offset_drift_m':max((s.get('maximum_grasp_offset_drift_m',0) for s in safety),default=0),
        'maximum_post_penetration_m':max((r.get('geometric_maximum_cloth_foot_penetration_m',0) for r in rows),default=0),
        'maximum_command_error':error,'original_task_success':m.get('task_success',{}).get('success',False),
        'original_failed_gates':m.get('task_success',{}).get('failed_gates',[]),
        'stage_maximum_displacement_m':{k:max(s.get('stage_maximum_displacement_m',{}).get(k,0) for s in safety)
            for k in {key for s in safety for key in s.get('stage_maximum_displacement_m',{})}}}
    result['conforming_passed']=result['safe'] and steps==100 and result['mean_tail_toes_containment']>=.75 and result['mean_tail_forefoot_containment']>=.5 and result['mean_tail_surface_containment']>=.25 and violations==0
    return result


def score(row):
    # Sample containment is discrete. Micrometre-scale floating point noise
    # must not outrank the lower-cost solver when physical outcomes coincide.
    return (not row['safe'],-round(row['mean_tail_surface_containment'],5),
            row['tail_stretch_violation_frames'],round(row['maximum_contact_correction_m'],5))


def choose_grasp_transport(controls):
    """Use the matched, fixed-offset controls to resolve the transport choice."""
    eligible=[row for row in controls if row.get('safe') and row.get('mode')=='local'
              and row.get('fixed') is True and row.get('collider_shape')=='skin-hull'
              and row.get('transport') in ('local','bulk')]
    if not eligible:
        return 'local',None
    best=min(eligible,key=lambda row:(*score(row),row['transport']!='local'))
    return best['transport'],best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['controls','materials','materials-final','final','all'],default='all')
    parser.add_argument('--output-root',type=Path,default=OUTPUT)
    parser.add_argument('--port',type=int,default=5200)
    parser.add_argument('--player-executable',type=Path,help='Use one immutable native player for the whole comparison')
    parser.add_argument('--foot-shape',choices=['boxes','skin-hull'],default='boxes')
    parser.add_argument('--transport',choices=['auto','local','bulk'],default='auto',
                        help='Resolve auto from the matched fixed-offset source controls')
    parser.add_argument('--workers',type=int,choices=range(1,9),default=1,
                        help='Run independent conditions concurrently on distinct simulator ports')
    args=parser.parse_args();out=args.output_root;out.mkdir(parents=True,exist_ok=True)
    current_transport='local' if args.transport=='auto' else args.transport
    base=load_config(ROOT/'config/opening_widen_w95.yaml')['obi']['expected']
    base={k:base[k] for k in LEVELS}
    results=[]
    results_lock=threading.Lock()
    common_state=out/'common_preparation.json'
    common_identity=out/'common_preparation_kernel.json'
    player=(args.player_executable or resolve_package_path(load_config(ROOT/'config/opening_widen_w95.yaml')['rcareworld']['executable'])).resolve()
    kernel=player.parent/'Player_Data/Managed'
    kernel_identity={name:hashlib.sha256((kernel/name).read_bytes()).hexdigest()
                     for name in ('Obi.dll','RCareWorld.dll')}
    if common_state.exists() and (not common_identity.exists() or
        json.loads(common_identity.read_text())!=kernel_identity):
        raise RuntimeError('Common preparation belongs to another compiled kernel; use a new comparison directory')
    def trial(parameters,steps=40,mode='local',transport='local',fixed=True,substeps=8,iterations=20,label='material',port_slot=0,shape=None,assigned_port=None):
        shape=shape or ('boxes' if mode=='legacy' else args.foot_shape)
        identity=condition_identity(parameters,steps=steps,mode=mode,transport=transport,fixed=fixed,
                                    substeps=substeps,iterations=iterations,shape=shape)
        name=condition_name(identity,label)
        directory=out/name;directory.mkdir(exist_ok=True);override=directory/'material.yaml';override.write_text(yaml.safe_dump(parameters))
        command=[sys.executable,'-u',str(ROOT/'scripts/replay_foot_contact.py'),'--output-root',str(out),
            '--candidate-name',name,'--steps',str(steps),'--substeps',str(substeps),'--iterations',str(iterations),
            '--surface-mode',mode,'--transport',transport,'--post-preparation-material','--material-overrides',str(override),
            '--port',str(assigned_port if assigned_port is not None else args.port+4*port_slot)]
        command+=['--player-executable',str(player)]
        command+=['--foot-shape',shape]
        if fixed:command+=['--fixed-grasp-offsets']
        if common_state.exists():command+=['--prepared-state',str(common_state.resolve())]
        if any(hashlib.sha256((kernel/file).read_bytes()).hexdigest()!=digest for file,digest in kernel_identity.items()):
            raise RuntimeError('Native player changed during the comparison')
        print('START '+name+' '+json.dumps(identity),flush=True)
        # An older recorder may still own the condition lock. Append rather
        # than truncating its stdout while the new invocation waits for it.
        with (directory/'replay.log').open('a') as log:
            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:raise RuntimeError(f'Candidate crashed: {name}; see replay.log')
        record=json.loads((directory/'result.json').read_text());row=assess(Path(record['metadata']),steps)
        if not common_state.exists():
            metadata=json.loads(Path(record['metadata']).read_text())
            if not metadata.get('pre_inference_drape',{}).get('passed'):
                raise RuntimeError('Cannot establish common preparation from failed drape QA')
            snapshot=directory/'prepared_state.json'
            if not snapshot.exists():raise RuntimeError('Canonical prepared state was not captured')
            common_state.write_text(snapshot.read_text())
            common_identity.write_text(json.dumps(kernel_identity,indent=2)+'\n')
        row.update(condition=name,parameters=parameters,substeps=substeps,iterations=iterations,mode=mode,transport=transport,fixed=fixed,collider_shape=shape)
        export_diagnostics(Path(record['metadata']),directory/'frame_diagnostics.csv')
        (directory/'assessment.json').write_text(json.dumps(row,indent=2)+'\n')
        with results_lock:
            results.append(row)
            (out/'trials-current.json').write_text(json.dumps(results,indent=2)+'\n')
        print('RESULT '+json.dumps({k:row[k] for k in ['condition','safe','mean_tail_surface_containment','tail_stretch_violation_frames','maximum_anchor_error_m','conforming_passed']}),flush=True)
        return row
    def batch(conditions):
        # Batch barriers keep each port exclusive, including slow conditions.
        # Dependencies (each coordinate update) are evaluated between batches.
        if args.workers>1 and not common_state.exists():
            raise RuntimeError('Concurrent comparisons require an already validated common preparation')
        rows=[];pending=[]
        for condition in conditions:
            condition={**condition,'transport':condition.get('transport',current_transport)}
            identity=condition_identity(condition['parameters'], **{key:condition[key] for key in
                ('steps','mode','transport','fixed','substeps','iterations') if key in condition},
                shape=condition.get('shape') or ('boxes' if condition.get('mode')=='legacy' else args.foot_shape))
            name=condition_name(identity,condition.get('label','material'))
            pending.append((condition,recorded_transport_port(out/name,identity['steps']) or active_transport_port(out/name)))
        for group in transport_groups(pending,args.port,args.workers):
            # Cached records can share a historical port. Check those in
            # separate groups so even a rejected cache cannot collide with
            # another native player. New conditions avoid every reserved port.
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures=[executor.submit(trial,**condition,assigned_port=port)
                         for condition,port in group]
                rows.extend(future.result() for future in futures)
        return rows
    if args.phase in ('controls','all'):
        control_conditions=[dict(parameters=base,mode=mode,transport=transport,fixed=fixed,label=label)
            for mode,transport,fixed,label in [('legacy','bulk',False,'control_legacy'),
                ('local','bulk',False,'local_surface'),('local','bulk',True,'fixed_grasp'),
                ('local','local',True,'local_transport')]]
        if args.foot_shape=='skin-hull':
            # This geometry control has the same independent preparation as
            # the other controls; it need not wait for their recordings.
            control_conditions.append(dict(parameters=base,shape='boxes',transport='local',label='local_box_geometry'))
        controls=batch(control_conditions)
        (out/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    if args.transport=='auto':
        controls_path=out/'controls.json'
        controls=json.loads(controls_path.read_text()) if controls_path.exists() else []
        current_transport,selected_control=choose_grasp_transport(controls)
        (out/'transport_selection.json').write_text(json.dumps({
            'transport':current_transport,'selected_control':selected_control,
            'source':'matched_safe_fixed_offset_controls' if selected_control else 'local_diagnostic_fallback',
            'reason':'Prioritize anatomical containment, then stretch violations and contact corrections; keep local transport on a tie. Material screening keeps the selected source behavior fixed.'},indent=2)+'\n')
    if args.phase in ('materials','materials-final','all'):
        conditions=[dict(parameters=base,label='material')]
        for key,values in LEVELS.items():
            for value in values:
                if value!=base[key]:conditions.append(dict(parameters={**base,key:value},label='material'))
        screening=batch(conditions)
        current=min(screening,key=score);selected=dict(current['parameters'])
        for sweep in range(2):
            changed=False
            for key,values in LEVELS.items():
                candidates=batch([dict(parameters={**selected,key:value},label='material')
                                  for value in values if value!=selected[key]])
                for candidate in candidates:
                    if candidate['safe'] and score(candidate)<score(current):
                        current=candidate;selected=dict(candidate['parameters']);changed=True
            if not changed:break
        (out/'selected_material.json').write_text(json.dumps(current,indent=2)+'\n')
        (out/'material_comparison.json').write_text(json.dumps([r for r in results if r['condition'].startswith('material_')],indent=2)+'\n')
        if not current['safe']:
            (out/'material_selection_failure.json').write_text(json.dumps(dict(
                reason='No material passed the fixed-motion safety checks after screening and coordinate comparisons',
                failed_checks=[name for name,passed in current['checks'].items() if not passed],
                diagnostic_candidate=current['condition'],selected_for_final=False),indent=2)+'\n')
            raise RuntimeError('No safe material; repair contact processing before the final 100-command grid')
    if args.phase in ('final','materials-final','all'):
        material_record=json.loads((out/'selected_material.json').read_text())
        selected=material_record['parameters']
        current_transport=material_record.get('transport',current_transport)
        finals=batch([dict(parameters=selected,steps=100,substeps=s,iterations=i,label='final')
                      for s,i in [(8,20),(8,40),(16,20),(16,40)]])
        passing=[r for r in finals if r['conforming_passed']];passing.sort(key=lambda r:(*score(r),r['substeps']*r['iterations']))
        (out/'final_comparison.json').write_text(json.dumps({'conditions':finals,'best':passing[0]['condition'] if passing else None,'visual_review_required':True},indent=2)+'\n')
        if not passing:raise RuntimeError('No conforming candidate; contact/strain corrections require further work')

if __name__=='__main__':main()
