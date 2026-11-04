#!/usr/bin/env python3
"""Independently summarize native replay traces and all acceptance gates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replay',type=Path,default=ROOT/'artifacts/compound-reaching/registered-replay-v6');ap.add_argument('--registration',type=Path,default=ROOT/'artifacts/compound-reaching/registration/candidate');args=ap.parse_args();tree=ET.parse(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf').getroot();metrics=[];series=[]
    for number in [10,38,50]:
        folder=args.replay/f'pose-{number:02d}';result=json.loads((folder/'result.json').read_text());rows=[json.loads(line) for line in (folder/'trace.jsonl').open()];assert [p['step'] for p in rows]==list(range(result['steps']))
        q=np.array([r['actual_q_rad'] for r in rows]);plan=json.loads((ROOT/result['source']).read_text());limits=[tree.find(f"joint[@name='{name}']/limit") for name in plan['joint_names']];lower=np.array([float(p.get('lower')) for p in limits]);upper=np.array([float(p.get('upper')) for p in limits]);velocity=np.array([float(p.get('velocity')) for p in limits]);acceleration=np.array([float(p.get('acceleration')) for p in limits]);clock=np.array([p['cloth_time_s'] for p in rows]);dt=float(np.round(np.median(np.diff(clock)),6));sampled_velocity=np.diff(q,axis=0)/dt;sampled_acceleration=np.diff(sampled_velocity,axis=0)/dt
        hard_limits=bool(np.all(q>=lower-1e-6) and np.all(q<=upper+1e-6));velocity_fraction=float(np.max(abs(sampled_velocity)/velocity));acceleration_fraction=float(np.max(abs(sampled_acceleration)/acceleration));same_source=hashlib.sha256((ROOT/result['source']).read_bytes()).hexdigest()==result['source_sha256'];clock_ok=result['final_steps']-result['initial_steps']==len(rows) and abs(result['final_clock']-result['initial_clock']-dt*len(rows))<1e-4
        pairs=Counter();depths={};moving_separations=[]
        for row in rows:
            for c in row['contacts']:
                pair=tuple(sorted([c['a_link'] or c['a'],c['b_link'] or c['b']]));pairs[pair]+=1;depths[pair]=max(depths.get(pair,0),-c['separation_m'])
                if c['a_link'].startswith(('right_arm/','right_gripper/')) or c['b_link'].startswith(('right_arm/','right_gripper/')):moving_separations.append(c['separation_m'])
        metric=dict(number=number,passed=bool(result['passed'] and hard_limits and velocity_fraction<=1 and acceleration_fraction<=1 and same_source and clock_ok),steps=len(rows),simulated_seconds=dt*len(rows),fixed_step_s=dt,endpoint_position_error_m=result['endpoint_position_error_m'],endpoint_orientation_error_deg=result['endpoint_orientation_error_deg'],maximum_tracking_error_m=result['maximum_tracking_error_m'],maximum_fixed_drift_m=result['maximum_fixed_drift_m'],maximum_sampled_velocity_fraction=velocity_fraction,maximum_sampled_acceleration_fraction=acceleration_fraction,actual_hard_joint_limits_passed=hard_limits,source_trajectory_unchanged=same_source,clock_verified=clock_ok,cloth_error_m=result['final_cloth_error_m'],unexpected_moving_contact_frames=len(result['unexpected_contacts']),actual_configuration_failures=len(result['actual_configuration_failures']),initial_scene_error_m=max(result['initial_human_error_m'],result['initial_cloth_error_m'],max(p['error_m'] for p in result['obstacle_errors'])),contact_pairs=[dict(pair=list(pair),count=count,maximum_penetration_m=depths[pair]) for pair,count in pairs.items()],result_file=str((folder/'result.json').relative_to(ROOT)))
        metrics.append(metric);series.append((number,clock-result['initial_clock'],np.array([r['tracking_error_m'] for r in rows])*1000,np.array([r['fixed_drift_m'] for r in rows])*1e6))
        metric.update(moving_contact_samples=len(moving_separations),minimum_moving_contact_separation_m=min(moving_separations,default=None),nonpositive_moving_contact_samples=sum(p<=0 for p in moving_separations))
        metric['passed']=metric['passed'] and not metric['nonpositive_moving_contact_samples']
    native=json.loads((args.registration/'native-validation/summary.json').read_text());source=json.loads((args.registration/'source-change-audit.json').read_text());routes=json.loads((args.registration/'route-recheck/summary.json').read_text());payload_hash=hashlib.sha256((args.registration/'payload.json').read_bytes()).hexdigest();geometry_ok=native['passed'] and native['payload_sha256']==payload_hash
    manifest_hash=hashlib.sha256((args.registration/'manifest.json').read_bytes()).hexdigest()
    route_geometry_ok=routes['offline_passed'] and all(p['geometry_manifest_sha256']==manifest_hash for p in routes['routes'])
    result=dict(passed=geometry_ok and source['passed'] and source['payload_sha256']==payload_hash and routes['offline_passed'] and all(p['passed'] for p in metrics),native_geometry_passed=geometry_ok,registered_parts=native['part_count'],rays=native['ray_count'],ray_misses=native['misses'],maximum_geometry_bound_m=native['maximum_combined_geometry_bound_m'],maximum_source_shape_change_bound_m=source['maximum_union_hausdorff_bound_m'],offline_recheck_samples=sum(p['sample_count'] for p in routes['routes']),replayed_numbers=[p['number'] for p in metrics],routes=metrics,reference_cloth_held=True,dynamic_grasp_validated=False,continuous_collision_certified=False)
    result['passed']=result['passed'] and route_geometry_ok;result['route_geometry_identity_verified']=route_geometry_ok
    (args.replay/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,8),sharex='col')
    for i,(number,t,tracking,fixed) in enumerate(series):
        axes[i,0].plot(t,tracking,lw=1);axes[i,1].plot(t,fixed,lw=1);axes[i,0].set_ylabel(f'No.{number}\ntracking [mm]');axes[i,1].set_ylabel('fixed links [µm]');axes[i,0].grid(alpha=.25);axes[i,1].grid(alpha=.25)
    axes[0,0].set_title('Measured TCP versus saved PCHIP reference');axes[0,1].set_title('Maximum displacement of stationary robot links');axes[-1,0].set_xlabel('Simulated time [s]');axes[-1,1].set_xlabel('Simulated time [s]');fig.suptitle('PhysX replay with held reference cloth; dynamic grasp not tested');fig.tight_layout();fig.savefig(args.replay/'tracking.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
