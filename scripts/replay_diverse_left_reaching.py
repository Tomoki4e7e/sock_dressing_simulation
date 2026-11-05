#!/usr/bin/env python3
"""Replay validated shared-start routes and export measured 18-column angles."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import shutil
import tarfile
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,URDF
from report_photo_grasp_adjustment import actual_recheck
from sock_dressing_simulation.grasp_geometry import RightHand
from sock_dressing_simulation.reaching_dataset import file_sha256,verify_dataset_inputs


def compress_intermediates(destination):
    """Retain large native snapshots and completed engine logs losslessly."""
    native=destination/'native-route-initial'
    for path in [native/'before-scene.json',native/'native-scene.json',destination/'player.log']:
        if not path.exists():continue
        target=path.with_suffix(path.suffix+'.gz')
        with path.open('rb') as source,gzip.open(target,'wb',compresslevel=1) as compressed:
            shutil.copyfileobj(source,compressed)
        path.unlink()
    assets=native/'assets'
    if assets.exists():
        with tarfile.open(native/'assets.tar.gz','w:gz',compresslevel=1) as archive:
            archive.add(assets,arcname='assets')
        shutil.rmtree(assets)


def export_angles(plan,trace,out):
    hand=RightHand(URDF)
    names=[f'left_arm/joint_{i}' for i in range(1,8)]+['left_gripper/finger_joint','left_gripper/mimic_joint']+[f'right_arm/joint_{i}' for i in range(1,8)]+['right_gripper/finger_joint','right_gripper/mimic_joint']
    with (out/'angle.csv').open('w') as f,(out/'frames.csv').open('w') as frames:
        writer=csv.writer(f);meta=csv.writer(frames);meta.writerow(['frame','time_s','stage','stage_time_s'])
        for row in trace:
            left=hand.joint_for_gap(row['actual_left_gap_m']);right=hand.joint_for_gap(row['actual_right_gap_m'])
            writer.writerow(row['actual_left_q_rad']+[left,.5*left]+row['actual_right_q_rad']+[right,.5*right])
            meta.writerow([row['step'],(row['step']+1)*.02,row['stage'],row['stage_time_s']])
    state=json.loads((ROOT/plan['geometry_record']['state_file']).read_text())
    placement=state.get('placement')
    ignored=list(plan.get('ignored_workbook_conditions',[]))
    if plan['applied_opening_stage']!=plan['workbook_pattern']['stage'] and 'stage' not in ignored:ignored.append('stage')
    write(out/'demonstration.json',dict(joint_names=names,angle_units='radians; gripper prismatic columns in metres',samples=len(trace),fixed_step_s=.02,
        shared_initial_configuration=plan['shared_initial_configuration'],dataset_inputs=plan['dataset_inputs'],workbook_pattern=plan['workbook_pattern'],applied_opening_stage=plan['applied_opening_stage'],
        cloth_geometry='normal anatomical preinsertion' if placement else 'saved held-A with 5 mm cuff bulge',cloth_placement=placement,
        ignored_workbook_conditions=ignored,workbook_row_fully_reproduced=False,
        angle_source='PhysX joint feedback; each mimic is 0.5 times its measured source joint',torque_recorded=False,images_synchronized=False,learning_ready=False,
        scope='Reaching and geometric jaw closure around frozen reference cloth; deformation, friction retention and pull-up are not validated'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/diverse-left-reaching/final');ap.add_argument('--port',type=int,default=16800);ap.add_argument('--limit',type=int);ap.add_argument('--match',default='');a=ap.parse_args();out=a.output.resolve()
    paths=sorted((out/'routes').glob('*-trajectory.json'))
    paths=[p for p in paths if a.match in p.stem]
    if a.limit:paths=paths[:a.limit]
    for i,path in enumerate(paths):
        plan=json.loads(path.read_text());verify_dataset_inputs(plan,ROOT)
        validation=json.loads(path.with_name(path.name.replace('trajectory','validation')).read_text())
        if not validation['offline_passed'] or validation['trajectory_sha256']!=file_sha256(path):continue
        destination=out/'replays'/path.stem;destination.mkdir(parents=True,exist_ok=True)
        result_path=destination/'native-route-initial/replay-result.json'
        if result_path.exists():
            previous=json.loads(result_path.read_text())
            closed_path=destination/'actual-closed-grasp-recheck.json'
            closed=json.loads(closed_path.read_text()) if closed_path.exists() else {}
            if previous.get('source_sha256')==file_sha256(path) and previous.get('passed') and closed.get('passed') and (destination/'angle.csv').exists():
                trace=[json.loads(line) for line in (destination/'native-route-initial/motion-trace.jsonl').read_text().splitlines()]
                export_angles(plan,trace,destination)
                compress_intermediates(destination)
                continue
        env=dict(os.environ)
        env.setdefault('__GLX_VENDOR_LIBRARY_NAME','mesa')
        env.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
        env['PYTHONUNBUFFERED']='1'
        env['SOCK_REGISTERED_PLAYER_LOG']=str(destination/'player.log')
        env.setdefault('SOCK_REGISTERED_BATCHMODE','1')
        env['PYTHONPATH']=str(ROOT/'.deps/RCareWorld/pyrcareworld')+os.pathsep+env.get('PYTHONPATH','')
        cmd=[sys.executable,str(ROOT/'scripts/capture_fixed_torso_initial.py'),'--trajectory',str(path),'--output',str(destination),'--port',str(a.port+i),'--replay']
        with (destination/'replay.log').open('w') as log:
            try:
                process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            except subprocess.TimeoutExpired:
                write(destination/'batch-status.json',dict(passed=False,reason='native replay timed out'))
                print('REPLAY TIMEOUT',path.stem,flush=True);continue
        compress_intermediates(destination)
        if process.returncode:
            write(destination/'batch-status.json',dict(passed=False,returncode=process.returncode,log=str((destination/'replay.log').relative_to(ROOT))))
            print('REPLAY ERROR',path.stem,process.returncode,flush=True);continue
        motion=json.loads(result_path.read_text())
        if motion['passed']:
            closed,trace=actual_recheck(plan,motion,destination)
            if closed['passed']:export_angles(plan,trace,destination)
            print('REPLAY',path.stem,'drive',motion['passed'],'actual capture',closed['passed'],flush=True)
        else:print('REPLAY FAILED',path.stem,flush=True)


if __name__=='__main__':main()
