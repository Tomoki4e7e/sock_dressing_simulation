#!/usr/bin/env python3
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,sys,subprocess,signal,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.common_opening_approach import inventory,sha256
from replay_contact_pregrasp import verify_source
from search_centered_reaching import write,REGISTERED_MANIFEST


def run_one(job):
    row,source,out,port,dt,timeout,resume=job;path=source/'routes'/f'{row["id"]}.json';initial=source/'candidate-initial.yaml'
    verify_source(path,initial,ROOT/row['state']);dest=out/'drives'/row['id'];dest.mkdir(parents=True,exist_ok=True)
    binding=dict(route_sha256=sha256(path),initial_sha256=sha256(initial),state_sha256=row['state_sha256'],fixed_step_s=dt,
        geometry_manifest_sha256=sha256(REGISTERED_MANIFEST),replay_source_sha256=sha256(ROOT/'scripts/replay_contact_pregrasp.py'),
        capture_source_sha256=sha256(ROOT/'scripts/capture_fixed_torso_initial.py'),checker_source_sha256=sha256(ROOT/'sock_dressing_simulation/continuous_pregrasp.py'))
    binding['native_geometry_source_sha256']=sha256(ROOT/'scripts/native_pregrasp_geometry.py')
    status_path=dest/'status.json'
    if resume and status_path.exists():
        previous=json.loads(status_path.read_text())
        if previous.get('binding')==binding and previous.get('finished')and previous.get('returncode')==0:return previous
    env=dict(os.environ,SOCK_REGISTERED_PLAYER_LOG=str(dest/'player.log'),SOCK_REGISTERED_BATCHMODE='1',LIBGL_ALWAYS_SOFTWARE='1',__GLX_VENDOR_LIBRARY_NAME='mesa')
    env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(ROOT/'.deps/RCareWorld/pyrcareworld'),'/tmp/sock-grasp-deps','/opt/ros/noetic/lib/python3/dist-packages',env.get('PYTHONPATH','')])
    cmd=[sys.executable,str(ROOT/'scripts/capture_fixed_torso_initial.py'),'--initial-configuration',str(initial),'--state',str(ROOT/row['state']),
         '--contact-route',str(path),'--output',str(dest),'--port',str(port),'--fixed-step',str(dt)]
    began=time.monotonic();status=dict(id=row['id'],condition=row,binding=binding,passed=False,finished=False)
    log_path=dest/f'replay-port-{port}.log'
    status['log']=str(log_path.relative_to(out))
    with log_path.open('w')as log:
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:status['returncode']=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM);process.wait();status.update(returncode=-15,error='Native replay timeout')
    result_path=dest/'native-route-initial/replay-result.json'
    if result_path.exists()and status['returncode']==0:
        r=json.loads(result_path.read_text());status.update(passed=r['passed'],completed=r['completed'],steps=r['steps'],
            maximum_tracking_error_m=r['maximum_tracking_error_m'],cloth_maximum_error_m=r['cloth_maximum_error_m'],
            human_skin_maximum_error_m=r['human_skin_maximum_error_m'],unexpected_contact_steps=len(r['unexpected_contacts']),
            geometric_failure_steps=len(r['actual_configuration_failures']),maximum_velocity_fraction=r.get('maximum_velocity_fraction'),
            maximum_acceleration_fraction=r.get('maximum_acceleration_fraction'),final_contact_passed={s:c['relaxed_passed']for s,c in r['final_contact_checks'].items()})
    else:status['error']=status.get('error','Native run failed; see replay.log and native-route-initial/result.json')
    status.update(finished=True,elapsed_s=time.monotonic()-began);write(status_path,status)
    import fcntl
    from replay_diverse_left_reaching import compress_intermediates
    native=dest/'native-route-initial';native.mkdir(exist_ok=True)
    with(native/'drive-replay.lock').open('a')as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX);compress_intermediates(dest)
    print('DRIVES',row['id'],status['passed'],status.get('completed'),status['returncode'],round(status['elapsed_s'],1),flush=True);return status


def report(out,dt):
    rows=[json.loads(p.read_text())for p in sorted((out/'drives').glob('*/status.json'))]
    result=dict(expected=70,completed=len(rows),passed=sum(r['passed']for r in rows),motion_completed=sum(r.get('completed',False)for r in rows),runtime_errors=sum(r['returncode']!=0 for r in rows),
                all_70_passed=len(rows)==70 and all(r['passed']for r in rows),fixed_step_s=dt,cloth_frozen=True,cloth_deformation_enabled=False,joint_drives_only=True,final_closure_performed=False,rows=rows)
    write(out/'drives-summary.json',result);print('DRIVES SUMMARY',{k:v for k,v in result.items()if k!='rows'},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'artifacts/contact-pregrasp-start-search');p.add_argument('--output',type=Path,default=ROOT/'artifacts/contact-pregrasp-verification');p.add_argument('--ids',nargs='*');p.add_argument('--reverse',action='store_true');p.add_argument('--workers',type=int,default=4);p.add_argument('--port',type=int,default=22000);p.add_argument('--dt',type=float,default=.02);p.add_argument('--timeout',type=float,default=1800.);p.add_argument('--resume',action='store_true');p.add_argument('--report-only',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not a.report_only:
        rows=[r for r in inventory(ROOT)if not a.ids or r['id']in a.ids]
        if a.reverse:rows.reverse()
        with ThreadPoolExecutor(max_workers=a.workers)as pool:
            for f in as_completed([pool.submit(run_one,(r,a.source.resolve(),out,a.port+i*4,a.dt,a.timeout,a.resume))for i,r in enumerate(rows)]):f.result()
    report(out,a.dt)

if __name__=='__main__':main()
