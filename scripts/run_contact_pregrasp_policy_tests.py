#!/usr/bin/env python3
"""Run the saved test split with native closed-loop GPU control and verify videos."""
import argparse,csv,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run_case(row,run,out,index,checkpoint,checkpoint_sha256):
    dest=out/row['id'];dest.mkdir(parents=True,exist_ok=True)
    report=dest/'policy/report.json'
    if report.exists():
        previous=json.loads(report.read_text())
        if previous.get('checkpoint_sha256')!=checkpoint_sha256:
            raise RuntimeError('Existing trial uses another checkpoint; preserve it and choose a new --output')
        if previous.get('completed') and previous.get('video_validation'):return previous
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LP_NUM_THREADS='2',MPLCONFIGDIR='/tmp/pregrasp-live-mpl',PYTHONUNBUFFERED='1',SOCK_REGISTERED_BATCHMODE='1',SOCK_REGISTERED_PLAYER=str(ROOT/'Build/SockDressingPlayer/ExactHumanContactV2/Player.x86_64'),SOCK_REGISTERED_PLAYER_LOG=str(dest/'player.log'),LIBGL_ALWAYS_SOFTWARE='1',__GLX_VENDOR_LIBRARY_NAME='mesa',PYTHONPATH=str(ROOT)+':'+str(ROOT/'.deps/RCareWorld/pyrcareworld')+':/opt/ros/noetic/lib/python3/dist-packages')
    command=['/tmp/sock-diversity-venv/bin/python','-u',str(ROOT/'scripts/contact_pregrasp_policy_scene.py'),'--run',str(run),'--checkpoint',str(checkpoint),'--episode',row['path'],'--output',str(dest),'--port',str(36700+index*4)]
    write(dest/'launch.json',dict(command=command,started_at=time.time()))
    print('START',row['id'],flush=True)
    with (dest/'run.log').open('w') as log:
        try:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        except subprocess.TimeoutExpired:return dict(id=row['id'],completed=False,failure='Native policy timeout')
    data=json.loads(report.read_text()) if report.exists() else dict(id=row['id'],completed=False,failure='No policy report')
    if result.returncode:data.update(completed=False,returncode=result.returncode)
    if data.get('completed'):
        checks=[]
        for key,frames in [('overview_video',data['overview_frames']),('head_video',data['head_frames'])]:
            path=data[key];probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate','-of','json',path],text=True))['streams'][0]
            assert probe['codec_name']=='h264' and int(probe['nb_frames'])==frames
            subprocess.run(['ffmpeg','-v','error','-i',path,'-f','null','-'],check=True)
            checks.append(dict(path=path,decoded=True,frames=frames))
        data['video_validation']=checks;write(report,data)
    print('FINISH',row['id'],data.get('completed'),data.get('steps'),flush=True)
    return data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--ids',nargs='*');ap.add_argument('--checkpoint',type=Path);ap.add_argument('--output',type=Path);a=ap.parse_args();run=a.run.resolve();out=a.output.resolve() if a.output else run/'live-test-motion';out.mkdir(parents=True,exist_ok=True)
    checkpoint=a.checkpoint.resolve() if a.checkpoint else run/'best-test-inference/SARNN_best_evaluated.pth'
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows=[r for r in json.loads((run/'manifest.json').read_text())['rows'] if r['split']=='test' and (not a.ids or r['id'] in a.ids)];results=[];start=time.time()
    if not rows or (a.ids and set(a.ids)!={r['id'] for r in rows}):raise ValueError('Select existing test IDs')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(run_case,r,run,out,i,checkpoint,checkpoint_sha256):r for i,r in enumerate(rows)}
        for job in as_completed(jobs):
            try:data=job.result()
            except Exception as e:data=dict(id=jobs[job]['id'],completed=False,failure=repr(e))
            results.append(data)
            write(out/'progress.json',dict(completed=False,finished=len(results),expected=len(rows),successful=sum(r.get('completed',False) for r in results),cases=[dict(id=r['id'],completed=r.get('completed',False),failure=r.get('failure')) for r in results]))
    results.sort(key=lambda r:r['id']);compact=[]
    for r in results:
        row={k:r.get(k) for k in ('id','completed','steps','requested_steps','simulated_duration_s','overview_video','head_video','failure','native_penetration_event_count','cloth_maximum_drift_m')}
        for side in ('left','right'):
            g=r.get('geometry',{}).get(side,{})
            row[side]={k:g.get(k) for k in ('cloth_surface_area_between_fingers_mm2','material_15_30mm_band_area_between_fingers_mm2','opening_rim_length_between_fingers_mm','opening_rim_fraction_percent','material_depth_range_mm','cloth_present','opening_rim_present')}
            row[side]['contact_geometry_passed']=g.get('contact_geometry_checks',{}).get('passed')
        compact.append(row)
    fields=['id','completed','steps','simulated_duration_s','native_penetration_event_count']+[side+'_'+k for side in ('left','right') for k in ('cloth_surface_area_between_fingers_mm2','opening_rim_length_between_fingers_mm','cloth_present','contact_geometry_passed')]+['overview_video']
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in compact:w.writerow(dict(r,**{side+'_'+k:v for side in ('left','right') for k,v in r[side].items()}))
    epochs={r['device']['epoch'] for r in results if r.get('completed')}
    assert len(epochs)<=1
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checkpoint_sha256, 'Checkpoint changed during inference'
    summary=dict(completed=len(results)==len(rows) and all(r.get('completed') for r in results),expected=len(rows),finished=len(results),successful=sum(r.get('completed',False) for r in results),mode='Live closed-loop GPU SAMDAMSARNN + SAM2 + Depth Anything; native joint motors and frozen cloth; no recorded trajectory commands',checkpoint=str(checkpoint),checkpoint_epoch=next(iter(epochs),None),checkpoint_sha256=checkpoint_sha256,rows=compact,elapsed_s=time.time()-start,scope='Completion means execution and video validation. Cloth between measured finite jaws is a geometric measurement, not grasp retention or pulling success.')
    write(out/'report.json',summary);write(out/'progress.json',dict(completed=summary['completed'],finished=len(results),expected=len(rows),successful=summary['successful']))
    print('LIVE TEST SUITE COMPLETE',json.dumps(summary),flush=True)
    if not summary['completed']:raise SystemExit(1)

if __name__=='__main__':main()
