#!/usr/bin/env python3
"""Evaluate the validation-selected pregrasp checkpoint on held-out cached episodes."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import train_contact_pregrasp as training


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
    run=a.run_dir.resolve();out=run/'best-test-inference';out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Evaluation report already exists; preserve the previous results')
    torch.set_num_threads(2);training.cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; no CPU fallback')
    stop=json.loads((run/'training-stop.json').read_text())
    assert stop['process_stopped']
    started=time.time();original=run/'model/SARNN_best.pth';checkpoint=out/'SARNN_best_evaluated.pth'
    shutil.copy2(original,checkpoint)
    assert digest(checkpoint)==stop['best_checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=json.loads((run/'configuration.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    rows=manifest['rows'];test=[r for r in rows if r['split']=='test'];assert len(test)==12
    assert len({(r['pitch_deg'],r['roll_deg']) for r in test}&{(r['pitch_deg'],r['roll_deg']) for r in rows if r['split']!='test'})==0
    old_normalization=(run/'normalization.json').read_bytes()
    datasets,lower,upper=training.load_arrays(rows,run)
    assert (run/'normalization.json').read_bytes()==old_normalization, 'Normalization changed unexpectedly'
    m=config['model'];model=training.SAMDAMSARNN(rec_dim=m['rec_dim'],k_dim=m['k_dim'],joint_dim=m['joint_dim'],temperature=m['temperature'],heatmap_size=m['heatmap_size'],im_size=m['im_size']).cuda()
    model.load_state_dict(saved['model_state_dict'],strict=True);model.eval()
    assert next(model.parameters()).is_cuda
    device=dict(device='cuda',gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,model_device=str(next(model.parameters()).device))
    print(json.dumps(dict(checkpoint_epoch=saved['epoch'],test_episodes=len(test),**device)),flush=True)
    write(run/'progress.json',dict(completed=False,phase='best_test_inference',epoch=stop['checkpoint_epoch'],training_early_stopped=True,best_epoch=saved['epoch'],**device))
    scheduler=training.LossScheduler(1000,'s');scheduler.counter=1000
    with torch.inference_mode():
        test_loss=training.epoch(model,datasets['test'],None,None,scheduler,torch.device('cuda'),config['hyperparameters'])
    print('TEST LOSS',json.dumps(test_loss),flush=True)
    training.infer(model,datasets['test'],lower,upper,out,saved['epoch'])
    prediction_root=out/f'inference-epoch-{saved["epoch"]}'
    base=json.loads((prediction_root/'report.json').read_text());assert len(base['rows'])==12
    arm=list(range(7))+list(range(9,16));torque=[18+i for i in arm];fingers=[7,16]
    squared=np.zeros(36,dtype='float64');count=0;norm_sse=0.;image_frames=0;results=[]
    for r in test:
        ep=prediction_root/r['id'];prediction=np.loadtxt(ep/'predicted_state.csv',delimiter=',',ndmin=2);truth=np.loadtxt(ep/'reference_next_state.csv',delimiter=',',ndmin=2)
        expected=len(range(0,r['frames'],config['hyperparameters']['skip_num']))-1
        assert prediction.shape==truth.shape==(expected,36)
        assert np.isfinite(prediction).all() and np.isfinite(truth).all()
        err=prediction-truth;squared+=np.square(err).sum(0);count+=len(err)
        row=dict(episode=r['id'],stage=r['stage'],frames=len(err),left_arm_rmse_rad=float(np.sqrt(np.mean(err[:,:7]**2))),right_arm_rmse_rad=float(np.sqrt(np.mean(err[:,9:16]**2))),all_arms_rmse_rad=float(np.sqrt(np.mean(err[:,arm]**2))),arm_effort_rmse_Nm=float(np.sqrt(np.mean(err[:,torque]**2))),finger_rmse_m=float(np.sqrt(np.mean(err[:,fingers]**2))))
        video=ep/'offline-prediction.mp4'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
        assert probe['codec_name']=='h264' and int(probe['nb_frames'])==expected
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
        row.update(video=str(video),video_decoded=True,video_probe=probe)
        write(ep/'evaluation.json',row);results.append(row)
        print('TEST EPISODE',json.dumps(row),flush=True)
    rmse=np.sqrt(squared/count)
    aggregate=dict(transitions=count,left_arm_rmse_rad=float(np.sqrt(squared[:7].sum()/(count*7))),right_arm_rmse_rad=float(np.sqrt(squared[9:16].sum()/(count*7))),all_arms_rmse_rad=float(np.sqrt(squared[arm].sum()/(count*14))),arm_effort_rmse_Nm=float(np.sqrt(squared[torque].sum()/(count*14))),finger_rmse_m=float(np.sqrt(squared[fingers].sum()/(count*2))),rmse_per_state_column=rmse.tolist(),weighting='All valid test transitions and selected columns pooled; mimic excluded from arm/finger summaries')
    fields=['episode','stage','frames','left_arm_rmse_rad','right_arm_rmse_rad','all_arms_rmse_rad','arm_effort_rmse_Nm','finger_rmse_m','video']
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    report=dict(completed=True,scope='Held-out observed-input one-step recurrent prediction; observed RGB, measured joints/effort and cached SAM2/depth supplied at each time step. Not autonomous simulator control.',selection='Minimum validation loss on 9 validation states; test data were not used for checkpoint selection',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),checkpoint_epoch=saved['epoch'],checkpoint_validation_loss=saved['validation_loss'],training_stopped_epoch=stop['checkpoint_epoch'],test_episodes=12,test_loss=test_loss,test_loss_precision='CUDA autocast, as in validation',prediction_metrics_precision='CUDA float32 forward',aggregate=aggregate,rows=results,elapsed_s=time.time()-started,manifest_sha256=digest(run/'manifest.json'),normalization_sha256=digest(run/'normalization.json'),**device)
    write(out/'report.json',report)
    write(run/'progress.json',dict(completed=True,phase='early_stopped_and_best_test_inference_complete',training_early_stopped=True,epoch=stop['checkpoint_epoch'],original_target_epochs=10000,best_epoch=saved['epoch'],best_validation_loss=saved['validation_loss'],test_loss=test_loss['loss'],test_episodes=12,test_report=str(out/'report.json'),**device))
    print('BEST TEST INFERENCE COMPLETE',json.dumps(aggregate),flush=True)


if __name__=='__main__':main()
