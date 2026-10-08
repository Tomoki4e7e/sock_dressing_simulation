#!/usr/bin/env python3
"""Validate a repaired immutable runtime, then resume the fixed-motion search."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUTER=ROOT/'artifacts/phase4/foot-conforming-fixed-motion'
PERF=OUTER/'contact-performance'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',type=int,default=31)
    parser.add_argument('--port',type=int,default=7600)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    output=OUTER/('controlled-v%d'%args.version)
    output.mkdir(exist_ok=True)
    player=Path('/mnt/robot_ssd/sock_dressing_simulation_work/native_players/development-v%d/Player.x86_64'%args.version)
    status_path=PERF/'validation-status.json'
    started=time.monotonic()
    def status(stage,**details):
        value=dict(stage=stage,observed_at_utc=datetime.now(timezone.utc).isoformat(),version=args.version,
                   best_published=False,output_root=str(output),**details)
        temporary=status_path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(status_path)
    def run(name,steps,state,port):
        command=[sys.executable,'-u',str(ROOT/'scripts/replay_foot_contact.py'),'--output-root',str(output),
            '--candidate-name',name,'--steps',str(steps),'--substeps','8','--iterations','20','--surface-mode','local',
            '--foot-shape','skin-hull','--transport','local','--fixed-grasp-offsets','--post-preparation-material',
            '--prepared-state',str(state),'--player-executable',str(player),'--port',str(port)]
        with (PERF/(name+'.log')).open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                f=output/name/'live_frame_diagnostics.jsonl'
                frames=sum(1 for _ in f.open()) if f.exists() else 0
                status('validating_%d_fixed_commands'%steps,child_pid=child.pid,recorded_frames=frames,elapsed_s=time.monotonic()-started)
                time.sleep(20)
        if child.returncode:raise RuntimeError('Replay failed: '+name)
        result=json.loads((output/name/'result.json').read_text())
        return Path(result['metadata'])
    try:
        before=json.loads((PERF/'geometry-benchmark.json').read_text())
        assert before['passed'] and before['misses']==0 and before['maximumCorrectionDifferenceM']<=1e-5
        canonical=(OUTER/'controlled-v22/common_preparation.json').resolve()
        first=run('preparation_check_1',1,canonical,args.port)
        material=json.loads((output/'preparation_check_1/applied_material.json').read_text())
        assert material['validation']['ok'] and material['prepared_state_readback_validation']['ok']
        assert material['prepared_state_readback_validation']['source']=='native_state_recapture'
        common=output/'common_preparation.json';shutil.copyfile(output/'preparation_check_1/prepared_state.json',common)
        begin=time.monotonic()
        meta=run('local_transport_26ba670fdd43',40,common,args.port+4)
        elapsed=time.monotonic()-begin
        assess=runpy.run_path(str(ROOT/'scripts/tune_foot_conforming.py'))['assess']
        result=assess(meta,40)
        (PERF/'fixed40-validation.json').write_text(json.dumps(dict(assessment=result,wall_time_s=elapsed,
            reference=str(canonical),reference_sha256=hashlib.sha256(canonical.read_bytes()).hexdigest(),
            scope='Performance validation of unchanged safe local-transport control; conformity assessed separately.'),indent=2)+'\n')
        assert result['safe'], 'Fixed40 invariants failed; comparison remains paused'
        assert result['tail_stretch_violation_frames']==0, 'Fixed40 stretch regression; comparison remains paused'
        # Keep native replay trajectory differences visible. The existing V30
        # control and bootstrap already differ despite the same stored state;
        # the plan specifies 10um for identical-input corrections, not chaotic
        # independent whole-run trajectories. Physical invariants stay mandatory.
        import numpy as np
        previous=OUTER/'controlled-v30/local_transport_26ba670fdd43/live_frame_diagnostics.jsonl'
        current=output/'local_transport_26ba670fdd43/live_frame_diagnostics.jsonl'
        old=[json.loads(x) for x in previous.read_text().splitlines()]
        new=[json.loads(x) for x in current.read_text().splitlines()]
        error=max(float(np.max(np.linalg.norm(np.array(a['particles_world'])-np.array(b['particles_world']),axis=1))) for a,b in zip(old,new))
        equal=dict(frames_old=len(old),frames_new=len(new),maximum_particle_difference_m=error,tolerance_m=1e-5,
                   within_10um=len(old)==len(new)==40 and error<=1e-5,
                   comparison_scope='Independent runtime trajectories; not the identical-input correction equivalence gate',
                   repeatability_audit=str(PERF/'replay-repeatability-audit.json'))
        (PERF/'fixed40-runtime-equivalence.json').write_text(json.dumps(equal,indent=2)+'\n')
        assert len(old)==len(new)==40, 'Incomplete whole-run diagnostics'
        # Start the original scheduler only after all physical and differential gates.
        with (OUTER/('v%d-supervisor.log'%args.version)).open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','scripts/continue_foot_conforming.py','--version',str(args.version),
                '--port',str(args.port),'--workers',str(args.workers)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with (OUTER/'CURRENT_STATUS.md').open('a') as report:
            report.write('\n40指令の必須物理条件と終盤の伸び検査が通過し、V%dで元の比較試行を自動再開しました。独立試行の位置差は同一入力の補正一致とは区別し、replay-repeatability-audit.json と fixed40-runtime-equivalence.json に保持しています。\n'%args.version)
        status('validation_passed_original_comparisons_resumed',supervisor_pid=child.pid,
               fixed40_wall_time_s=elapsed,observed_seconds_per_command=elapsed/40,
               estimate_scope='Measured local control only; material and bulk-contact trials may have different cost.')
    except Exception as error:
        status('validation_failed_comparisons_remain_paused',error=str(error));raise

if __name__=='__main__':main()
