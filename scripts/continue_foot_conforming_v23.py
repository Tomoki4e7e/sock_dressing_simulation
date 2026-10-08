#!/usr/bin/env python3
"""Bootstrap the repaired runtime from the validated canonical state, then compare."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUTER=ROOT/'artifacts/phase4/foot-conforming-fixed-motion'
OUTPUT=OUTER/'controlled-v23'
PLAYER=Path('/mnt/robot_ssd/sock_dressing_simulation_work/native_players/development-v23/Player.x86_64')
REFERENCE=(OUTER/'controlled-v22/common_preparation.json').resolve()
STATUS=OUTER/'v23-status.json'

def status(stage, **values):
    record={'stage':stage,'observed_at_utc':datetime.now(timezone.utc).isoformat(),
            'best_published':False,'output_root':str(OUTPUT),'player':str(PLAYER),**values}
    temporary=STATUS.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(STATUS)

def run(command, log, stage):
    with log.open('a') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        while child.poll() is None:
            status(stage,child_pid=child.pid,log=str(log));time.sleep(20)
    if child.returncode:raise RuntimeError(f'{stage} exited {child.returncode}; see {log}')

def main():
    OUTPUT.mkdir(exist_ok=True)
    kernel=PLAYER.parent/'Player_Data/Managed'
    expected={name:hashlib.sha256((kernel/name).read_bytes()).hexdigest() for name in ('Obi.dll','RCareWorld.dll')}
    run([sys.executable,'-u',str(ROOT/'scripts/replay_foot_contact.py'),
         '--output-root',str(OUTPUT),'--candidate-name','preparation_check_1',
         '--steps','1','--substeps','8','--iterations','20','--surface-mode','local',
         '--foot-shape','skin-hull','--transport','local','--fixed-grasp-offsets',
         '--post-preparation-material','--prepared-state',str(REFERENCE),
         '--player-executable',str(PLAYER),'--port','6500'],
        OUTPUT/'preparation-check.log','validating_canonical_state_in_repaired_runtime')
    directory=OUTPUT/'preparation_check_1'
    result=json.loads((directory/'result.json').read_text())
    metadata=json.loads(Path(result['metadata']).read_text())
    material=json.loads((directory/'applied_material.json').read_text())
    assert metadata['frames']==1 and metadata['stop_reason']=='max_steps'
    assert metadata['pre_inference_drape']['passed']
    assert material['prepared_state_readback_validation']['ok']
    assert material['prepared_state_readback_validation']['source']=='native_state_recapture'
    assert material['validation']['ok']
    assert expected=={name:hashlib.sha256((kernel/name).read_bytes()).hexdigest() for name in expected}
    snapshot=directory/'prepared_state.json'
    shutil.copyfile(snapshot,OUTPUT/'common_preparation.json')
    (OUTPUT/'common_preparation_kernel.json').write_text(json.dumps(expected,indent=2)+'\n')
    (OUTPUT/'common_preparation_provenance.json').write_text(json.dumps({
        'restored_reference':str(REFERENCE),'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        'validation_record':str(directory/'applied_material.json'),
        'captured_in_runtime':expected,'recapture_validation':material['prepared_state_readback_validation'],
        'diagnostic_steps':1,'included_in_material_selection':False},indent=2)+'\n')
    run([sys.executable,'-u',str(ROOT/'scripts/tune_foot_conforming.py'),
         '--phase','all','--workers','4','--foot-shape','skin-hull',
         '--output-root',str(OUTPUT),'--player-executable',str(PLAYER),'--port','6520'],
        OUTPUT/'scheduler.log','recording_controls_materials_and_final_grid')
    status('final_grid_complete_manual_visual_review_required')

if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed_requires_source_or_candidate_repair',error=str(error))
        raise
