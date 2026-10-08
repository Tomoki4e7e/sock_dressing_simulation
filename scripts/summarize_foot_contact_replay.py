#!/usr/bin/env python3
"""Fail-closed physical contact comparison for the fixed W95 command replay."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'artifacts/phase4/foot-nonpenetration-fixed-motion'
BASELINE = ROOT / 'artifacts/phase4/opening-short-edge-widen/final100-w95/data_sock_sim_smoke/train/phase4_20261005T011108Z'


def export_diagnostics(metadata_path: Path, target: Path):
    metadata = json.loads(metadata_path.read_text())
    cloth = metadata.get('cloth_quality_by_frame', [])
    records = []
    for frame, qa in enumerate(metadata.get('dressing_quality_by_frame', [])):
        record = dict(frame=frame)
        record.update(qa.get('foot_collision_safety', {}))
        for key,value in list(record.items()):
            if isinstance(value,(dict,list)):
                record[key]=json.dumps(value,sort_keys=True)
        for key in ['geometric_maximum_cloth_foot_penetration_m',
                    'obi_maximum_cloth_foot_penetration_m', 'surface_containment_ratio',
                    'cuff_progress_toward_ankle_m', 'foot_contact_count']:
            record[key] = qa.get(key)
        geometric = qa.get('geometric_foot_penetration', {})
        # Preserve the historical radius-plus-margin metric alongside the
        # physical radius metric; predictive discovery is not cloth thickness.
        for key in ('particle_shell_metric', 'particle_shell_m',
                    'physical_particle_radius_m',
                    'maximum_physical_particle_penetration_m'):
            record['geometric_' + key] = geometric.get(key)
        sections = {section['name']:section for section in qa.get('sections', [])}
        for name in ('toes', 'forefoot', 'heel', 'ankle', 'calf'):
            record[name + '_containment_ratio'] = sections.get(name, {}).get('containment_ratio')
        record['foot_conformity'] = json.dumps(qa.get('foot_conformity', {}), sort_keys=True)
        anatomical=qa.get('anatomical_foot_conformity',{})
        record['anatomical_foot_conformity']=json.dumps(anatomical,sort_keys=True)
        record['anatomical_surface_containment_ratio']=anatomical.get('surface_containment_ratio')
        for section in anatomical.get('sections',[]):
            record['anatomical_'+section['name']+'_containment_ratio']=section.get('containment_ratio')
        material = cloth[frame] if frame < len(cloth) else {}
        record['cloth_quality_ok'] = material.get('ok')
        record['stretch_ok'] = material.get('stretch', {}).get('passes')
        record['circumferential_stretch_proxy'] = material.get('stretch', {}).get('circumferential_stretch_proxy')
        record['stretch_edge_classes'] = json.dumps(material.get('stretch', {}).get('edge_classes', {}), sort_keys=True)
        records.append(record)
    if records:
        with target.open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def evaluate(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text())
    rows = metadata.get('dressing_quality_by_frame',[])
    safety = [r.get('foot_collision_safety',{}) for r in rows]
    applied = metadata_path.parent/'applied_action.csv'
    commands = np.loadtxt(applied,delimiter=',',ndmin=2) if applied.exists() else np.empty((0,18))
    baseline = np.loadtxt(BASELINE/'applied_action.csv',delimiter=',',ndmin=2)
    command_error = float(np.max(np.abs(commands-baseline))) if commands.shape == baseline.shape else None
    task = metadata.get('task_success',{})
    diagnostics = bool(len(rows)==100 and all(s and 'remaining_swept_intersections' in s for s in safety))
    checks = {
        'completed_100_steps':metadata.get('frames')==100 and metadata.get('stop_reason')=='max_steps',
        'exact_commands': command_error is not None and command_error <= 1e-12,
        'diagnostics_available':diagnostics,
        'contact_safety_armed':diagnostics and all(s.get('contact_safety_armed',False) for s in safety),
        'no_remaining_swept_intersections':diagnostics and all(s['remaining_swept_intersections']==0 for s in safety),
        'no_remaining_surface_intersections':diagnostics and all(s['remaining_surface_intersections']==0 for s in safety),
        'post_correction_penetration_within_2mm':diagnostics and all(r.get('geometric_maximum_cloth_foot_penetration_m',float('inf'))<=.002 for r in rows),
        'continuous_grasp':bool(task.get('continuous_grasp_ok')),
        'foot_skin_covered':diagnostics and all(s.get('foot_skin_vertex_count',0)>0 and s.get('uncovered_foot_skin_vertices',1)==0 for s in safety),
    }
    cloth = metadata.get('cloth_quality_by_frame',[])
    result = {
        'metadata':str(metadata_path), 'checks':checks,'passed':all(checks.values()),
        'maximum_command_error':command_error,
        'surface_containment':float(task.get('final_surface_containment_ratio') or 0),
        'cloth_quality_passing_frames':sum(bool(r.get('ok')) for r in cloth),
        'stretch_violation_frames':sum(not r.get('stretch',{}).get('passes',False) for r in cloth),
        'maximum_structural_excess_length_m':max((edge.get('maximum_excess_length_m',0)
            for r in cloth for edge in r.get('stretch',{}).get('edge_classes',{}).values()),default=0),
        'maximum_post_correction_penetration_m':max((r.get('geometric_maximum_cloth_foot_penetration_m',0) for r in rows),default=None),
        'maximum_original_obi_penetration_m':task.get('maximum_obi_cloth_foot_penetration_m'),
        'total_sweep_corrections':sum(s.get('swept_contact_corrections',0) for s in safety),
        'total_surface_corrections':sum(s.get('surface_contact_corrections',0) for s in safety),
        'maximum_contact_correction_m':max((s.get('maximum_correction_m',0) for s in safety),default=None),
        'task_success':bool(task.get('success')), 'failed_task_gates':task.get('failed_gates',[]),
    }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--publish-best',action='store_true')
    args=parser.parse_args()
    results=[]
    for substeps,iterations in [(8,20),(8,40),(16,20),(16,40)]:
        directory=args.output_root/f's{substeps}_i{iterations}'
        files=list(directory.glob('data_sock_sim_smoke/train/phase4_*/metadata.json'))
        if not files:
            results.append({'condition':directory.name,'passed':False,'missing':True});continue
        path=max(files,key=lambda p:p.stat().st_mtime)
        export_diagnostics(path,directory/'frame_diagnostics.csv')
        row=evaluate(path);row.update(condition=directory.name,substeps=substeps,iterations=iterations)
        results.append(row)
    passing=[r for r in results if r['passed']]
    passing.sort(key=lambda r:(-r['surface_containment'],r['stretch_violation_frames'],
        r['maximum_structural_excess_length_m'],
        r['maximum_contact_correction_m'],r['substeps']*r['iterations']))
    summary={'baseline':str(BASELINE),'conditions':results,'best':passing[0]['condition'] if passing else None,
        'visual_review_required':True}
    args.output_root.mkdir(parents=True,exist_ok=True)
    (args.output_root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.publish_best:
        if not passing: raise RuntimeError('No passing replay; do not label an unsafe recording best')
        row=passing[0];episode=Path(row['metadata']).parent;condition=episode.parents[2]
        best=args.output_root/'best';best.mkdir(exist_ok=True)
        for name in ['demo.mp4','applied_action.csv','metadata.json','inference_camera.mp4']:
            shutil.copy2(episode/name,best/name)
        for name in ['config.yaml','side_demo.mp4','frame_diagnostics.csv']:
            shutil.copy2(condition/name,best/name)
        (best/'selection.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
