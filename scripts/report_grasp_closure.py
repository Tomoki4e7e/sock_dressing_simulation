#!/usr/bin/env python3
"""Aggregate raw closure trials without treating missing data as zero slip."""
import csv,json,sys,argparse
import yaml
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.grasp_geometry import RightHand
from sock_dressing_simulation.grasp_cloth_geometry import ray_interval
from measure_grasp_closure import analyze
from sock_dressing_simulation.closure_pad_metrics import sections,local_pad_proximity

DEFAULT_CRITERIA=yaml.safe_load((ROOT/'config/grasp_closure_decision.yaml').read_text())

def retention_status(abort,stop_reason,pad_fraction,contact_fraction,target_drift,tangent_drift,end_speed,outside_events=0,local_contact_fraction=None,criteria=None):
    cfg=DEFAULT_CRITERIA if criteria is None else criteria
    if abort:return 'safety_abort'
    if outside_events:return 'outside_band_contact'
    values=[pad_fraction,contact_fraction,target_drift,tangent_drift,end_speed,local_contact_fraction]
    if any(v is not None and not np.isfinite(v) for v in values):return 'invalid_numeric_measurement'
    if stop_reason!='bilateral_contact_persisted':return 'bilateral_stop_not_reached'
    if pad_fraction is None or pad_fraction<cfg['target_in_pad_fraction']:return 'target_not_retained_in_pad_window'
    if contact_fraction is None or contact_fraction<cfg['bilateral_fraction']:return 'bilateral_contact_not_persistent'
    if local_contact_fraction is not None and local_contact_fraction<cfg['local_pad_geometry_fraction']:return 'local_pad_contact_not_persistent'
    if target_drift is None or tangent_drift is None:return 'material_drift_not_evaluable'
    if target_drift>cfg['target_drift_m'] or tangent_drift>cfg['contact_support_tangent_drift_m']:return 'material_drift_over_threshold'
    if end_speed is None or end_speed>cfg['end_band_speed_m_s']:return 'band_not_settled'
    return 'provisional_retention_proxy'

def material_motion(holds,rotation,pull_axis):
    if not holds:return dict(target_relative_motion_all_hold_m=None,target_tangential_motion_all_hold_m=None,target_tangential_speed_peak_m_s=None,target_tangential_speed_p95_m_s=None,pull_follow_ratio=None)
    local=np.asarray([s['target_relative_tcp_world'] for s in holds])@rotation
    difference=local-local[0];times=np.asarray([s['time_s'] for s in holds]);velocity=np.linalg.norm(np.diff(local[:,[0,2]],axis=0)/np.diff(times)[:,None],axis=1) if len(holds)>1 else np.array([])
    before=[s for s in holds if s['phase']=='hold'];after=[s for s in holds if s['phase']=='post_hold'];follow=None
    if before and after:
        command=after[-1]['pull_m']-before[-1]['pull_m']
        if command>1e-6:
            pull_axis=np.asarray(pull_axis);pull_axis=pull_axis/np.linalg.norm(pull_axis)
            follow=float((np.asarray(after[-1]['target_material_world'])-np.asarray(before[-1]['target_material_world']))@pull_axis/command)
    return dict(target_relative_motion_all_hold_m=float(np.linalg.norm(difference,axis=1).max()),target_tangential_motion_all_hold_m=float(np.linalg.norm(difference[:,[0,2]],axis=1).max()),
       target_tangential_speed_peak_m_s=float(velocity.max()) if len(velocity) else None,target_tangential_speed_p95_m_s=float(np.quantile(velocity,.95)) if len(velocity) else None,pull_follow_ratio=follow)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--criteria',type=Path,default=ROOT/'config/grasp_closure_decision.yaml');args=parser.parse_args()
 cfg=yaml.safe_load(args.criteria.read_text())
 if set(cfg)!=set(DEFAULT_CRITERIA) or any(not np.isfinite(v) or v<0 for v in cfg.values()):raise ValueError('Invalid decision criteria')
 if any(cfg[k]>1 for k in ('bilateral_fraction','target_in_pad_fraction','local_pad_geometry_fraction')):raise ValueError('Fractions must be at most one')
 out=ROOT/'artifacts/grasp-closure-measurements';out.mkdir(exist_ok=True)
 source=json.loads((ROOT/'artifacts/grasp-cloth-refilter/summary.json').read_text());hand=RightHand(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf');cache={};rows=[];poses=[]
 for pose in source['trials']:
  n=pose['number'];expected=pose['counts'].get('static_accepted',0);per=[]
  for i in range(expected):
   path=out/f'pose-{n:02d}/candidate-{i:03d}.json';base=dict(number=n,pitch_deg=pose['pitch_deg'],roll_deg=pose['roll_deg'],candidate_index=i)
   if not path.exists():
    error=out/f'pose-{n:02d}/error-{i:03d}.txt';rows.append(dict(base,status='measurement_error' if error.exists() else 'pending'));continue
   r=json.loads(path.read_text());
   if r.get('protocol_version')!=5:raise RuntimeError(f'Unexpected measurement version: {path}')
   data=r['measurement'];c=r['candidate'];R=np.asarray(c['rotation_matrix']);a=analyze(data,R)
   samples=[s for s in data['samples'] if 'pieces' in s];holds=[s for s in samples if s['phase'] in ('hold','pull','post_hold')]
   within=[];local_near=[];distances=[]
   initial=np.asarray(r['initial_particles_world']);faces=np.asarray(r['triangles']).reshape(-1,3);band=np.asarray(data['band_particle_indices']);band_mask=np.zeros(len(initial),dtype=bool);band_mask[band]=True;active_faces=faces[np.all(band_mask[faces],axis=1)]
   section_cache={}
   for s in holds:
    gap=round(s['gap_m'],8)
    if gap not in cache:
     fingers=[v for name,v in hand.vertices(gap).items() if name.endswith('/left_finger_link') or name.endswith('/right_finger_link')]
     fingers=sorted(fingers,key=lambda v:v[:,1].mean());cache[gap]=[ConvexHull(v).equations for v in fingers]
    target=np.asarray(s['target_relative_tcp_world'])@R;x,y,z=target;intervals=[ray_interval(h,x,z) for h in cache[gap]]
    ok=abs(x)<=.004 and abs(z)<=.004 and all(v is not None for v in intervals)
    if ok:ok=intervals[0][1]-.0005<=y<=intervals[1][0]+.0005
    within.append(bool(ok))
    if gap not in section_cache:section_cache[gap]=sections(hand,gap)
    current=initial.copy();current[band]=np.asarray(s['band_positions_world']);local=(current-np.asarray(s['tcp_world']))@R
    envelope=data['particle_radius_m']+data['obi_thickness_m']
    proximity=local_pad_proximity(local,active_faces,section_cache[gap],envelope)
    local_near.append(proximity['negative_near'] and proximity['positive_near']);distances.append(proximity)
   fraction=float(np.mean(within)) if within else None
   local_fraction=float(np.mean(local_near)) if local_near else None
   last=[s for s in holds if s['time_s']>=holds[-1]['time_s']-.5] if holds else []
   end_speed=max((s['maximum_band_speed_m_s'] for s in last),default=None)
   drift=a['target_material_relative_drift_m'];tan=a['tangential_contact_material_drift']
   tangential=[v['maximum_tangential_drift_m'] for v in tan.values() if v['maximum_tangential_drift_m'] is not None]
   max_tangent=max(tangential) if len(tangential)==2 else None
   contact_fraction=a['hold_bilateral_fraction']
   status=retention_status(data['abort_reason'],data['closing_stop_reason'],fraction,contact_fraction,drift,max_tangent,end_speed,sum(s['outside_band_events'] for s in samples),local_fraction,cfg)
   record=dict(base,status=status,opening_gap_m=c['gap_m'],sample_index=c['sample_index'],yaw_deg=c['yaw_deg'],tilt_deg=c['tilt_deg'],closing_sign=c['closing_sign'],
    stop_reason=data['closing_stop_reason'],stop_gap_m=data['closing_stop_gap_m'],first_bilateral_gap_m=a['first_bilateral_gap_m'],hold_bilateral_fraction=contact_fraction,
    local_bilateral_pad_near_fraction=local_fraction,target_in_pad_fraction=fraction,target_relative_drift_m=drift,contact_support_tangent_drift_max_m=max_tangent,end_band_speed_max_m_s=end_speed,
    all_steps_band_speed_max_m_s=a['maximum_band_speed_m_s'],edge_stretch_max=r['analysis'].get('final_edge_stretch_ratio_max'),
    outside_band_events=sum(s['outside_band_events'] for s in samples),raw_report=str(path.relative_to(ROOT)))
   record.update(material_motion(holds,R,[r['protocol']['pullAxis'][k] for k in 'xyz']))
   for name in ('left','right'):
    pieces=[next(p for p in s['pieces'] if p['link'].endswith('/'+name+'_finger_link')) for s in samples]
    record[name+'_solver_normal_impulse_sum']=sum(p['solver_normal_impulse_sum'] for p in pieces)
    record[name+'_solver_friction_impulse_sum']=sum(p['solver_friction_impulse_sum'] for p in pieces)
    record[name+'_active_steps']=sum(p.get('active_physics_steps',0) for p in pieces)
    first=next((s for s,piece in zip(samples,pieces) if piece['active_events']>0),None)
    record[name+'_first_active_gap_m_binned']=first['gap_m'] if first else None
   for label,suffix in [('palm','gripper_base'),('left_base','left_finger_base_link'),('right_base','right_finger_base_link')]:
    pieces=[next(p for p in sample['pieces'] if p['link'].endswith('/'+suffix)) for sample in samples]
    record[label+'_solver_normal_impulse_sum']=sum(p['solver_normal_impulse_sum'] for p in pieces)
    record[label+'_active_steps']=sum(p.get('active_physics_steps',0) for p in pieces)
   gates=dict(no_safety_abort=data['abort_reason'] is None,stopped_bilateral=data['closing_stop_reason']=='bilateral_contact_persisted',
     target_in_pad=None if fraction is None else fraction>=cfg['target_in_pad_fraction'],local_pad_proximity=None if local_fraction is None else local_fraction>=cfg['local_pad_geometry_fraction'],
     bilateral_persistence=None if contact_fraction is None else contact_fraction>=cfg['bilateral_fraction'],
     target_drift=None if drift is None else drift<=cfg['target_drift_m'],tangential_drift=None if max_tangent is None else max_tangent<=cfg['contact_support_tangent_drift_m'],
     end_band_settled=None if end_speed is None else end_speed<=cfg['end_band_speed_m_s'],no_outside_band=record['outside_band_events']==0)
   record.update({'gate_'+key:value for key,value in gates.items()})
   rows.append(record);per.append(record)
  poses.append(dict(number=n,pitch_deg=pose['pitch_deg'],roll_deg=pose['roll_deg'],expected_candidates=expected,completed_candidates=len(per),status_counts=dict(Counter(r['status'] for r in per)),status='no_screened_candidates' if expected==0 else 'complete' if len(per)==expected else 'incomplete'))
 count=Counter(r['status'] for r in rows);count.setdefault('provisional_retention_proxy',0);complete=sum('raw_report' in r for r in rows)
 expected=source['counts']['static_accepted']
 summary=dict(expected_candidates=expected,completed_candidates=complete,complete=complete==expected,status_counts=dict(count),poses=poses,
   criteria=cfg,
   capture_or_retention_success_validated=False,impulses_are_uncalibrated_solver_values=True,toe_support_retained=True)
 summary['failed_gates_nonexclusive']={key:sum(row.get(key) is False for row in rows) for key in dict.fromkeys(k for row in rows for k in row if k.startswith('gate_'))}
 summary['numeric_statistics']={}
 for key in ['stop_gap_m','first_bilateral_gap_m','hold_bilateral_fraction','target_in_pad_fraction','local_bilateral_pad_near_fraction','target_relative_drift_m','contact_support_tangent_drift_max_m','end_band_speed_max_m_s','all_steps_band_speed_max_m_s','edge_stretch_max','left_solver_normal_impulse_sum','right_solver_normal_impulse_sum','left_solver_friction_impulse_sum','right_solver_friction_impulse_sum','pull_follow_ratio','target_tangential_motion_all_hold_m','target_tangential_speed_peak_m_s','target_tangential_speed_p95_m_s']:
  values=[row[key] for row in rows if row.get(key) is not None and np.isfinite(row[key])]
  if values:summary['numeric_statistics'][key]=dict(n=len(values),minimum=float(np.min(values)),median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=float(np.max(values)))
 sensitivity=[]
 for drift_limit in [.001,.002,.003,.005,.010]:
  for contact_limit in [.5,.8,.9]:
   variant=dict(cfg,target_drift_m=drift_limit,contact_support_tangent_drift_m=drift_limit,bilateral_fraction=contact_limit)
   accepted=0
   for row in rows:
    if 'raw_report' not in row:continue
    # All other criteria stay fixed; this does not replay the closure controller.
    status=retention_status(None if row['gate_no_safety_abort'] else 'aborted',row['stop_reason'],row['target_in_pad_fraction'],row['hold_bilateral_fraction'],row['target_relative_drift_m'],row['contact_support_tangent_drift_max_m'],row['end_band_speed_max_m_s'],row['outside_band_events'],row['local_bilateral_pad_near_fraction'],variant)
    accepted+=status=='provisional_retention_proxy'
   sensitivity.append(dict(drift_limit_m=drift_limit,bilateral_fraction=contact_limit,provisional_retention_proxy_count=accepted))
 summary['postprocessing_sensitivity']=sensitivity
 (out/'measurement-report.json').write_text(json.dumps(summary,indent=2)+'\n')
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (out/'measurements.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 with (out/'poses.csv').open('w',newline='') as stream:
  fields=['number','pitch_deg','roll_deg','expected_candidates','completed_candidates','status'];writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(poses)
 reasons=Counter(row.get('stop_reason','pending') for row in rows)
 lines=['# 低速閉鎖・保持の計測結果','',f"記録済み {complete}/{expected} 配置。状態：{'全件完了' if summary['complete'] else '計測中'}。",'',
  '32足首姿勢のうち候補のある29姿勢・436配置を対象に、各配置1試行を独立した初期状態から実施した。',
  'No.9（30°/5°）、No.32（30°/15°）、No.24（30°/30°）は選別候補がなく、閉鎖試験は対象なし。','',
  '速度は指間の開き幅として5mm/s。両側の正のソルバー接触が直近0.1秒続いたら停止し、2秒保持、1秒で5mm引き、1秒保持した。','',
  '## 閉鎖停止理由','', '| 理由 | 件数 |','| --- | ---: |']
 for reason,quantity in reasons.items():lines.append(f'| {reason} | {quantity} |')
 lines+=['','## 暫定保持判定','', '| 判定 | 件数 |','| --- | ---: |']
 for status,quantity in count.items():lines.append(f'| {status} | {quantity} |')

 lines+=['','`provisional_retention_proxy` も物理的な把持成功の検証を意味しない。その他の分類も把持不可能性の証明ではない。','',
  '## 主な測定値','', '| 測定値 | 最小 | 中央値 | 95%点 | 最大 |','| --- | ---: | ---: | ---: | ---: |']
 for key,label,scale in [('stop_gap_m','閉鎖停止の開き幅 mm',1000),('hold_bilateral_fraction','保持中の同時接触率',1),('target_in_pad_fraction','材料点の把持面内残留率',1),('local_bilateral_pad_near_fraction','局所把持面付近の両側近接率',1),('target_relative_drift_m','接触評価後の材料点相対移動 mm',1000),('pull_follow_ratio','引き方向への材料点の追従率',1),('all_steps_band_speed_max_m_s','帯の速度ピーク m/s',1)]:
  stat=summary['numeric_statistics'].get(key)
  if stat:lines.append('| '+label+' | '+' | '.join(f"{stat[k]*scale:.4g}" for k in ['minimum','median','p95','maximum'])+' |')
 lines+=['','欠測は0として集計しない。接触の生ソルバー値はNewtonへ校正しておらず、実機把持力ではない。','',
 '## 撮影済み3配置との対応','', '| 配置 | 元の開き幅 mm | 停止開き幅 mm | 保持中の同時接触率 | 材料点相対移動 mm |','| --- | ---: | ---: | ---: | ---: |']
 for label,sample,yaw,gap in [('E',9,330,.03),('F',10,300,.04),('G',8,60,.05)]:
  match=next((row for row in rows if row['number']==1 and row.get('sample_index')==sample and row.get('yaw_deg')==yaw and row.get('closing_sign')==1 and abs(row.get('opening_gap_m',0)-gap)<1e-8),None)
  if match:
   values=[match.get('stop_gap_m'),match.get('hold_bilateral_fraction'),match.get('target_relative_drift_m')]
   formatted=[('未計測' if value is None else f'{value*scale:.4g}') for value,scale in zip(values,[1000,1,1000])]
   lines.append(f'| {label} | {gap*1000:.0f} | '+' | '.join(formatted)+' |')
 lines+=['','## ファイル','',
 '- [全配置の測定値CSV](measurements.csv)', '- [足首姿勢別CSV](poses.csv)', '- [集計JSON・閾値感度表](measurement-report.json)',
 '- [データ整合性検査](data-validation.json)', '- [比較図](diagnostics.png)',
 '- 各 pose-XX/candidate-NNN.json に生時系列、初期・終了頂点、指の配置、プロトコルを保存。',
 '- [試験条件・計測値の定義・再現方法](../../docs/grasp_closure_measurement.md)','',
 '検査用キネマティック手での計測で、腕のIKやモーター制御は再現していない。布への把持拘束は追加していない。',
 'つま先支持を保ち、布と足の内部接触には既存の凸包補正を用いる。この条件での相対移動指標であり、厳密な界面滑り・自重保持・実機の把持成功を検証したものではない。']
 (out/'README.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='poses'}),flush=True)
if __name__=='__main__':main()
