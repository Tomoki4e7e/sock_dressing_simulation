#!/usr/bin/env python3
"""Build a browsable index of validated, synchronized contact demonstrations."""
import argparse,csv,html,json,os
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

STYLE='body{font:16px/1.65 sans-serif;color:#172330;max-width:1240px;margin:28px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ccd4dc;text-align:left}a{color:#075bb4}.notice{background:#e9f3fa;padding:16px}.views{display:flex;gap:8px}.views figure{width:33%;margin:0}.views img{width:100%}input{width:75%}code{background:#eef1f5;padding:3px}'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--additional',nargs='*',type=Path,default=[])
    args=parser.parse_args();out=args.dataset.resolve();roots=[out]+[p.resolve() for p in args.additional]
    selected={};attempts=[]
    for root in roots:
        for p in sorted((root/'attempts').glob('*/status.json')):
            status=json.loads(p.read_text());status['root']=str(root);attempts.append(status)
            if status.get('finished') and status.get('passed'):selected[status['id']]=status
    viewers=out/'viewers';viewers.mkdir(exist_ok=True);rows=[]
    for ident,status in sorted(selected.items()):
        root=Path(status['root']);episode=root/status['episode'];native=episode.parent/'native-route-initial'
        meta=json.loads((episode/'metadata.json').read_text());motion=json.loads((native/'replay-result.json').read_text())
        route=json.loads((ROOT/meta['trajectory']).read_text())
        frames=list(csv.DictReader((episode/'frames.csv').open()))
        assert meta['contact_pregrasp_learning_ready'] and motion['passed'] and len(frames)==meta['frames']
        condition=status['condition'];row=dict(id=ident,**{k:condition[k] for k in ['stage','number','scene_number','pitch_deg','roll_deg']},
            episode=os.path.relpath(episode,out),frames=len(frames),duration_s=meta['physics_steps']*.02,
            image_size=meta['image_size'],image_rate_hz=meta['image_rate_hz'],physics_rate_hz=50,
            trajectory=meta['trajectory'],trajectory_sha256=meta['trajectory_sha256'],
            initial_sha256=meta['shared_initial_sha256'],drive_profile=meta.get('drive_profile'),
            human_collision_model=meta.get('human_collision_model'),planning_proofs=meta.get('planning_proofs'),
            duration_scale=meta.get('duration_scale',1.),
            native_clearance_repairs=route.get('native_clearance_repairs',([route['native_clearance_repair']] if 'native_clearance_repair' in route else [])),
            maximum_tracking_error_m=motion['maximum_tracking_error_m'],final_contact_checks=motion['final_contact_checks'],
            actual_final_x=motion['actual_final_x'])
        rows.append(row);prefix=os.path.relpath(episode,viewers)
        images=''.join(f'<figure><figcaption>{label}</figcaption><a id="link_{side}" target="_blank"><img id="{side}"></a></figure>'
                       for side,label in [('left','頭部・左'),('right','頭部・右'),('overview','確認用カメラ')])
        links=' · '.join(f'<a href="{prefix}/{name}">{label}</a>' for name,label in
                        [('metadata.json','メタデータ'),('angle.csv','関節角'),('torque.csv','駆動力'),('external_torque.csv','外力信号'),('frames.csv','フレーム時刻'),('raw-validation.json','同期検査')])
        distances=[]
        for side,label in [('left','左'),('right','右')]:
            contact=motion['final_contact_checks'][side]
            values=['許容範囲内の交差' if v['intersects'] else f"{v['distance_m']*1000:.4f} mm" for v in contact['contacts'].values()]
            distances.append(f'<tr><td>{label}</td><td>{values[0]}</td><td>{values[1]}</td><td>{contact["rim_length_m"]*1000:.3f} mm</td></tr>')
        page=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>{ident} 接近教示</title><style>{STYLE}</style>
<a href="../report.html">全状態の一覧</a><h1>{ident} — 合格した接近教示</h1>
<p>足姿勢：pitch {condition['pitch_deg']}° / roll {condition['roll_deg']}°。{len(frames)}フレーム、{row['duration_s']:.2f}秒。画像は{meta['image_size'][0]}×{meta['image_size'][1]}、10 Hz。</p>
<p>{links}</p><button id="play">再生 / 停止</button><p><input id="frame" type="range" min="0" max="{len(frames)-1}" value="0"><span id="label"></span></p>
<div class="views">{images}</div><p id="clock"></p>
<table><tr><th>グリッパー</th><th>布との距離</th><th>人体との距離</th><th>爪間の開口境界長</th></tr>{''.join(distances)}</table>
<p>接触判定は0.5 mm以内。規定の局所領域では0.5 mmまでの侵入を許容しています。靴下・人体姿勢は固定し、最後の把持閉鎖は行っていません。</p>
<script>const frames={json.dumps(frames)},prefix={json.dumps(prefix)};const slider=document.getElementById('frame');let timer;
function show(){{const f=frames[+slider.value];for(const side of ['left','right','overview']){{let url=prefix+'/camera_'+side+'/'+f.frame+'.png';document.getElementById(side).src=url;document.getElementById('link_'+side).href=url;}}document.getElementById('label').textContent=f.frame+' / '+(frames.length-1);document.getElementById('clock').textContent='時刻 '+Number(f.time_s).toFixed(2)+' s / '+f.stage+' / 物理ステップ '+f.step;}}
slider.oninput=show;document.getElementById('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;return}}timer=setInterval(()=>{{slider.value=(+slider.value+1)%frames.length;show()}},100)}};show();</script></html>'''
        (viewers/f'{ident}.html').write_text(page)
    counts=Counter(r['stage'] for r in rows)
    index=dict(expected=70,accepted=len(rows),all_70_passed=len(rows)==70,groups=dict(counts),
               total_frames=sum(r['frames'] for r in rows),total_images=3*sum(r['frames'] for r in rows),
               total_duration_s=sum(r['duration_s'] for r in rows),cloth_frozen=True,final_closure_performed=False,
               criteria=dict(contact_tolerance_m=.0005,local_intrusion_allowance_m=.0005,standing_left_leg_floor_allowance_m=.15),
               scope='Frozen-cloth paired contact pregrasp; continuous commanded-route certificates, native joint-drive validation and synchronized RGB/measured signals',
               actual_motion_verification='Native contacts recorded every physics tick; measured geometry checked every five ticks and at segment ends',rows=rows)
    write(out/'dataset-index.json',index)
    with (out/'dataset-index.csv').open('w',newline='') as stream:
        fields=['id','stage','number','scene_number','pitch_deg','roll_deg','episode','frames','duration_s','duration_scale','maximum_tracking_error_m']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    with (out/'measured-grasp-poses.csv').open('w',newline='') as stream:
        fields=['id','stage','side','tcp_x_m','tcp_y_m','tcp_z_m','opening_gap_m',
                'cloth_distance_m','foot_distance_m','cloth_intersects','foot_intersects','rim_length_m']
        fields += ['rotation_'+str(i)+str(j) for i in range(3) for j in range(3)]
        fields += ['joint_'+str(i)+'_rad' for i in range(1,8)]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            for side,offset,gap_index in [('left',0,14),('right',7,15)]:
                contact=row['final_contact_checks'][side]
                pose=dict(id=row['id'],stage=row['stage'],side=side,opening_gap_m=row['actual_final_x'][gap_index],rim_length_m=contact['rim_length_m'])
                pose.update(zip(['tcp_x_m','tcp_y_m','tcp_z_m'],contact['tcp_world_m']))
                for kind in ['cloth','foot']:
                    pose[kind+'_distance_m']=contact['contacts'][kind]['distance_m']
                    pose[kind+'_intersects']=contact['contacts'][kind]['intersects']
                pose.update({'rotation_'+str(i)+str(j):contact['rotation'][i][j] for i in range(3) for j in range(3)})
                pose.update({'joint_'+str(i+1)+'_rad':row['actual_final_x'][offset+i] for i in range(7)})
                writer.writerow(pose)
    body=''.join(f'<tr><td><a href="viewers/{r["id"]}.html">{r["id"]}</a></td><td>{r["pitch_deg"]}° / {r["roll_deg"]}°</td><td>{r["frames"]}</td><td>{r["duration_s"]:.2f}</td><td>{1000*r["maximum_tracking_error_m"]:.3f}</td></tr>' for r in rows)
    failed=[s['id'] for s in attempts if s.get('finished') and not s.get('passed') and s['id'] not in selected]
    groups=' / '.join(f'{stage}: {counts[stage]}' for stage in 'ABCDE')
    timing=', '.join(f'{r["id"]}: {r["duration_scale"]:g}倍' for r in rows if r['duration_scale']!=1)
    timing_note=f'<p>軌道形状を保って実行時間を延長した状態：{html.escape(timing)}。画像10 Hz・物理計算50 Hzは共通です。</p>' if timing else ''
    adjusted=[r['id'] for r in rows if r['native_clearance_repairs']]
    extra_links=''.join(f' · <a href="{name}">{label}</a>' for name,label in [('README.txt','データ形式・使い方'),('completion.json','全状態の最終検査')] if (out/name).exists())
    page=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>画像付き接近教示データ</title><style>{STYLE}</style>
<h1>画像付き接近教示データ — {len(rows)} / 70 状態を採用</h1>
<p class="notice">{groups}<br>各視点の合計 {index['total_frames']:,} フレーム、3視点で {index['total_images']:,} 枚。実測関節角・駆動力は50 Hz、画像と対応するCSVは10 Hzです。</p>
<p>人体の実表面メッシュを使用し、補助カプセルとロボットの衝突を除外した設定です。左右の駆動設定と時間補償をそろえ、採用済みのうち{len(adjusted)}状態では布または人体との隙間を確保するため約0.05 mmの経路調整を行いました（{html.escape(', '.join(adjusted))}）。共通初期位置、人体・椅子の配置、接触・侵入許容値は維持しています。</p>
{timing_note}
<p>採用には、使用軌道に対応した終点・連続時間検査の証明、関節駆動後の実測検査、画像・関節データの同期検査が必要です。公称余裕で証明した64状態と、既定の1 µm縮小余裕で証明した6状態の条件を引き継いでいます。</p>
<p>靴下と人体姿勢を固定した接近データです。最後の把持閉鎖は行っていません。接触距離0.5 mm以内、規定の局所領域の侵入0.5 mm以内、接地側左足と床の既存許容150 mmを引き継いでいます。実測運動は物理接触を毎ステップ、幾何形状を5ステップごとと区間終点で検査しています。</p>
<p><a href="dataset-index.json">データ一覧 JSON</a> · <a href="dataset-index.csv">データ一覧 CSV</a> · <a href="measured-grasp-poses.csv">実測終点姿勢 CSV</a> · <a href="collection.json">実行・同期検査の集計</a> · <a href="integrity-summary.json">画像・入力ファイルの照合</a> · <a href="programs/manifest.json">実行コードの記録</a>{extra_links}</p>
<p>未採用の実行結果：{html.escape(', '.join(sorted(set(failed))) or '現時点ではなし')}。未実行・実行中の状態は採用数に含みません。</p>
<table><tr><th>状態・画像再生</th><th>足の pitch / roll</th><th>フレーム数</th><th>時間 (秒)</th><th>最大追従誤差 (mm)</th></tr>{body}</table></html>'''
    (out/'report.html').write_text(page);print('REPORT',len(rows),'accepted',index['total_frames'],'frames',flush=True)


if __name__=='__main__':main()
