#!/usr/bin/env python3
"""Build a browsable local video/measurement report from native policy trials."""
import argparse,csv,html,json,shutil
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path);a=ap.parse_args();run=a.run.resolve();out=a.output.resolve() if a.output else run/'live-test-motion'
    suite=json.loads((out/'report.json').read_text());split=suite.get('dataset_split','test');selected=suite.get('selected_case_ids')
    expected=[r for r in json.loads((run/'manifest.json').read_text())['rows'] if r['split']==split and (selected is None or r['id'] in selected)];rows=[];cards=[];table=[]
    for item in expected:
        path=out/item['id']/'policy/report.json'
        if not path.exists():continue
        r=json.loads(path.read_text());row=dict(id=item['id'],completed=r['completed'],steps=r['steps'],simulated_duration_s=r['simulated_duration_s'],native_penetration_event_count=r['native_penetration_event_count'],overview_video=str(Path(r['overview_video']).relative_to(out)),head_video=str(Path(r['head_video']).relative_to(out)))
        for side in ('left','right'):
            g=r.get('geometry',{}).get(side,{})
            for key in ('cloth_surface_area_between_fingers_mm2','material_15_30mm_band_area_between_fingers_mm2','opening_rim_length_between_fingers_mm','opening_rim_fraction_percent','cloth_present','opening_rim_present'):
                row[side+'_'+key]=g.get(key)
            c=g.get('contact_geometry_checks',{})
            row[side+'_target_contact_geometry_passed']=c.get('passed')
            cloth=c.get('contacts',{}).get('cloth',{});foot=c.get('contacts',{}).get('foot',{})
            row[side+'_cloth_band_distance_mm']=None if cloth.get('distance_m') is None else cloth['distance_m']*1000
            row[side+'_cloth_band_intersects']=cloth.get('intersects')
            row[side+'_foot_distance_mm']=None if foot.get('distance_m') is None else foot['distance_m']*1000
        rows.append(row)
        def n(k):return '—' if row.get(k) is None else f'{row[k]:.3f}'
        table.append(f'<tr><td>{row["id"]}</td><td>{n("left_cloth_surface_area_between_fingers_mm2")}</td><td>{n("right_cloth_surface_area_between_fingers_mm2")}</td><td>{n("left_opening_rim_length_between_fingers_mm")}</td><td>{n("right_opening_rim_length_between_fingers_mm")}</td><td>{n("left_cloth_band_distance_mm")}</td><td>{n("right_cloth_band_distance_mm")}</td></tr>')
        base=f'{item["id"]}/policy'
        details=''.join(f'<a href="{base}/{side}-{view}.png"><img loading="lazy" src="{base}/{side}-{view}.png" alt="{side} {view}"></a>' for side in ('left','right') for view in ('gap-side','gap-opposite')) if r.get('geometry') else ''
        cards.append(f'<article><h2>{row["id"]}</h2><video controls preload="none" poster="{base}/overview-final.png" src="{base}/overview.mp4"></video><p>爪間の布表面積：左 {n("left_cloth_surface_area_between_fingers_mm2")} mm² ／右 {n("right_cloth_surface_area_between_fingers_mm2")} mm²<br>爪間の開口縁：左 {n("left_opening_rim_length_between_fingers_mm")} mm ／右 {n("right_opening_rim_length_between_fingers_mm")} mm</p><div class="closeups">{details}</div><p><a href="{base}/terminal-cloth-between-fingers.json">終端の測定値</a> · <a href="{base}/head-input.mp4">頭部入力映像</a> · <a href="{base}/motion-trace.jsonl">動作ログ</a></p></article>')
    columns=['id','completed','steps','simulated_duration_s','native_penetration_event_count']+[side+'_'+k for side in ('left','right') for k in ('cloth_surface_area_between_fingers_mm2','material_15_30mm_band_area_between_fingers_mm2','opening_rim_length_between_fingers_mm','opening_rim_fraction_percent','cloth_band_distance_mm','cloth_band_intersects','foot_distance_mm','cloth_present','opening_rim_present','target_contact_geometry_passed')]+['overview_video','head_video']
    with (out/'terminal-measurements.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    aggregate=dict(expected=len(expected),finished=len(rows),completed_episodes=sum(r['completed'] for r in rows),left_cloth_present=sum(bool(r['left_cloth_present']) for r in rows),right_cloth_present=sum(bool(r['right_cloth_present']) for r in rows),both_cloth_present=sum(bool(r['left_cloth_present'] and r['right_cloth_present']) for r in rows),both_target_contact_geometry_passed=sum(bool(r['left_target_contact_geometry_passed'] and r['right_target_contact_geometry_passed']) for r in rows),rows=rows)
    (out/'terminal-summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    doc='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU閉ループ推論・爪間の布検証</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#eef2f6;color:#182537;margin:24px auto;max-width:1400px;padding:0 18px}h1{font-size:28px}h2{margin:0 0 12px}a{color:#12639b}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 12px;border:1px solid #d5dee8;text-align:right}th{font-size:13px}article{background:white;border-radius:12px;padding:20px;margin:24px 0}video{width:100%;max-height:760px;background:#111}.closeups{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.closeups img{width:100%}.note{background:#fff8df;padding:16px;border-radius:8px}.scroll{overflow:auto}</style><h1>GPU閉ループ推論：俯瞰動画と爪間の布検証</h1><p>モデル：SARNN_best.pth、epoch 4780。テスト12状態。Unityの実測RGB・関節角・駆動力から、SAMDAMSARNN・SAM2・Depth AnythingをGPUで実行し、予測関節角をモーターに指令しています。</p><p class="note">人体と布は教示と同じ固定条件です。保存画像を入力した先のオフライン予測評価とは別の実験です。爪間の布表面積と開口縁長は、終端の実測リンク姿勢と左右の爪の有限凸包から測定しています。布が爪間に存在しても、閉鎖・摩擦保持・引き上げの成立は示しません。駆動力の入力は過去29サンプルまでを使う因果的平滑化で、学習時の前後サンプルによる平滑化とは異なります。</p><p><a href="terminal-measurements.csv">測定CSV</a> · <a href="terminal-summary.json">集計JSON</a> · <a href="report.json">実行結果</a></p><div class="scroll"><table><thead><tr><th>状態</th><th>左 布面積 mm²</th><th>右 布面積 mm²</th><th>左 開口縁 mm</th><th>右 開口縁 mm</th><th>左 指定爪と布帯の距離 mm</th><th>右 指定爪と布帯の距離 mm</th></tr></thead><tbody>'''+''.join(table)+'</tbody></table></div>'+''.join(cards)+'</html>'
    suite=json.loads((out/'report.json').read_text())
    epoch=suite['checkpoint_epoch']
    label={'train':'教示データ（学習に使用）','validation':'検証データ','test':'テストデータ'}[split]
    doc=doc.replace('モデル：SARNN_best.pth、epoch 4780。テスト12状態。',f'モデル：{html.escape(Path(suite["checkpoint"]).name)}、epoch {epoch}。{label} {len(expected)}状態。')
    if json.loads((run/'configuration.json').read_text())['model'].get('name')=='SAMDAMSARNNFlow':
        doc=doc.replace('SAMDAMSARNN・SAM2', 'SAMDAMSARNNFlow・SAM2')
    doc=doc.replace('保存画像を入力した先のオフライン予測評価とは別の実験です。','各時刻のシミュレータ観測をモデルに入力した閉ループ実行です。')
    (out/'report.html').write_text(doc)
    print(json.dumps({k:v for k,v in aggregate.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
