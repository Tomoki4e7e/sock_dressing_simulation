#!/usr/bin/env python3
"""Publish the static-gravity evidence and its limits after native validation."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.gravity_reference import StaticGravityReference,ARM_COLUMNS,ARM_NAMES


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/gravity-calibration-v1');a=ap.parse_args();out=a.output.resolve()
    report=json.loads((out/'report.json').read_text());model=StaticGravityReference(out/'gravity-model.json')
    native=json.loads((out/'native-validation.json').read_text());manifest=json.loads((out/'reference-manifest.json').read_text())
    contact=json.loads((out/'contact-review.json').read_text());routes=json.loads((out/'simulation-routes/manifest.json').read_text())
    if not native['passed'] or not all(r['visually_separated_both_grippers'] for r in contact['rows']):raise ValueError('Native or image review incomplete')
    if routes['frames']!=6607 or routes['episodes']!=12:raise ValueError('Incomplete reaching gravity export')
    for row in manifest['rows']:
        for name,digest in row['sha256'].items():
            if hashlib.sha256((Path(row['path'])/(name+'.csv')).read_bytes()).hexdigest()!=digest:raise ValueError('Archive CSV copy changed')
    for row in routes['rows']:
        for name,digest in row['source_sha256'].items():
            if hashlib.sha256((Path(row['source'])/name).read_bytes()).hexdigest()!=digest:raise ValueError('Original learning signals changed')
    labels=[('L' if n.startswith('left') else 'R')+n.rsplit('_',1)[1] for n in ARM_NAMES]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    keys=['zero_gravity_baseline_vs_proxy','test_nominal_gravity_vs_proxy','test_calibrated_gravity_vs_proxy']
    values=[report[k]['rmse_Nm'] for k in keys];axes[0].bar(['Zero gravity','Nominal URDF','Calibrated'],values,color=['#88909a','#5d93bd','#2a8d70']);axes[0].set_yscale('log');axes[0].set_ylabel('RMSE [N m], log scale');axes[0].set_title('Held-out released low-speed samples')
    for i,v in enumerate(values):axes[0].text(i,v*1.1,f'{v:.4f}',ha='center')
    for key,label in zip(keys[1:],['Nominal URDF','Calibrated']):axes[1].plot(labels,report[key]['per_joint_rmse_Nm'],'o-',label=label)
    axes[1].set_ylabel('RMSE [N m]');axes[1].set_title('Controller gravity proxy, by arm joint');axes[1].legend();axes[1].grid(alpha=.25)
    fig.savefig(out/'gravity-errors.png',dpi=160);plt.close(fig)
    test=next(r for r in manifest['rows'] if r['split']=='test');q=np.loadtxt(Path(test['path'])/'angle.csv',delimiter=',');tau=np.loadtxt(Path(test['path'])/'torque.csv',delimiter=',')[:,ARM_COLUMNS];ext=np.loadtxt(Path(test['path'])/'external_torque.csv',delimiter=',')[:,ARM_COLUMNS];g=model.predict(q)
    segment=next(r for r in report['segments'] if r['id']==test['id']);time=np.arange(len(q))*.1
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    for axis,joint in zip(axes,(0,1,3)):
        axis.plot(time,tau[:,joint],color='#999999',label='Recorded link effort')
        axis.plot(time,(tau-ext)[:,joint],color='#cf8135',ls='--',label='link - external (proxy)')
        axis.plot(time,g[:,joint],color='#167962',label='Calibrated static gravity')
        axis.axvspan(segment['start']*.1,(segment['end_exclusive']-1)*.1,color='#6cb69c',alpha=.25,label='Selected released low-speed interval')
        axis.set_ylabel(labels[joint]+' [N m]');axis.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='best');axes[-1].set_xlabel('Nominal archive time [s]; original stamps unavailable')
    fig.savefig(out/'gravity-trajectory.png',dpi=160);plt.close(fig)
    with (out/'per-joint-errors.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['joint','nominal_vs_proxy_RMSE_Nm','calibrated_vs_proxy_RMSE_Nm','gravity_vs_raw_link_effort_RMSE_Nm'])
        for i,name in enumerate(ARM_NAMES):writer.writerow([name,report['test_nominal_gravity_vs_proxy']['per_joint_rmse_Nm'][i],report['test_calibrated_gravity_vs_proxy']['per_joint_rmse_Nm'][i],report['test_gravity_vs_raw_link_effort']['per_joint_rmse_Nm'][i]])
    proxy=report['test_calibrated_gravity_vs_proxy']['rmse_Nm'];raw=report['test_gravity_vs_raw_link_effort']['rmse_Nm']
    before=json.loads((out/'native/inertials-applied.json').read_text());physical=model.tree
    helper=[b['name'] for b in before['before'] if physical.masses[b['name']]==0 and b['mass_kg']>.1]
    report.update(native_validation_passed=True,contact_image_review_passed=True,original_learning_dataset_unchanged=True,
        gravity_reference_complete=True,native_validation_summary={k:v for k,v in native.items() if k!='rows'},
        missing_inertial_helpers_corrected=helper,source_signal_hashes_verified=True,report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'report.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>実機アーカイブに基づく重力負荷の基準モデル</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:7px 14px;border:1px solid #ccc;text-align:left}}code{{background:#eee}}</style>
<h1>実機アーカイブに基づく重力負荷の基準モデル</h1>
<p>指定tarの靴下11経路から、両グリッパーを開いて布から離れた終端の低速区間を135フレーム抽出しました。元の9 train経路114フレームで較正・経路単位交差検証を行い、未使用の2 test経路21フレームで評価しました。対象は左右の腕14関節、単位はNmです。グリッパー4列の力較正は含みません。</p>
<p>保存元は <code>ToroboState.link_effort</code> と <code>external_effort</code> です。アーカイブには <code>gravity_effort</code> 自体が保存されていません。低速区間の <code>link_effort - external_effort</code> を、制御器による重力成分の推定値として比較しました。これは独立した重力センサーの正解ではありません。生の <code>torque.csv</code> には静止摩擦・オフセットなどの残差があり、重力だけの値とは区別しています。</p>
<table><tr><th>評価</th><th>テストRMSE [Nm]</th></tr><tr><td>重力ゼロ → 重力推定値</td><td>{values[0]:.5f}</td></tr><tr><td>元URDF → 重力推定値</td><td>{values[1]:.5f}</td></tr><tr><td>較正モデル → 重力推定値</td><td>{proxy:.5f}</td></tr><tr><td>較正した重力のみ → 生のlink_effort</td><td>{raw:.5f}</td></tr></table>
<p>質量と重心の小さな補正を、元URDFへの正則化と物理的な範囲制約付きで推定しました。64パラメータに対して重力回帰行列のランクは28です。個々の質量・重心を一意に実測同定した結果ではありません。慣性テンソル・摩擦・布の反力は同定していません。胴体姿勢はアーカイブの靴下用 <code>kb</code> 設定[-45,100,0]度を根拠にしていますが、各動画に時刻付きで保存されていた値ではありません。</p>
<img src="gravity-errors.png" alt="重力推定誤差"><img src="gravity-trajectory.png" alt="実機と同じ角度列で計算した重力負荷">
<h2>Unityでの重力の再現</h2><p>質量指定のない補助リンク{len(helper)}個にUnity既定の1kgが残っていました。専用APIでURDFに合わせ、質量ゼロの補助リンクには数値計算用の1mg（0.000001kg）のみを設定しました。姿勢を直接設定する際のグリッパー連動軸も同期しています。関節重力を有効にし、135姿勢で解析モデルとの最大誤差は{native['maximum_native_vs_analytic_error_Nm']:.6f} Nmでした。姿勢保持中の駆動力と重力負荷のRMSEは{native['static_hold_drive_vs_gravity_rmse_Nm']:.6f} Nmです。接触を無効にした重力単独の検証です。</p>
<p>実機CSVの一部はURDF上限を最大{native['maximum_source_limit_overrun_rad']:.8f} radだけ超えています。Unity比較ではその分のみ上限に合わせ、元CSVは変更していません。135フレームの撮影時刻は公称10Hzから推定し、元の高周波時刻・加速度は利用できていません。</p>
<h2>抽出した区間</h2><p>各行は区間の先頭・中央・末尾です。両グリッパーと布の離隔を画像で確認しました。外力値はゼロ判定に使っていません。力センサーによる厳密な無接触証明はありません。</p><a href="selected-contact-review.jpg"><img src="selected-contact-review.jpg" alt="抽出区間の画像確認"></a>
<h2>利用方法と保存先</h2><p><code>gravity-model.json</code> と <code>calibrated_gravity.urdf</code> が基準モデルです。<code>StaticGravityReference.predict(q18)</code> は腕14関節の重力負荷を返し、<code>configure_unity(env)</code> は専用RegisteredGravityV1プレイヤーで質量・重心・重力を適用します。静止残差のオフセットは別項で保存し、物理的な重力には加えていません。</p>
<p>既存12経路の全6,607姿勢に対する結果は <code>simulation-routes/</code> の <code>gravity_arm.csv</code> です。元の18列 <code>torque.csv</code>・画像・ログを保持しています。出力14列をそのまま既存18列トルクとして読み込まないでください。従来ShareSetの学習対象は今回自動変更していません。</p>
<p><a href="report.json">集計・選定基準</a> · <a href="gravity-model.json">モデル</a> · <a href="native-validation.json">Unity検証</a> · <a href="contact-review.json">画像確認</a> · <a href="per-joint-errors.csv">関節別誤差</a> · <a href="simulation-routes/manifest.json">12経路の重力出力</a></p></html>''')
    print('GRAVITY REPORT COMPLETE',out/'report.html',flush=True)


if __name__=='__main__':main()
