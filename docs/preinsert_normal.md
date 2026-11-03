# 通常形状の途中挿入 A〜E

`artifacts/sock-pose-pattern/sock_pose_patterns.xlsx` の開口位置 A〜E に対応する、
しわ・ねじれ・縁の折り返しを意図的に加えない初期配置です。
足首角度は設定ファイルの既存値を使います。Excelの50行の角度・変形条件を
読み込む機能は含みません。

## 実行

```bash
SOCK_BUILD_LOG_DIR=/tmp/sock-preinsert-unity-logs \
  ./scripts/build_custom_player.sh development Build/SockDressingPlayer/Preinsert
python3 scripts/preview_preinsert.py
# 1段階だけ生成、Eの高さを変更する場合
python3 scripts/preview_preinsert.py --stages E --above-heel-m 0.03 --output /tmp/preinsert-E
```

画像の取得には描画可能な `DISPLAY` が必要です。画像取得時はロボットを一時的に
非表示にし、靴下に検査用の青色マテリアルを適用します。撮影後に元へ戻します。
撮影と保存・復元の際は物理時間を進めません。

既定の出力先は `artifacts/sock-pose-pattern/preinsert-normal/` です。

- `A-E-overview.jpg`: 各段階の側面・上面画像。
- `A-side.png` など: Unityが描画した画像。
- `A-geometry.json` など: 足の形状、粒子、三角形、自然長、配置条件と検査結果。
- `A-prepared-state.json` など: 同じシーン・メッシュ・開口フレームで復元できる状態。
- `report.json`: 全段階の位置と検査結果。

## 初期配置の定義

Excelには座標軸・寸法付きの参照画像がないため、次の幾何学的な解釈を採用します。

| 段階 | 開口位置 |
| --- | --- |
| A | つま先端からDまでの滑らかな経路の1/4 |
| B | 同2/4 |
| C | 同3/4 |
| D | かかとから足首への境目。足の甲の表面より上ですねを囲む断面 |
| E | Dからすねの軸方向へ `above_heel_m`（既定5 cm） |

足の外形はUnityの解剖学的な足メッシュと接触用ボックスから取得します。
Dの断面は、足首のボーン位置からすね方向に測った足表面の最大位置に
`clearance_m`（既定15 mm）を加えた位置です。
そのボーンからの距離は `heel_cuff_above_ankle_m` に記録します。
A〜Cはワールド座標の直線距離ではなく、足首へ曲がる経路の長さを分割します。
断面の中心は皮膚形状に合わせて横方向に補正するため、ボーンの中心とは一致しません。
実験の目印と照合する場合は、この定義を調整してください。

靴下は既存の長さ300 mm・半径40 mmの閉じたつま先付きメッシュです。
余った長さはつま先の先に伸ばし、しわとして束ねません。
D/Eで長さが足りない場合は初期配置を軸方向に伸ばし、倍率を
`axial_initial_scale` に記録します。1.5倍を超える要求は拒否します。
形状は滑らかな楕円断面をつなぎ、開口部は平面内の楕円とします。

## シミュレータへの組み込み

`config/preinsert_normal.yaml` は専用Playerと把持なしの初期化設定です。
`SockDressingEnv.apply_scenario()` が通常のシーン配置の最後に `preinsert` を適用します。
適用直後の最初の `observe()` は物理時間を進めず、配置した状態を撮影します。
それ以降の観測・コマンドは通常の物理ステップに戻ります。

```yaml
preinsert:
  stage: C
  above_heel_m: 0.05
  clearance_m: 0.015
```

既にロードしたnative custom Playerには以下でも適用できます。

```python
report = environment.initialize_preinsert("C")
```

初期化は接触ロールアウト開始前に行います。足との交差・布の自己交差を検査した後、
左右の把持を外し、粒子位置・前フレーム位置を設定して速度をゼロにします。
Unityから取得した位置との誤差も照合します。対応APIのない古いPlayerは拒否します。
ロボットで引き上げる場合は、この配置後の開口を改めて把持する処理が必要です。

## 検証の範囲

初期配置の検査は、粒子の体内侵入、布の辺による人体横断、人体の辺による
布面横断、隣接しない布要素の交差を対象とします。開口の平面性、座標系の回転・移動、
nativeの粒子並べ替えと丸め誤差もテストします。保存・復元はnative状態の再取得で照合します。

**この段階の生成結果は幾何学的な初期状態です。物理的に平衡した状態ではありません。**
自然長・物性・衝突条件は書き換えません。初期形状には伸張があり、既存の材料の
許容伸びを超える辺もあります。`maximum_edge_stretch` と `p95_edge_stretch` に
記録しており、`physical_settling_validated` は `false` です。
物理時間を進めると形が変わる可能性があるため、安定性や把持後の動作は別途検証が必要です。
