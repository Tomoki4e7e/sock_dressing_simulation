# held-Aのサイズを保ち、自由な先端を重力で垂らす

`config/pose_patterns_legacy_drape.yaml`は、保存済みheld-A形状で足首のpitch・rollを設定した後、
足に掛かる部分を保ち、余った先端側を重力で垂らします。
しわ・つぶれ・ねじれ・折り返しの追加生成は無効です。

## 先端が浮いていた原因

重力の初期設定は`[0, -9.81, 0] m/s²`でした。
前の静止再現では全801粒子を固定し、毎ステップ座標と速度を保存値へ戻していたため、
重力があっても先端は落下しませんでした。
現在は足に掛かる約7.5 cmだけを保持し、自由部分を重力で動くモデルへ接続しています。

## 現在の物理モデル

視覚再現を優先し、自由部分は**断面ごとの質点を重力で動かす簡略モデル**です。
全布粒子を独立に動かすObiの布ソルバーではありません。

- 足側7断面・224頂点は保存済みの位置を保持します。
- 自由側18断面に合計約36 gの質量を置きます。
- 断面中心を重力で積分し、断面間の長さ、曲がり過ぎ、離れた断面同士の接近を拘束します。
- 描画メッシュの残り577頂点は、この動的な中心線へ連続的に追従します。
- ネイティブ布粒子自体は描画用の追従点です。`native_mesh_finite_mass_particles`は0です。
  物理的な有限質量は`dynamic_sections`と`free_mass_kg`に記録します。
- 収束用の減衰を使い、準備後も重力モデルは動作を続けます。先端のワールド位置は固定しません。

固定部と自由部の接続は、元のheld形状の傾き・凹凸を引き継いで補間します。
境界の交差が残る場合は、物理中心線を変えずに接続範囲を調整して再検査します。
準備中は他物体のObiコライダーを無効にし、ロボット等に先端が支えられたり押し飛ばされたりする影響を除きます。
これは初期姿勢の視覚再現用です。布の素材力学、グリッパーとの接触、着衣の引き上げ力の検証には
別途接続が必要です。リーチング・把持の開始は現在も保護されています。

## 参照形状と姿勢

参照先はユーザー指定の
`artifacts/sock-pose-pattern/preinsert-fitted/A/contraction-overview.jpg`のheldパネルに対応する
`fitted-geometry.json`・`fitted-prepared-state.json`・`report.json`です。参照ファイルは変更しません。

Excelの50行に対応する32種類のpitch・rollを適用します。
開口位置はAのままで、B〜E、前後ずらし、斜め回転、開口変形、S01〜S05の寸法・素材差は適用しません。
pitch 0°は足先が水平で甲が鉛直上向き、正はつま先を持ち上げる方向です。
roll正はpitch適用後の足首→つま先軸の右手方向です。

初期配置では保存座標を倍率1で回転します。アバターの姿勢変更によるスキン差を避けるため、
必要な場合だけつま先方向へ最大8 mm平行移動します。その後、自由部分は重力で形を変えます。
足側と開口の大きさは保持し、開口外周は約26.4846 cmです。
`maximum_edge_length_error_m`等の剛体不変検査は重力で垂らす前の初期配置を対象とします。

## 設定と実行

```yaml
pose_pattern:
  held_reference: artifacts/sock-pose-pattern/preinsert-fitted/A
  gravity_drape:
    enabled: true
    pinned_length_m: 0.075
    settle_steps: 150
```

`settle_steps: 150`は3秒間の物理計算です。
`enabled: false`は以前の全固定モードになり、先端も落下しません。

```bash
python3 -m pip install -e '.[fit,test]'
SOCK_BUILD_LOG_DIR=/tmp/sock-gravity-build \
  ./scripts/build_custom_player.sh development Build/SockDressingPlayer/HeldGravityV8
python3 scripts/preview_pose_patterns.py --config config/pose_patterns_legacy_drape.yaml --numbers 1 2 6 13 --hold-steps 50
```

番号の指定を省くと全50行を生成します。画像なしの検証は`--headless`です。
同じ初期形状・同じカメラで重力を切り替える比較試験：

```bash
python3 scripts/check_gravity_drape.py
```

通常の`SockDressingEnv.apply_scenario()`でも同じ初期化・重力計算・検査を行います。

## 検証結果と保存先

修正後の出力は`artifacts/sock-pose-pattern/held-gravity/`です。
全50行で、重力・足側保持・開口サイズ・実スキンと布同士の交差検査を通過しました。
各姿勢で3秒の準備後、さらに50ステップ（1秒）を進めて検査しています。

- 先端の低下量：約12.2〜21.8 cm。
- 開口外周：26.484576〜26.484585 cm。
- 足側の保持位置の変化：0 m。
- 終盤の先端位置変化：最大約0.89 mm。
- 保持検査後の断面速度：最大約0.87 mm/s。
- 検査した最終状態の足・布の交差、布同士の交差：0。

重力比較試験では、自由部分への重力を0にした3秒間の先端移動量は0 m、
通常重力に戻した3秒間の低下量は約21.8 cmでした。

- `overview-1.jpg`：代表4姿勢（1・2・6・13番）の側面・甲側の画像。
- `gravity-control/gravity-comparison.jpg`：初期状態・重力ゼロ・通常重力の比較。
- `gravity-control/report.json`：比較試験の重力値・質量・時計・落下量・交差検査。
- `validation-summary.json`：全50行の集計。
- `report.json`：全50行の結果と検証元の記録。
- 各`report.json`：重力計算中の時系列、動的断面数、保持誤差、交差検査等。
- 各`visual-state.json`・`runtime-config.yaml`：頂点・面と再実行用の設定。

現在の比較画像を保存したのは1・2・6・13番で、それ以外は画像なしで同じ検査を実施しました。
表面交差検査は、任意の共面接触や体積包含を完全に保証するものではありません。
以前の[局所変形モード](pose_patterns_local.md)は現在の既定設定では無効です。
