# 足の剛体衝突形状を実皮膚に合わせる

既定の `config/pose_patterns.yaml` と `config/pose_patterns_toe_contact.yaml` は
`Build/SockDressingPlayer/RigidSkinV1/Player.x86_64` を使う。
`pose_pattern.toe_contact.rigid_skin_contact: true` で、足首姿勢を設定した後に
現在のスキンをBakeし、右足・足指・下腿に対応する皮膚三角形を剛体接触用の
非凸 `MeshCollider` に設定する。表示用の皮膚と幾何探索が参照する面と同じで、
凸包化による足背と開口部付近の膨らみを作らない。

旧来の足の凸包と右脚のボックス計6個は、Unityの剛体接触に対して無効にする。
右グリッパーとの接触は有効、腕などの除外は既存のscene設定を引き継ぐ。
スキンのボーン行列が変わった場合はメッシュを更新する。
初期化を終了すると追加したメッシュを破棄し、旧コライダーの有効状態を復元する。

布の安定化に使う足の内部凸包・摩擦補正は維持する。この内部形状は右手との
剛体衝突には使わない。つま先支持、開口部30mm帯の動的粒子、指のObi接触設定も維持する。
新しい皮膚メッシュにObiColliderは追加しない。

## 検証

```bash
PYTHONPATH=/tmp/sock-grasp-deps:/opt/ros/noetic/lib/python3/dist-packages \
  python3 scripts/validate_rigid_skin_contact.py --all --workers 4
```

`/tmp/sock-grasp-deps` はこの環境で追加の幾何ライブラリを置いた場所。
通常はパッケージの `geometry` extraを導入すれば不要。
出力は `artifacts/rigid-skin-contact/`。全姿勢で次を検査する。

- ネイティブコライダーの全頂点・全三角形が現在のBake済み皮膚と一致する。
- 古い剛体コライダーが無効で、右グリッパーとの衝突ペアが有効。
- 布のつま先支持、自由な帯領域、静定、皮膚・自己交差の検査を通る。
- 既存の探索結果から余裕最大・最小の候補を選び、5個の実グリッパーメッシュを
  Unity上で候補位置に複製し、形状の座標を照合して皮膚との干渉を検査する。
- 皮膚に半径5mmの動的な球を接触させ、実際の衝突イベントを確認する。

候補メッシュと球は検査用に生成・破棄する。ロボットのIK、到達経路、靴下の把持成功は検証しない。
`contact_offset_m` はネイティブの実測値を保存する。幾何的非交差と、動的な接触余裕は区別する。

## 今回の結果

[summary.json](../artifacts/rigid-skin-contact/summary.json) に32姿勢の結果を保存した。
全32姿勢で形状一致・旧コライダー無効化・右手接触有効化・布の静定が通った。
皮膚の頂点誤差は0m。以前の探索データとの皮膚形状の差も0mで、
開口部帯の粒子位置の差は最大約0.22µmだった。

帯内で指接触が有効な既存10,790候補の世界座標に対する皮膚干渉判定は再利用できる。
このうち各姿勢で余裕最大・最小の2候補、計64候補をUnityの実メッシュでも確認し、
全て非干渉だった。全32姿勢の球による物理接触試験も通った。

[布の接触試験](../artifacts/rigid-skin-contact/cloth-contact-response/report.json) では、
帯内738件の接触と約42mmの変位、帯外0件の接触を確認した。
関連するPythonの自動テスト95件も通過した。
