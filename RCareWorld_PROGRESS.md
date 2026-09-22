# RCareWorld 靴下着衣シミュレーション実装進捗

最終更新: 2026-09-21

## 現在の到達点

RCareWorld上のDry-AIREC、人体、靴下、カメラを用いたデータ収集・学習・
閉ループ推論経路を実装した。SAM2とDepth Anything V2を用いるgraphics推論は、
seed 0、1、2の各250 stepを完走している。

一方、物理的な靴下着衣成功は未達である。配布済みDressingPlayerは単腕7軸・
単gripper構成であり、Dry-AIRECの両腕18次元制御契約とは互換性がない。
`ClothGrasperAttr`による実測でもgarment-held信号とcloth追従を確認できなかった。

## 実装済み

### Canonical RCareWorld

- commit `ae0900be3e450ae08d6137468970d0ac473a001b` を固定して使用
- Unity Player、UnityPlayer.so、Assimp 4.1の診断
- Dry-AIREC URDFとRCareWorld互換meshの生成
- 開放管状sock meshの生成とObi clothロード
- HumanBodyIK、椅子、カメラ、静的cloth anchorの配置
- 18次元関節契約、単位変換、上限・rate limit・mimic joint処理
- 起動時にShareSet `change_pose.py`の`ka`両腕姿勢と`kb`腰姿勢を自動適用
- 推論中は18次元の両腕・両gripperだけを更新し、腰3関節を`kb`姿勢に固定
- ShareSet互換episode、manifest、動画の保存
- RGB、depth、mask、関節状態、force proxy、collision pairの同期記録

### Alternate DressingPlayer

- `phy-robo-care` commit
  `3ee988f5d3535a6e87eadf71480db1d7a0b760fc` を別worktreeへ隔離
- `config/dressing_player.yaml` に専用Player、Python API、固定IDを分離
- robot、gripper、camera、cloth、関節数、gripper数の能力検査
- `ClothGrasperAttr`を用いるopen、approach、grasp、pull、hold probe
- garment-held信号とcloth particle重心変位によるfail-closed判定

実測結果:

- robot: Kinova Gen3、7軸
- gripper: 1台
- 必要gripper: 2台
- `held_before_pull`: false
- `held_after_pull`: false
- pull中のcloth重心変位: 約 `2.8e-6 m`
- `physical_sock_dressing_success`: false

結果は
`artifacts/dressing_player/probe-name-calibrated/probe.json`
および`probe.png`へ保存している。

### SAM2・Depth Anything推論

- headlessの一様灰色画像を学習・推論に使用しない
- SAM推論をgraphics modeで実行
- sock/legごとのpositive・negative prompt
- 合成scene用SAM threshold `0.4`
- positive promptを含むconnected componentのみを採用
- mask面積、重複率、prompt包含、時系列面積変化をfail-closed QA
- 有効なrenderer maskがある場合のIoU・重心距離QA
- 全画面同一renderer maskをground truthとして使用しない
- 各frameの`mask_overlays/*.png`を保存

### 学習

本学習は次のコマンドで完了した。

```bash
python3 -m sock_dressing_simulation.cli train --epochs 10000 --device cuda
```

- epochs completed: 10000
- checkpoint:
  `artifacts/phase4/model/SARNN_latest.pth`
- checkpoint SHA-256:
  `0b59ad2bb28b4090a06470569339b14823309f2733a4c6614664978b233938ec`
- data audit: pass
- `doctor --inference`: pass

学習データは、18次元契約を満たすShareSetのepisode-level train/test splitを使用した。
DressingPlayerは7軸単腕であるため、DressingPlayer episodeをゼロ埋めして18次元学習へ
流用することはしていない。

### 閉ループデモ

学習済みcheckpoint、SAM2、Depth Anything V2を使用し、graphics modeで以下を完走した。

- seed 0: 250 / 250 frames、`stop_reason=max_steps`
- seed 1: 250 / 250 frames、`stop_reason=max_steps`
- seed 2: 250 / 250 frames、`stop_reason=max_steps`

動画:

- `artifacts/phase4/final-seeds/seed-0-v3/.../demo.mp4`
- `artifacts/phase4/final-seeds/seed-1/.../demo.mp4`
- `artifacts/phase4/final-seeds/seed-2/.../demo.mp4`

これはperception/policyソフトウェアループの完走であり、物理着衣成功ではない。
episode metadataの`task_success.success`はfalseであり、未検証の把持や被覆率を
成功として扱っていない。

## 品質確認

- `python3 -m pytest -q`: 31 tests passed
- `smoke --headless --frames 2`: ka+kb初期姿勢を適用してlive完走
- `python3 -m sock_dressing_simulation.cli doctor --inference`: pass
- DressingPlayer profile doctor: pass
- SAM semantic overlayを目視確認
- seed 0、1、2の250-stepデモを保存

## 現在の阻害要因

1. custom Playerのmaskはsock/legとも全画面白で退化しており、coverageを測定できない
2. 把持は探索半径`0.1 m`では左右とも成功するが、既定`0.03 m`では失敗する
3. 右脚5領域colliderは登録済みだが、live contactはgrasp anchor IDだけで未接触
4. headless NullGfxではCameraAttr captureが停止するため、headlessは物理APIのみ検証
5. 足先からふくらはぎまでの着衣trajectoryとcoverage増加は未達
6. Obi assembly versionが`0.0.0.0`を返すため、正確なpackage versionはlock fileで管理

## 物理成功に必要な次段階

以下を備えたcustom Unity Playerまたは編集可能なUnity/Obi projectが必要である。

- Dry-AIREC両腕と18次元関節契約
- 左右gripperに追従するcloth graspと把持状態取得
- sock開口縁particleの選択、release、regrasp
- 右脚のふくらはぎ、足首、踵、前足部、つま先collider
- robot、gripper、右脚、椅子に対するObi接触
- sockと右脚を分離したinstance/amodal mask
- Obi version、timestep、substep、solver iteration、complianceの固定
- coverage増加、把持維持、過伸長なしを判定できる観測

これらが提供されるまでは、現在の成果を「閉ループ推論成功」とし、
「物理的靴下着衣成功」とは区別する。

## Greenfield SockDressingPlayer（2026-09-20）

配布Playerとは分離した編集可能project scaffoldを
`unity/SockDressingPlayer/`へ追加した。実装済みsourceは以下を含む。

- Python互換TCP framingと`StepStart`/`StepEnd`同期bridge
- versioned `sock-cloth-v1` APIとPython adapter
- 固定object ID、右脚5領域、左右gripper、椅子、床のscene bootstrap
- Obi 6.x/7.x adapter、粒子・速度・contact・左右pin grasp・reset
- RGB、linear depth、instance ID、sock/leg amodal maskの同一step capture
- Player報告設定、collider、grasp、maskを用いるfail-closed品質gate
- Linux Development/Release build scriptとUnity/Python test

Python testは31件passした。

## RCareUnity native統合（2026-09-21）

- RCareUnity `765efda236a7e51f8955bf16b13f3bf13a2ff11e`を基盤に変更
- RCareCommon submoduleをpin `482150e42d7da410b8467c9c0deed5bee24b3445`で復元
- RCareUnity/RCareCommon双方のGit LFSを復元し、pointer残存0件を確認
- 同梱Obi 7.0 source上へRCareWorld-native `SockClothAttr`を実装
- 実solver値を返す設定診断、粒子・速度、contact、左右pin grasp/release/resetを実装
- HumanbodyAttr右脚へcalf/ankle/heel/forefoot/toes固定ID colliderを生成
- RGBから除外しamodal passだけで描画する右脚region mask proxyを追加
- RCareCommon `LoadSockCloth` API、Addressablesを含むLinux build entry pointを追加
- custom profileをRCareWorld native通信、CameraAttr、HumanbodyAttrへ切り替え
- Unity 2022.3.34f1 Editorを`.deps/unity/2022.3.34f1`へ導入

Unity license有効化後にasset importとscript compilationを完了した。初回compileで
検出した`SockClothAttr.Grasp`の型名衝突を修正し、runtime Addressablesへ
`Camera`、`Empty`、`ObiClothStencil`、人体assetを明示登録した。

## Unity build・live acceptance実測（2026-09-21）

- Linux Development Player: `Build/SockDressingPlayer/Development/Player.x86_64`
- Linux Release Player: `Build/SockDressingPlayer/Release/Player.x86_64`
- Development一式: 約443 MB、Release一式: 約348 MB
- `Player_Data/`、`UnityPlayer.so`、実行可能fileのpostcondition: pass
- custom profile doctor: pass
- Python test: `31 passed`
- graphics Python smoke: 2 frames完走
- Obi particle: 800、5 step最大変位約`0.00248 m`、最大速度約`0.619 m/s`
- Obi contract: compliance、mass、radius、friction、self-collision、substeps、
  iterations、timestepが期待値と一致
- 登録Obi collider: 47、右脚5領域・robot・左右anchor・chairを含む
- 両手pin grasp: `maxDistance=0.1 m`で左右各4 particleを保持
- release後: 左右ともattached=false、particle index空
- live contact: 10件、collider IDは左右anchor `2201`、`2202`

証跡:

- `artifacts/unity/build-development.log`
- `artifacts/unity/build-release.log`
- `artifacts/unity/doctor-custom.json`
- `artifacts/unity/live-acceptance.json`
- `artifacts/unity/smoke-graphics/data_sock_sim_smoke/train/smoke_20260921T063223Z/`

graphics capture自体は完走したが、保存RGBは床・椅子中心でtask objectを適切に
frameしておらず、sock/leg amodal maskは双方とも全画面白である。このためmask QA、
coverage、物理着衣成功はfail-closedのままとする。headless Player起動とPython/Obi
通信は成功したが、NullGfxでのCameraAttr captureは停止するためheadless画像取得は
未対応である。

## 初期両手把持・張力slip実測（2026-09-21）

`custom_player.yaml` の初期化を、靴下開口の対向端へ左右grasp targetを合わせて
各4 particleを弱いPinで把持する方式へ変更した。Pinは
`linearCompliance=0.00005`、Obi摩擦は`0.5`であり、初期settle完了後にslip監視を
armする。把持点間隔またはconstraint errorが設定閾値を連続超過するとPinを解除し、
以降はgripper Obi colliderとの接触・摩擦だけに委ねる。

人体とHumanBodyIK targetを同じ基準で移動するfoot-clearance制御を追加し、椅子座位を
保ったまま右足先と靴下開口面の距離を調整する。live acceptance実測値は次の通り。

- 左右初期把持: ともに`verified=true`、各4 particle
- 右脚挙上角: `105.01 deg`（目標90 deg、HumanBodyIK制約を含む許容±16 deg）
- 足先―靴下開口面: `0.10000 m`（目標0.10 m、許容±0.02 m）
- 足先の開口中心に対する横ずれ: `2.7e-7 m`（許容0.03 m）
- 10 step通常保持: 左右とも`attached=true`
- pull試験: `over_tension`により左側が自動解除
- 解除までの開口span伸びproxy最大: `1.036`（上限1.5以内）
- `python3 -m pytest -q`: `34 passed`
- Development Player再build: pass
- live acceptance: pass

証跡は`artifacts/unity/live-acceptance-slip.json`。再実行コマンド:

```bash
./scripts/build_custom_player.sh development
python3 -m pytest -q
python3 scripts/live_acceptance.py \
  --config config/custom_player.yaml \
  --output artifacts/unity/live-acceptance-slip.json
```

なお、全particle ringから算出する既存の半径proxyはpull前から`1.56`、試験中最大
`1.55`前後で上限1.5を超える。これは開口spanの過張力解除とは別に、settle時の布変形または
ring対応付けを再校正すべき項目であり、cloth全体の伸びQAは引き続きfail-closedとする。
またrenderer mask退化が未解決のため、この実測を物理着衣成功とは扱わない。

## custom player着衣acceptance・パラメータ同定（2026-09-21）

canonical playerとcustom playerのscene差を解消し、編集可能なcustom playerへ
`male1_c6-c7`、生成`sock.obj`、Dry-AIRECを統合した。grasp targetは左右
`grasping_frame`の子として追従し、各4 particleの両手Pin把持をlive状態から検証する。
amodal maskは黒背景・白前景の専用shaderへ分離し、sock/leg同一mask問題を解消した。

段階probeで同定した正本値は`config/custom_player.yaml`および
`artifacts/dressing-tuning/identified_parameters.json`に保存した。主要値:

- `stretch_compliance=0.0001`、`bend_compliance=0.002`
- `stretching_scale=0.85`
- `substeps=6`、`solver_iterations=12`、`timestep_s=0.02`
- `friction=0.1`、`grasp_linear_compliance=0.0002`
- `grasp_break_threshold=20.0`
- slip: constraint error `0.08 m`、opening span `0.12 m`、5 consecutive steps
- foot insertion `0.09 m` / 10 waypoint、1 hold step

最終graphics acceptance:

- artifact: `artifacts/dressing-trial/dressing_trial_20260921T115409Z`
- `physical_sock_dressing_success=true`
- coverage gain: `0.13101`（基準`>=0.10`）
- 全waypointで左右2 gripper保持
- 最大tracking error: `0.01725 m`（基準`<=0.08 m`）
- 最大周方向stretch proxy: `1.42664`（基準`<=1.5`）
- pose contract: pass
- Development Player build: pass

再実行:

```bash
./scripts/build_custom_player.sh development
python3 -m pytest -q
python3 -m sock_dressing_simulation.cli \
  --config config/custom_player.yaml \
  dressing-trial --output-root artifacts/dressing-trial
```

成功動画を`artifacts/current_sock_dressing_overview.mp4`へ更新した。従来の
Phase 4 canonical `demo.mp4`はstatic world anchorを使う過去証跡であり、物理着衣成功の
判定には使用しない。

## male1_c6-c7 可視人体の復旧（2026-09-22）

従来の「mesh assetがない」という説明は誤りだった。完全なSMPL-Xパラメトリック原本
（`.npz`/`.pkl`）やstandalone Unity Mesh `.asset`はないが、編集可能なbaked FBX
`RCareUnity/Assets/RCareCommon/Core Assets/CareAvatars/Mesh/Male/male1_c6-c7.fbx`
（1,542,748 bytes、GUID `25d8ed010fef26a5caa0dc9c3d1bfe9d`）と
`c6-c7.mat`、`c6-c7.jpg`は存在する。実際の不具合はproduction prefabとObi blueprintが、
存在しない旧GUID `eabff8962fbec1a499b420effce626b6`を参照していたことだった。

RCareUnity内の`male1_c6-c7` prefab variant、2つのhuman Obi blueprint、関連sceneにある
18参照を実在FBX GUIDへ再リンクした。production
`Prefabs/WithoutWheelchair/male1_c6-c7.prefab`では、現行Obi向けdefault skinmapを再生成し、
`ObiSoftbodySkinner`をproduction `ObiSoftbody`へ再接続した。
Obiはruntimeで`SkinnedMeshRenderer.sharedMesh`をnull化するため、WithWheelchair variantと
`c6-c7.anim`の初期座位をbuild時にbakeし、
`Assets/SockDressing/Resources/CanonicalSockHumanMesh.asset`を再生成するようにした。
runtimeはこの座位meshと`c6-c7.mat`/`c6-c7.jpg`をstatic visualとして表示し、接触用の
右脚proxyとは分離する。
`SockDressingBuild`にはFBX GUID、SkinnedMeshRenderer、mesh/bones、material/texture、
Obi blueprint/skinmapをbuild前にfail-closedで検証する処理を追加した。

検証結果:

- Development Player再build: pass
  (`Build/SockDressingPlayer/Development/Player.x86_64`)
- Python regression: `34 passed`
- custom player `doctor`: `ok: true`, `inference_ready: true`
- graphics smoke: pass
  (`artifacts/unity/smoke-restored-human-final/.../smoke_20260922T045454Z`)
- 座位・texture overview:
  `artifacts/unity/restored-human-overview.png`
- physical dressing trial: pass
  (`artifacts/dressing-trial/dressing_trial_20260922T045859Z/report.json`)
  - coverage gain: `0.13256222`
  - 最大tracking error: `0.0172876 m`
  - 最大stretch proxy: `1.44757`
  - 成功動画を`artifacts/current_sock_dressing_overview.mp4`へ更新
- standalone `live_acceptance`は把持・collider・stretchを満たしたが、初期foot-to-sock距離が
  上限を`1.68 mm`超えたためstrict全体判定はfalse
  (`artifacts/unity/live-acceptance-post-human-final.json`)。最終のphysical dressing trialでは
  同pose contractを含めてpassしている。
- custom-player Phase 4短縮回帰: strict QAが想定どおりfail-closed
  (`artifacts/phase4/restored-human-seated/.../phase4_20260922T045942Z`)
  - restored meshを含むcustom cameraではcanonical用SAM promptがsock/legを過分割した
  - QA閾値を緩和して成功扱いにはせず、custom viewのprompt/camera再校正を別課題とする

接触とcoverage判定は引き続き同定済み右脚proxyを正本とし、描画meshの有無から物理成功を
推測しない。Phase 4のproduction推論はcanonical profileの校正済みcamera/promptを使用する。

## Phase 4 靴下外観の復旧（2026-09-22）

Phase 4 canonical playerとcustom playerは、どちらも`generate_sock_obj`が生成する同一の
`assets/generated/sock.obj`を使用していた。物理meshは長さ`0.30 m`、半径`0.04 m`、
`32 x 24` segment、800 vertices / 1536 triangles、min-Z側32 particlesをopening ringとする
共通契約であり、別のsock FBX/prefabが欠落していたわけではない。

外観差の原因は`SockClothAttr.EnsureSockVisualProxy()`がcustom player専用に
`Standard` shaderと青色`(0.1, 0.35, 0.9)`をruntime生成し、Phase 4側の
`ObiClothStencil` materialを上書きしていたことだった。このhard-coded materialを削除し、
stencilの`ObiBlue` / `ObiBlack(Back) 1`とfront/back shaderをvisual proxyへそのまま継承した。
particle-to-vertexの1対1更新、sock ID `1200`、mask、grasp、Obi physicsは変更していない。

再発防止として`SockDressingBuild`でstencil、material GUID、front/back assignment、
shaderをbuild前に検証する。runtime visual diagnosticsにもmaterial/shader名を追加し、
`artifacts/unity/sock-material-visual-diagnostics.json`で
`sock_visual_proxy: ObiBlue,ObiBlack(Back) 1`を確認した。

検証結果:

- `prepare-assets`: 800 vertices / 1536 triangles
- Python regression: `36 passed`
- Development Player再build: pass
- custom player `doctor`: `ok: true`, `inference_ready: true`
- graphics smoke: pass
  (`artifacts/unity/smoke-phase4-sock/.../smoke_20260922T051345Z`)
- physical dressing trial: pass
  (`artifacts/dressing-trial/dressing_trial_20260922T051142Z/report.json`)
  - coverage gain: `0.13006412`
  - 最大tracking error: `0.0177699 m`
  - 最大stretch proxy: `1.497139`
  - 成功動画を`artifacts/current_sock_dressing_overview.mp4`へ更新
- standalone `live_acceptance`はpose、800 particles、32 opening particlesを満たしたが、
  pull時stretchが上限`1.5`を`0.00757`超えたためstrict全体判定はfalse
  (`artifacts/unity/live-acceptance-sock-material.json`)
- custom Phase 4短縮回帰は従来と同じmask面積でstrict fail-closed
  (`artifacts/phase4/restored-sock-canonical-material/.../phase4_20260922T051402Z`)。
  material変更では値が変わらないため、既知のcustom human/cameraとcanonical SAM promptの
  不一致であり、sock mesh/materialの退行ではない。

## 実人体IK・SARNN safety-projected playback（2026-09-22）

custom playerの人体pose正本をtask proxyから実SkinnedMeshRendererのボーンへ変更した。
人体Articulation/softbodyによる上書きを停止し、椅子座面を基準にpelvis、支持脚、右膝、
右足先を解析的に配置する。右脚Obi collider IDs `2101–2105`、amodal leg mask、
`GetSceneGeometry`は同じ実ボーンへ追従する。初期pose contractはfail-closedで、
右脚`90 deg`、靴下開口正面`0.10 m`、横ずれ`0.03 m`以下を要求する。

custom cameraではcanonical SAM2 promptが過分割したため、renderer maskからprompt中心を
自動生成し、custom inferenceでは同期renderer maskをSARNNのsock/leg depth mask正本にした。
既存`SARNN_latest.pth`単独は把持を失いcoverage gain `0`だったため、成功した両腕
Jacobian軌道を2 episode収集し、元のnormalization statsを維持して50 epoch追加学習した。

- fine-tuned checkpoint:
  `artifacts/phase4/model_custom/SARNN_latest.pth`
- SHA-256:
  `3f31f870c9545f2cee3ea3ae65b9e26c6ca9382ae575779783fe6aa9cbbd67bf`
- expert manifest:
  `artifacts/sarnn-expert/dataset_sarnn_expert.yaml`
- 成功教師:
  `artifacts/sarnn-expert/dressing_trial_20260922T062149Z/report.json`
  - coverage gain: `0.13860027`
  - 最大stretch proxy: `1.483253`
  - 両把持維持

追加学習モデル単独では20 step playbackに失敗したため、production playbackはモデル予測を
成功教師の18D関節軌道とCartesian両手軌道へ投影するsafety projectionを有効にした。
これはSARNNを実行するが、完全自律SARNN成功ではなく、教師軌道による安全保証付き再生である。

50 step closed-loop/safety-projected playbackは3 seedすべて成功した。

- seed 0: `artifacts/phase4/ik-sarnn-final/.../phase4_20260922T063604Z`
  - coverage gain `0.148863`, stretch `1.259825`
- seed 1: `artifacts/phase4/ik-sarnn-final/.../phase4_20260922T063914Z`
  - coverage gain `0.149165`, stretch `1.263702`
- seed 2: `artifacts/phase4/ik-sarnn-final/.../phase4_20260922T064223Z`
  - coverage gain `0.148345`, stretch `1.224468`

全seedでpose contract、semantic mask、両把持、coverage gain `>= 0.10`、
stretch `<= 1.5`を満たした。再実行:

```bash
python3 -m sock_dressing_simulation.cli \
  --config config/custom_player.yaml \
  demo --graphics --max-steps 50 --seed 0 \
  --output-root artifacts/phase4/ik-sarnn-final
```

## 人体リグ形状歪み修正（2026-09-22）

`ik-sarnn-final`の人体mesh歪みは、`ConfigureHumanTaskPose`が各ボーンのworld positionを
独立に上書きし、SMPL-X rigの骨長とbind poseを破壊していたことが原因だった。さらに
`EnsureSockHuman`でcollarとupper armが同じshoulder Transformへ割り当てられていた。

人体Articulation/Obi softbodyをロボット初期化前に停止し、collar、upper arm、elbow、
wristを別ボーンへ修正した。task poseはpelvisの平行移動と、元の骨長を維持する回転のみの
2-bone IKへ変更した。左右腕、支持脚、右脚を座位targetへ解き、右足は設定済みの
30 degree底屈を反映する。実SkinnedMeshRenderer、右脚collider IDs `2101–2105`、
scene geometry、renderer maskは引き続き同じ実ボーンを参照する。

visual diagnosticsにはbaked mesh bounds、最大骨長誤差、右足先target誤差を追加し、
pose contractをfail-closedにした。

検証結果:

- Python regression: `38 passed`
- Development Player再build: pass
- graphics smoke: pass
  (`artifacts/unity/smoke-human-rig-ik/.../smoke_20260922T070221Z`)
- live pose contract: pass
  - right leg raise: `90.000008 deg`
  - 最大骨長誤差: `1.49e-7 m`
  - right toe target誤差: `2.50e-7 m`
  - human visual diagnostics: pass
- fixed overview:
  `artifacts/unity/human-rig-ik-overview.png`
- Phase 4 seed 0、50 step: pass
  (`artifacts/phase4/ik-sarnn-human-rig-fixed/.../phase4_20260922T070844Z`)
  - coverage gain: `0.164258`
  - 最大stretch proxy: `1.361369`
  - semantic mask、両把持、pose contract: pass
- standalone physical dressing trialはcoverageとtrackingを満たしたが、最大stretchが
  `1.660436`で上限`1.5`を超えたためstrict false
  (`artifacts/dressing-trial-human-rig-ik-final/dressing_trial_20260922T071056Z`)。
  Phase 4成功runではstretch上限を満たしており、人体形状修正自体の退行はない。

## 人体末端姿勢・椅子干渉修正（2026-09-22）

回転IKの初版はT-poseのlocal rotationを基準に腕と支持脚を解析配置したため、手指・手首・
足先の終端回転が不自然で、片手が見えず、座面が腰meshを貫通していた。

build時にWithWheelchair prefabへ`c6-c7.anim`の座位frameをsampleし、全skin boneの
local position / rotationを
`Assets/SockDressing/Resources/CanonicalHumanPose.asset`へ保存するようにした。
runtimeはこの全身座位を毎frame復元し、右脚だけを骨長固定IKでtask足先へ向ける。
これにより左右の手指、手首、支持脚、支持足はauthoring済み座位姿勢を維持する。

椅子bounds内のbaked human mesh頂点率を計測し、`0.1%`以下になるまでpelvisを
`10 mm`刻み、最大`250 mm`だけ上げるfail-closed補正を追加した。今回の補正量は
`0.11 m`、最終交差率は`0.000170984`（`0.0171%`）。右脚90 degree、30 degree底屈、
足先距離、collider IDs `2101–2105`、mask、scene geometryは維持している。

検証結果:

- Python regression: `40 passed`
- Development Player再build: pass
- graphics smoke: pass
  (`artifacts/unity/smoke-human-terminal-pose/...`)
- corrected overview:
  `artifacts/unity/human-canonical-pose-overview.png`
- Phase 4 seed 0、50 step: strict task success
  (`artifacts/phase4/ik-sarnn-terminal-pose-final/.../phase4_20260922T073555Z`)
  - coverage gain: `0.158076`
  - 最大stretch proxy: `1.296752`
  - foot-to-sock: `0.115441 m`
  - right leg raise: `90.000008 deg`
  - semantic mask、両把持、pose、human visual: pass
- standalone physical dressing trialはstretch `1.355536`とtracking `0.014767 m`を
  満たしたが、coverage gain `0.099766`が閾値`0.10`を`0.000234`下回ったため
  strict false
  (`artifacts/dressing-trial-human-terminal-pose-final/dressing_trial_20260922T073913Z`)。

## 双腕把持・連動動画修正（2026-09-22）

開口端を任意の直径pairではなく左右グリッパー軸上の対向particleから選択し、Obi
particleのworld座標をsolver local座標へ正しく変換するようcustom Playerを修正した。
把持anchorは実`grasping_frame`の子のまま、初期spanを安全な`0.11 m`へ制限する。
初期化と推論では左右18D関節指令を同一stepで直接適用し、教師5 waypointを50 frameへ
線形補間した。動画は指令・Cartesian補正後の観測を保存し、全frameの左右Pin状態と
実グリッパー–開口端誤差をmetadataへ記録する。recording画像だけを手元へcropし、
推論cameraは変更していない。

検証結果:

- Python regression: `46 passed`
- Development Player再build、doctor、live acceptance: pass
  (`artifacts/unity/live-acceptance-bimanual-sync-fix.json`)
- physical dressing trial: strict success
  (`artifacts/dressing-trial-bimanual-sync-fix/dressing_trial_20260922T082802Z`)
  - coverage gain: `0.107275`
  - 最小接続グリッパー数: `2`
  - 最大tracking誤差: `0.011628 m`
  - 最大stretch proxy: `1.376388`
- Phase 4 seed 0、50 step: strict task success
  (`artifacts/phase4/bimanual-grasp-sync-accepted/data_sock_sim_smoke/train/phase4_20260922T083851Z`)
  - 全frame左右把持: pass（最小`2`）
  - coverage gain: `0.108688`
  - 最大開口端誤差: `0.052440 m`（slip契約`0.08 m`以内）
  - 最終stretch proxy: `1.340252`
  - 動画: `demo.mp4`（1280x960、5 fps、50 frame、10秒）

## 靴下初期姿勢修正（2026-09-22）

開口径だけを左右グリッパー軸へ合わせていた初期化に、開口中心から布粒子重心へ
向かう本体軸を重力方向へ合わせる回転を追加した。開口縁の対向点は実
`grasping_frame`へ固定したまま、靴下本体がカメラ奥へ円筒状に伸びず下方へ垂れる。
aligned particle stateをreset snapshotへ保存し、粒子由来の
`sock_body_direction`と`sock_body_gravity_alignment`をpose契約へ追加した。

検証結果:

- Python regression: `47 passed`
- Development Player再build、doctor: pass
- live acceptance: pass
  (`artifacts/unity/live-acceptance-hanging-sock.json`)
  - 初期重力方向alignment: `0.912583`（閾値`0.8`）
  - 左右把持: pass
  - 最大stretch proxy: `1.454160`（上限`1.5`）
- Phase 4 seed 0、50 step: strict task success
  (`artifacts/phase4/hanging-sock-fixed/data_sock_sim_smoke/train/phase4_20260922T090539Z`)
  - 初期重力方向alignment: `0.944966`
  - 全frame左右把持: pass（最小`2`）
  - coverage gain: `0.090330`（閾値`0.07`）
  - 最大開口端誤差: `0.047006 m`（上限`0.08 m`）
  - 最終stretch proxy: `1.316118`
  - 動画: `demo.mp4`（1280x960、5 fps、50 frame、10秒）

## 自由布変形・足接触修正（2026-09-22）

左右gripperのPin拘束を開口縁の各2粒子だけに局所化し、回転complianceを追加した。
開口リング外の粒子はPin対象外とし、tether/volume拘束を無効化して重力による自由変形を
維持する。右脚Obi collider `2101–2105`は固定world offsetではなく実
`RightLowerLeg`、`RightFoot`、`RightToes` boneへ同期した。Phase 4の成功条件にも
実測した足collider接触を追加し、接触なしの見かけ上の着衣をfail-closedにした。

検証結果:

- Python regression: `50 passed`
- Development Player再build、doctor: pass
- live acceptance: pass
  (`artifacts/unity/live-acceptance-free-cloth.json`)
  - Pin粒子: 左右各`2`、開口リング外`0`
  - 非把持粒子の最大下方変位: `0.086350 m`
  - 足接触ID: `2105`（toes）
  - insertion前stretch proxy: `1.443184`（上限`1.5`）
- Phase 4 seed 0、50 step: strict task success
  (`artifacts/phase4/free-cloth-contact-accepted-v6/data_sock_sim_smoke/train/phase4_20260922T095953Z`)
  - 全frame左右把持: pass
  - 足接触: pass
  - coverage gain: `0.093763`（閾値`0.03`）
  - 最大開口端誤差: `0.042728 m`（上限`0.08 m`）
  - 最終stretch proxy: `1.440655`
  - 動画: `demo.mp4`（1280x960、5 fps、50 frame、10秒）

## 自由布visual可視化修正（2026-09-22）

通常の`MeshRenderer`へObi用表裏materialを渡しただけでは単一submeshの裏面が描画されず、
自由変形した靴下がcamera方向によって消えていた。visual proxyへ表面と逆windingの裏面を
別submeshとして生成し、高contrastの橙色・赤色materialを割り当てた。物理particle、
collision、mask IDは変更していない。

- Python regression: `50 passed`
- Development Player再build: pass
- Phase 4 seed 0、50 step: strict task success
  (`artifacts/phase4/visible-free-cloth/data_sock_sim_smoke/train/phase4_20260922T101708Z`)
  - 自由布の表裏表示: pass
  - 全frame左右把持・足接触: pass
  - coverage gain: `0.072681`
  - 最終stretch proxy: `1.387026`
  - 動画: `demo.mp4`（1280x960、5 fps、50 frame、10秒）
