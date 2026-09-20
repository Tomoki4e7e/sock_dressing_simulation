# RCareWorld 靴下着衣シミュレーション実装進捗

最終更新: 2026-09-18

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

- `python3 -m pytest -q`: 26 tests passed
- `smoke --headless --frames 2`: ka+kb初期姿勢を適用してlive完走
- `python3 -m sock_dressing_simulation.cli doctor --inference`: pass
- DressingPlayer profile doctor: pass
- SAM semantic overlayを目視確認
- seed 0、1、2の250-stepデモを保存

## 現在の阻害要因

1. canonical Playerのcloth anchorはworld固定で、gripper追従が未検証
2. alternate DressingPlayerは単腕・単gripperで、両手によるsock開口保持が不可能
3. DressingPlayerのgarment-held信号がfalseで、cloth追従を実測できない
4. 配布PlayerのUnity projectがなく、scene、collider、Obi設定を編集できない
5. 正確な右足collider IDと非退化renderer maskがない
6. coverage gainを画像から信頼して測定できず、物理成功条件を満たせない

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
