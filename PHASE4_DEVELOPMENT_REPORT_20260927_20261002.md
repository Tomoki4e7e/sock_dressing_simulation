# Phase 4 靴下着衣シミュレーション 開発変遷レポート

## 1. 目的と調査範囲

本レポートは、2026年9月27日から10月2日までの Phase 4 開発について、**どの部分へどのような変更を加え、その結果として何が改善し、何が悪化または未解決のまま残ったか**を整理したものである。

評価上の最新試行は、次の指定試行とする。

- 試行: `human-chair-down12cm-plantar30deg-toe-clearance-autonomous-final-100`
- episode: `phase4_20261002T125245Z`
- モード: `airec-closed-loop`
- 推論: 100/100 frame、`stop_reason=max_steps`
- 方策: 実世界データ学習済み SARNN
- 知覚: SAM2 + Depth Anything V2

このepisodeより後に作成された同日中の plantarflexion / foot-passage 試行は、ディレクトリ上に存在していても本レポートの比較対象には含めない。Gitの現行working treeにあるtoe-clearance判定は最新試行に使用されている一方、後続のgripper-foot passage探索用コードと設定は未コミットであるため、「後続開発」としてのみ区別して記載する。

根拠は、Git履歴、`RCareWorld_PROGRESS.md`、設定YAMLの継承関係、各試行の`metadata.json`、探索結果JSON、テストを相互参照した。単体テストが示す「実装契約の成立」と、Unity上の自律試行が示す「実際の着衣性能」は分けて評価する。

## 2. 結論

この期間の開発は、次の順で進んだ。

1. 靴下先端を方策開始前に重力で垂下させる。
2. 開口角度を60〜270度で探索し、方策から見える開口姿勢を探す。
3. 270度開口を基準に、直線下垂、開口部自己貫通防止、+180度のpre-inference drapeを加える。
4. 人体・椅子・足首姿勢、把持連動、記録pose、カメラ、SAM promptを調整する。
5. 開口幅保持と人体位置探索を行い、最後に椅子を12 cm下げ、足首を30度底屈してtoe clearanceを確保する。

主要な改善は、**coverage gainが9/27の約0.085から最終試行の約0.644まで増えたこと**、開口幅・把持・足接触・toe clearanceを100 frameにわたり維持できたこと、カメラ整列や初期化手順が再現可能な設定とテストへ分離されたことである。

一方、**物理的着衣成功は未達**である。最終試行でも`task_success.success=false`であり、全期間の主要な自律試行にも成功例はない。最終試行はcoverage指標では最良だったが、cloth-QAは0/100 frameへ悪化し、stretch、containment、開口部貫通、cloth-foot penetration、初期pose契約が失敗した。したがって、現在の到達点は「方策が足方向へ布を大きく移動させる閉ループ動作」ではあるが、「無理な伸長や貫通なしに靴下を足へ通した状態」ではない。

## 3. 代表試行の定量比較

`cloth-QA合格frame`は、各`metadata.json`の`cloth_quality_by_frame[].ok`を再集計した値である。`coverage gain`はマスク由来の進行proxyであり、containmentや物理成功そのものではない。

| 日付 | 代表試行 | 主な変更 | Coverage gain | cloth-QA合格 | 結果 |
|---|---|---|---:|---:|---|
| 9/27 | robot-side-tip-final-250-v2 | 先端をロボット側へ配置し、推論前垂下gateを導入 | 0.085 | 90/250 | 先端配置は安定、着衣失敗 |
| 9/27 | reverse-180deg-gated-final-250 | 開口を逆向き180度、重力固定frameで垂下 | 0.000 | 116/250 | 布状態は比較的安定、進行なし |
| 9/28 | reverse-120deg-close-final-250 | 120度開口、近接俯瞰カメラ | 0.365 | 174/250 | 角度探索中の最良バランス |
| 9/28 | reverse-270deg-close-final-250 | 270度開口、近接姿勢 | 0.000 | 8/250 | 大幅悪化 |
| 9/28 | reverse-270deg-straight-final-250 | rest bendを除去し直線mesh化 | 0.226 | 14/250 | coverage回復、cloth-QAは低い |
| 9/29 | positive-180-drape-autonomous-final-250 | 270度開口から+180度pre-drape | 0.000 | 77/250 | 初期化手順は成立、進行なし |
| 9/30 | recorded-pose-gripper-coupled-final-250 | 記録pose、把持追従、低速制御 | 0.173 | 129/250 | 布安定性が回復 |
| 10/1 | human-chair-away2cm-down7cm-final-100 | 人体を2 cm離し7 cm下降、fast policy | 0.536 | 1/100 | coverage急増、布品質は悪化 |
| 10/2 | opening-preserved-grasp-final-100 | 開口span/面積/上下順序を連続監視 | 0.413 | 40/100 | 開口保持を改善 |
| 10/2 | down12cm/plantar30deg validation-60 | 椅子12 cm下降、底屈30度 | 0.599 | 32/60 | toe clearance合格 |
| 10/2 | down10cm/plantar25deg validation-60 | 椅子10 cm下降、底屈25度 | 0.534 | 47/60 | 布品質は良いがtoe clearance失敗 |
| 10/2 | **指定最新試行 final-100** | down12 cm、底屈30度、toe-clearance gate | **0.644** | **0/100** | coverage最良、物理QA失敗 |

この比較から、coverageの改善とcloth-QAの改善は同じ方向には進んでいない。特に10/1以降はcoverageが0.5を超える一方で、stretchやbarrier penetrationを含むcloth-QAがほぼ通らず、方策の大きな布移動が物理品質を犠牲にしている。

## 4. 開発の変遷

### 4.1 9/27: 推論前の先端垂下と重力固定座標

#### 変更

- `demo.py`へ`tip_drape_wait`を追加し、先端guidanceを解除した後、布だけを同期stepするようにした。
- 先端下降量、開口より下にあること、左右gripper間のspan、把持、開口rim、stretchを連続stepで確認してから方策と知覚を初期化した。
- 逆向き開口では、開口自身とともに回転する座標ではなく、span軸・world鉛直軸・水平depth軸からなる`gravity_aligned`座標を導入した。
- 60、180、240度の逆向き開口profileを追加した。

#### 改善

- 方策開始前に布の自由垂下を確認でき、初期guidanceの弾性反発を方策入力へ持ち込む問題を減らした。
- reverse-180ではgate時stretch 1.406、reverse-240では1.402で、1.5上限内のまま推論へ移行した。
- reverse-240のgate待機は0.18秒まで短縮され、初期化時間が改善した。

#### 悪化・残課題

- gate通過は着衣成功を意味しなかった。reverse-180はcloth-QA 116/250と比較的安定したがcoverage gainは0であった。
- robot-side-tipはcoverage gain 0.085まで進んだものの、stretch、containment、cuff進行などは未達だった。
- 先端位置や開口角を整えても、学習済み方策が足への挿入軌道を生成できるとは限らないことが明確になった。

### 4.2 9/28: 開口角度探索と270度系への収束

#### 変更

- reverse 90、120、150、180、240、270度のprofileを作成した。
- rest bend、方位角、tip target、damping、strain iterationを角度ごとに調整した。
- 記録用カメラを`[-1.10, 0.85, -0.55]`から`[-0.75, 0.73, -0.10]`へ近づけ、開口と足先を同時に観察しやすくした。
- 270度系では「inward」「straight」「vertical dynamics」を比較し、最終的にrest bendを持たない直線meshと鉛直下垂へ移行した。

#### 改善

- reverse-120 closeはcoverage gain 0.365、cloth-QA 174/250で、角度探索中の最良結果となった。
- 270度straightは、270度closeのcoverage 0から0.226へ回復した。
- vertical dynamicsでは靴下本体の重力整合0.985、distal response合格率0.856を記録し、最大distal follow errorを旧設定の約0.41 mから0.179 mへ低減した。

#### 悪化・残課題

- reverse-270 closeはcoverage 0、cloth-QA 8/250まで悪化した。
- straight化でcoverageは回復したがcloth-QAは14/250に留まり、全体の物理安定性は120度系より低かった。
- 120度系が定量的には良好でも、後段のdrape方向、カメラ、足への挿入経路を重視して270度系へ進んだため、短期指標上の最良候補を採用したわけではない。

### 4.3 9/29: 270度開口の鉛直下垂、自己貫通防止、+180度drape

#### 変更

- `minimum_sock_body_gravity_alignment`とdistal follow/response QAを追加した。
- 開口矩形を描画物から片側衝突barrierへ拡張し、開口rim以外の布粒子が内向き側から面を横切る場合に補正した。
- `drape.py`を追加し、270度開口からgripper span軸まわりへ符号付き180度回転し、settleしてから方策を開始する`pre_inference_drape`を実装した。
- +180度と-180度を比較し、最終重力整合が0.549対0.386だった+180度を採用した。

#### 改善

- barrier単体の80-step検証では開口部貫通0を達成した。
- +180度drapeを方策・SAM2・Depth Anythingの初期化前へ分離し、初期観測姿勢を再現可能にした。
- 回転中の開口・rim target alignmentは1.0を維持した。

#### 悪化・残課題

- barrier付き250-step試行では209 frameで補正が必要であり、distal response fractionは0.368へ悪化した。
- +180度drape後の自律試行はcoverage gain 0、cloth-QA 77/250で、drape手順の成立が方策性能へ直結しなかった。
- barrierは自己貫通を減らす一方、大きな補正が布追従を妨げる可能性があり、後の最終試行でも`opening_body_penetration_ok=false`が残った。

### 4.4 9/30: 人体姿勢、gripper coupling、記録pose、カメラ

#### 変更

- 人体・椅子を5 cm下げ、足首底屈を20度にするprofile群を追加した。
- gripper-coupled profileで`physics_steps_per_action=20`、関節`max_delta=0.005`とし、布がgripperへ追従する時間を増やした。
- 過去の実測人体・椅子poseを復元するrecorded-pose profileを追加し、初期poseの強制を一部緩和した。
- front-right cameraと`inference_camera.mp4`記録を追加し、カメラ位置・スケール・maskを調整した。

#### 改善

- recorded-pose + gripper-coupled試行はcoverage gain 0.173、cloth-QA 129/250で、+180度drape単独の77/250から布安定性が改善した。
- 把持追従とworld座標復元をテストで固定し、設定変更による意図しない開口/drape契約の破壊を防げるようになった。
- overviewとは別に方策が実際に見るカメラ映像を保存でき、知覚失敗の診断性が上がった。

#### 悪化・残課題

- 低速化とphysics step増加は安定性を上げたが、coverageは9/28の120度closeより低かった。
- recorded poseでは`initial_pose_ok=false`が残り、見た目の再現と現行pose契約が一致していなかった。
- skinned-foot-mask / neck-camera pitch 60度の試行はcoverage 0、cloth-QA 0/100となり、カメラ・mask変更が大幅な退行を起こし得ることが分かった。

### 4.5 10/1: 方策観測系、人体・椅子位置探索、rim span緩和

#### 変更

- head camera、frame-zero physics、SAMの`single_centroid` prompt、fast policyを段階的に追加した。
- fast policyでは関節`max_delta`を0.005から0.02へ戻し、動作量を増やした。
- 人体・椅子をロボットから2〜3 cm離し、7〜8 cm下げる探索を行った。
- 開口rimのspan方向だけ最大stretch 1.20、shape stiffness 0.25とし、開口を広く保ちやすくした。
- renderer maskで人体全体ではなく右脚専用ID 2099を使うようにした。

#### 改善

- `away2cm/down7cm`でcoverage gainが0.536まで増え、9/30以前から大きく改善した。
- 位置探索のscreeningでは`away2cm/down8cm`が最良候補となり、後続のカメラ整列とopening-preserved profileの基準になった。
- 専用leg maskとsingle-centroid promptにより、知覚・カメラ調整を設定として独立に比較できるようになった。

#### 悪化・残課題

- `away2cm/down7cm`のcloth-QAは1/100であり、coverage増加と引き換えにstretch等の品質が悪化した。
- fast policyは足方向への移動量を増やした可能性が高いが、最終stretch proxyは2.145となり、上限1.5を大きく超えた。
- 人体を下げるとtoe clearanceは得やすい一方、既存のfoot-to-sock初期距離契約から外れやすくなった。

### 4.6 10/2前半: 左カメラ整列とopening-preserved grasp

#### 変更

- 左See3CAMの位置とyawを自動探索する`search_left_camera_toe_alignment.py`を追加した。
- toe-centeredのyaw -3度からfoot-axis-verticalのyaw -12度へ変更し、脚軸の画像内誤差を低減した。
- opening-preserved profileで次を追加した。
  - grasp target span: 0.115 m
  - slip minimum opening span: 0.092 m
  - rollout minimum opening span: 0.09 m
  - opening ring area retention: 0.95以上
  - 左右graspの上下順序交差を許さない
- `_task_success`へ`continuous_opening_span_ok`を追加した。

#### 改善

- カメラ探索で脚軸/中心の画像誤差を粗探索時の約48.5 pxから23.5 pxへ低減した。
- opening-preservedの自律試行はcoverage gain 0.413、cloth-QA 40/100で、固定軸variantの0/100より安定した。
- 最新試行でもopening spanは0.0920〜0.1150 m、ring area retention最小0.984を保ち、`continuous_opening_span_ok=true`となった。

#### 悪化・残課題

- 画像内整列の改善は、100-stepの着衣成功を保証しなかった。
- 開口を広く保持しても、布本体のstretch、barrier penetration、containmentは別問題として残った。
- QAを厳密化したため、見た目上進んでいる試行でも成功判定はより厳しくなった。ただし、これは退行を隠さないために必要な変更である。

### 4.7 10/2: 椅子12 cm下降・足首30度底屈・toe clearance

#### 変更

最終profileはopening-preserved / foot-axis-vertical基準から、次の3点だけを変更した。

```yaml
scenario:
  foot:
    plantarflexion_degrees: 30.0
scene:
  initial_pose_contract:
    down_m: 0.12
inference:
  minimum_grasp_toe_vertical_clearance_m: 0.0
```

`demo.py`では左右graspの低い方と右toeのworld Y差を`grasp_toe_vertical_clearance_m`として計測し、全frameで0 m以上を要求するgateを追加した。

#### 60-step比較

- down12 cm / plantar30度:
  - coverage gain 0.599
  - cloth-QA 32/60
  - toe clearance合格
- down10 cm / plantar25度:
  - coverage gain 0.534
  - cloth-QA 47/60
  - toe clearance不合格

12 cm / 30度はcoverageとtoe clearanceを改善したが、布品質では10 cm / 25度の方が良かった。この段階ですでに、足先をgripperより下へ逃がす幾何と、布の安定性にトレードオフが見られる。

## 5. 指定最新試行の詳細評価

### 5.1 改善した点

- 100/100 frameを閉ループ推論し、`stop_reason=max_steps`で完走した。
- coverage gainは`0.644450`で、本レポート対象の主要試行中で最大だった。
- `coverage_ok=true`
- `continuous_grasp_ok=true`
- `continuous_opening_span_ok=true`
- 最小opening span: `0.091999 m`（下限0.09 m）
- 最大opening span: `0.115000 m`（上限0.12 m）
- 最小opening ring area retention: `0.984390`（下限0.95）
- 最小grasp-toe vertical clearance: `0.036809 m`（下限0 m）
- `grasp_toe_clearance_ok=true`
- `grasp_vertical_order_preserved=true`
- `foot_contact_ok=true`
- `distal_response_ok=true`、response fraction 1.0
- `tip_drape_wait.passed=true`

これらは、**開口を保持したまま、gripperをtoeより上に保ち、布を足へ接触させつつ方策を完走する**ところまで到達したことを示す。

### 5.2 悪化または失敗した点

`task_success.success=false`で、失敗gateは次の8項目だった。

1. `initial_pose_ok`
2. `final_stretch_ok`
3. `continuous_stretch_ok`
4. `opening_body_penetration_ok`
5. `final_surface_containment_ok`
6. `final_section_containment_ok`
7. `cuff_progress_ok`
8. `cloth_foot_penetration_ok`

特に重要な値は次のとおりである。

- cloth-QA合格: **0/100 frame**
- foot-to-sock距離: `0.200729 m`（目標`0.11 ± 0.06 m`の上限0.17 mを超過）
- 最終circumferential stretch proxy: `1.590240`（上限1.5）
- 最終surface containment ratio: `0.0`（下限0.9）
- 最大opening-body penetration: `0.009066 m`（上限0.0005 m）
- 最大cloth-foot penetration: metadata上`1.100007 m`（上限0.002 m）
- 最大cuff reverse: `0.022938 m`（上限0.002 m）
- 最大cuff beyond distal toe: `0.076943 m`（上限0.002 m）
- 最大distal follow error: `0.333634 m`

coverage gainが高いにもかかわらずcontainmentが0であるため、coverageは布が足画像領域へ大きく重なったことを捉えていても、靴下内部へ足が正しく収まったことを示していない。さらにcloth-foot penetration値は他指標と比べて極端に大きく、実際の深い貫通に加えて、collider/単位/集計方法の再確認も必要である。

### 5.3 60-step validationから100-step finalへの退行

同じdown12 cm / plantar30度の60-step validationではcloth-QAが32/60だったが、100-step finalでは0/100になった。coverageは0.599から0.644へ増えたため、後半の追加動作で見かけ上の被覆は伸びた一方、stretchまたはbarrier penetrationが全frameの総合`ok`を失わせたと考えられる。

この結果は、単に「100 stepへ延長すれば着衣が進む」という仮説を支持しない。むしろ、**初期〜中盤の良好な状態を検出して停止する条件**、またはstretch/penetrationが増える前に方策を抑制する安全投影が必要である。

## 6. 実装面で得られたもの

### 再現性

- 開口角、rest shape、drape、人体位置、足首角度、カメラ、grasp spanをYAML継承で分離した。
- 各profileが変更すべき項目だけを変えていることをテストで固定した。
- `demo.mp4`に加え、方策視点の`inference_camera.mp4`を保存した。
- `metadata.json`へ把持、開口、stretch、barrier、collision、containmentをframe単位で記録した。

### Fail-closed評価

- 高coverageだけで成功とせず、stretch、把持、足接触、containment、penetration、cuff進行を全て要求した。
- 期間中の主要試行はすべて`task_success.success=false`であり、改善途中の結果を物理着衣成功として扱っていない。
- 10/2のtoe-clearance gateにより、gripperが足先の下へ潜る軌道を独立に検出できるようになった。

## 7. 現在のボトルネックと次の優先項目

1. **coverageとcontainmentの乖離を解消する**  
   coverageだけでなく、section containmentとsurface containmentが増える軌道を直接評価・制御する必要がある。

2. **stretch超過前に停止または行動を制限する**  
   100-step finalの後半進行が物理品質を悪化させている。frameごとのstretchとbarrier penetrationを使ったearly stop、action scaling、safety projectionを比較するべきである。

3. **opening barrierを再調整する**  
   barrierは開口自己貫通を抑えるが、最大9.1 mmの違反と大量補正が残る。補正量、shell厚、solver iteration、方策動作との競合を分離して調べる必要がある。

4. **人体下降後のinitial pose契約を再定義する**  
   down12 cmはtoe clearanceを改善するが、既存のfoot-to-sock距離契約を外れる。座標変更に合わせて正しい目標値を再計測し、単に閾値を緩めず物理的意味を保つ必要がある。

5. **cloth-foot penetrationの値を監査する**  
   `1.100007 m`は靴下寸法に対して異常に大きい。collider pair、penetrationの符号・単位、最大値集計を可視化と照合する必要がある。

6. **最良状態で止める条件を導入する**  
   down12/30度validationの32/60 cloth-QAとfinalの0/100の差から、固定step数ではなく、coverage増加・stretch・containment・cuff位置を組み合わせた停止条件が有効と考えられる。

## 8. 最新試行動画

以下の2本は、指定された最新試行`phase4_20261002T125245Z`の記録である。

### 8.1 Overview (`demo.mp4`)

<video controls width="960" src="artifacts/phase4/human-chair-down12cm-plantar30deg-toe-clearance-autonomous-final-100/data_sock_sim_smoke/train/phase4_20261002T125245Z/demo.mp4">
  お使いのMarkdown viewerがvideoタグを表示できない場合は、下のリンクから開いてください。
</video>

[demo.mp4を開く](artifacts/phase4/human-chair-down12cm-plantar30deg-toe-clearance-autonomous-final-100/data_sock_sim_smoke/train/phase4_20261002T125245Z/demo.mp4)

### 8.2 方策入力カメラ (`inference_camera.mp4`)

<video controls width="960" src="artifacts/phase4/human-chair-down12cm-plantar30deg-toe-clearance-autonomous-final-100/data_sock_sim_smoke/train/phase4_20261002T125245Z/inference_camera.mp4">
  お使いのMarkdown viewerがvideoタグを表示できない場合は、下のリンクから開いてください。
</video>

[inference_camera.mp4を開く](artifacts/phase4/human-chair-down12cm-plantar30deg-toe-clearance-autonomous-final-100/data_sock_sim_smoke/train/phase4_20261002T125245Z/inference_camera.mp4)

## 9. 総括

9/27以降の開発により、先端垂下、開口角、pre-inference drape、把持追従、カメラ整列、人体・椅子位置、足首角度、開口保持を個別に調整・検証できる基盤が整った。最終試行は、coverage、開口保持、把持、toe clearanceという幾何・知覚上の指標では期間中の最良水準に達した。

しかし、布の実状態ではstretch、containment、開口barrier、cloth-foot penetrationが失敗している。特に最終試行の「coverage 0.644」と「containment 0、cloth-QA 0/100」の組み合わせは、現在の改善がまだ物理着衣成功へ変換されていないことを明確に示す。

次の段階では、姿勢やカメラの追加探索よりも、**良好な中間状態を壊さない停止・安全制御**と、**containmentを直接増やす方策/軌道**を優先すべきである。

## 10. 主な根拠

- `RCareWorld_PROGRESS.md`
- 2026-09-27〜2026-10-02のGit履歴（`34460319`〜`9d16c0ba`）
- `config/autonomous_real_only_opening_reverse_*.yaml`
- `config/*positive_180_drape*.yaml`
- `config/*human_chair*.yaml`
- `scripts/search_human_chair_offsets.py`
- `scripts/search_left_camera_toe_alignment.py`
- `sock_dressing_simulation/demo.py`
- `sock_dressing_simulation/drape.py`
- `sock_dressing_simulation/environment.py`
- `sock_dressing_simulation/sock_cloth.py`
- `tests/test_demo.py`
- `tests/test_drape.py`
- `tests/test_left_camera_toe_alignment.py`
- `tests/test_sock_cloth.py`
- `artifacts/phase4/**/metadata.json`
- `artifacts/phase4/**/search_summary.json`
- `artifacts/phase4/**/alignment_search.json`
