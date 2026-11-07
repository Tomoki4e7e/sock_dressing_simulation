# Sock Dressing Simulation — Phase 1

This package provides reproducible, testable infrastructure for a Dry-AIREC3
sock-dressing scene in RCareWorld. Phase 1 adds reproducible scenarios,
render-quality gates, real-data proxy calibration, and the existing
`dress_regrasping` feature/QA/audit pipeline. It does not include the Unity
project or claim that cloth contact is physically validated.

## Setup

RCareWorld is pinned to commit
`ae0900be3e450ae08d6137468970d0ac473a001b`:

```bash
cd /home/cm-pc/catkin_ws/src/inoue/sock_dressing_simulation
chmod +x scripts/setup_rcareworld.sh
./scripts/setup_rcareworld.sh
python3 -m pip install -e '.[test]'
```

The script checks out the repository (including its Linux player) under
`.deps/RCareWorld`, marks the player executable, installs its Python client,
and builds the pinned Assimp 4.1 native library under `.deps/assimp-4.1.0`.
RCareWorld's bundled AssimpNet assembly is not ABI-compatible with Ubuntu
20.04's Assimp 5 library; using the system library crashes `LoadCloth`.
Upstream recommends Python 3.10; this package also supports Python 3.8 for
Phase 0. Building the compatibility library requires CMake, Ninja, and a C++
compiler.

The pinned Git tree does not itself contain the complete Linux runtime. This
working deployment has the Unity 2022.3.34f1 `UnityPlayer.so` and Mono runtime
beside the player, and `doctor` verifies them. A fresh checkout must restore the
same matching runtime before live smoke; Python-only checks remain usable
without it.

## Greenfield SockDressingPlayer

The editable replacement project is isolated at
`unity/SockDressingPlayer/`. It implements the `sock-cloth-v1` TCP contract,
stable right-leg/gripper IDs, synchronized RGB/depth/instance/amodal outputs,
Obi contact reporting, and verified bimanual grasp state. It never loads DLLs
from either distributed Player.

```bash
python3 -m sock_dressing_simulation.cli \
  --config config/custom_player.yaml doctor
./scripts/build_custom_player.sh development
python3 -m sock_dressing_simulation.cli \
  --config config/custom_player.yaml smoke --headless
```

Unity `2022.3.34f1` with Linux Build Support and a licensed Obi package are
required. Until both are installed, `doctor` and the build fail closed; a
communication-only stub build may be produced with `SOCK_ALLOW_STUB_BUILD=1`
but is not valid physics or learning data. Import Obi, enable the
`SOCKDRESSING_OBI` scripting define, generate the `sock.obj` cloth blueprint,
and assign `ObiSockBackend` before physical validation.

## Commands

For ankle poses with local toe support, native finger contact in the 30 mm cuff band,
and a gravity-driven free tail, use
`python3 scripts/preview_pose_patterns.py` and
[`config/pose_patterns.yaml`](config/pose_patterns.yaml).
See [pose pattern reconstruction](docs/pose_patterns.md) for the angle conventions,
the contact model, validation results, and saved images. The earlier visual-only
drape is available through `config/pose_patterns_legacy_drape.yaml`.

For right-hand grasp-center calibration and collision searches across the 32
ankle angles, see [grasp geometry](docs/right_grasp_geometry.md). The results
distinguish actual skin geometry from the current convex foot contact proxy.

For smooth, partially dressed initial states A–E from the sock pose workbook,
see [the preinsertion guide](docs/preinsert_normal.md). Generate and inspect all
five states with `python3 scripts/preview_preinsert.py` after building its Player.

For a sock that contracts around the foot and holds through contact friction,
use `python3 scripts/fit_preinsert.py --stage A` and
[`config/preinsert_fitted.yaml`](config/preinsert_fitted.yaml).
See [circumference fitting](docs/circumference_fit.md) for the native build,
gravity validation, saved states, and A–E selection.

```bash
python3 -m sock_dressing_simulation.cli doctor
python3 -m sock_dressing_simulation.cli prepare-assets
python3 -m sock_dressing_simulation.cli smoke --offline
python3 -m sock_dressing_simulation.cli smoke --headless --frames 3
python3 -m sock_dressing_simulation.cli calibrate
python3 -m sock_dressing_simulation.cli scene-preview --angles 0 15 30
python3 -m sock_dressing_simulation.cli collect --graphics --frames 30 --seed 0
python3 -m sock_dressing_simulation.cli train --audit-only
python3 -m sock_dressing_simulation.cli train --epochs 10000 --device cuda
python3 -m sock_dressing_simulation.cli doctor --inference
python3 -m sock_dressing_simulation.cli demo --graphics --max-steps 250
```

`doctor` checks Python, xacro, the sibling `torobo_ros` packages, the product
YAML, the Assimp 4.1 runtime, and the 18-D contract. An editable git
installation allows it to verify the exact RCareWorld commit; wheel/non-git
installs are reported as unverified.

`prepare-assets` expands Dry-AIREC3 `GripperV2.yaml` through xacro, copies every
referenced `package://` mesh into `assets/generated/meshes`, and rewrites the
URDF to portable relative paths. It also creates
`dry_airec3_gripper_v2_rcareworld.urdf`, which avoids a native TriLib/Assimp
crash by replacing Collada visuals with each link's existing STL collision
mesh. The four wheel visuals have no STL counterpart and are omitted from this
runtime-only URDF. Joint, inertia, and collision definitions are unchanged.

The command also creates `assets/generated/sock.obj`. The sock is an
open-ended, fully triangulated tube intended as a starting cloth mesh, not a
calibrated garment.

`smoke --offline` validates only command bounding and episode serialization.
The default `smoke` launches Unity; `--headless` makes the intended mode
explicit. The live smoke loads the bundled `HumanBodyIK.json`, creates a camera,
loads the RCareWorld-compatible Dry-AIREC URDF and cloth, records observations,
and sends a bounded 0.005-rad test motion. Compatibility details are stored in
episode metadata. Unity error logging remains enabled so native failures are
not discarded. Every run gets a new UTC-named episode.

Whenever Dry-AIREC is loaded, both arms are first moved to the ShareSet
`change_pose.py` `na` apron posture (`[47, 0, -2.7, 131, 1, 1, 20]`
degrees for each arm) and the three torso joints are set to its `kb`
posture (`[-45, 100, 0]` degrees). The torso targets are outside the learned
18-D action and are held fixed while inference continues to command only both
arms and both grippers.

`calibrate` reads the existing ShareSet sample without modifying it and writes
`artifacts/phase1/randomization_calibration.json`. It records robust state,
coverage, image foot-axis, and normalized-keypoint proxy ranges. Quantities not
stored in metric form (sock dimensions, Cartesian grasp pose, physical foot
Euler angles) are explicitly reported as unmeasured.

`collect` regenerates the sock from a seeded scenario, applies foot IK and the
bounded initial 18-D grasp pose, settles, and records command→observe-aligned
frames. The Phase 1 camera contract is 1280×960. A live collection fails closed
unless RGB, depth, masks, temporal variation, and exact configured foot
collider IDs pass; the episode is retained for diagnosis.

### Nominal Phase 1 scene

The default scenario reproduces the visual task arrangement with the canonical
player: a four-collider seat, the human facing Dry-AIREC, the right leg
extended with HumanBodyIK target `3`, and a 300 mm × 40 mm open tubular sock at
the toe. The selected right-ankle plantarflexion probe is 30 degrees about the
configured x axis. Dry-AIREC starts from the real-robot `ka` arm posture and
keeps the real-robot `kb` torso posture fixed; this replaces the earlier
position-only IK arm seed.

`scene-preview` launches each requested angle in a fresh Unity process and
writes RGB images plus `preview.json` under `artifacts/phase1/preview`. Probe
angles are not physical measurements and are never selected automatically.

The distributed player does not expose verifiable Dry-AIREC child-link IDs for
cloth attachment. The nominal scene therefore requests two static
opening-edge attachments and records them as unverified; it does not claim
that the grippers physically grasp the cloth. Likewise, requested Obi stretch
and bend compliance are recorded in metadata but cannot be applied through
the pinned Python API. Exact foot collider IDs and non-degenerate masks remain
unavailable, so visual reproduction does not make an episode
`learning_ready`.

Build the full feature package, visualization QA, and residual audit with:

```bash
python3 -m sock_dressing_simulation.cli features \
  --episode artifacts/phase1/data/data_sock_sim_smoke/train/<episode> \
  --data-root artifacts/phase1/data \
  --manifest artifacts/phase1/data/dataset_phase1.yaml \
  --audit-output artifacts/phase1/audit.json
```

The gate requires at least one valid `features_ok` frame, passing
`features/viz/validation.json`, a usable audit entry, and passing live render
quality. Missing feature values remain NaN/False; they are never imputed.

## Joint and data contracts

The 18 columns match the existing ShareSet sock controller order:

1. seven left-arm joints;
2. left gripper command and mimic joint;
3. seven right-arm joints;
4. right gripper command and mimic joint.

`JointMap.bound()` applies position/rate limits and always recomputes each mimic
joint as `0.5 * finger_joint`. Limits are Phase 0 guards and do not replace
workspace, collision, impedance, pause, or retreat safety.

The Unity player reports 29 independently driven articulation values rather
than every non-fixed URDF joint. Runtime mapping is validated against Unity's
reported joint names. Each gripper mimic column shares its source simulator
index and is reconstructed at `0.5 * finger_joint` for the 18-column contract.

`EpisodeWriter` creates frame-aligned, headerless 18-column `angle.csv`,
`torque.csv`, and `external_torque.csv`, plus:

```text
camera_right/*.png
camera_right_mask/sock_mask/*.png
camera_right_mask/leg_mask/*.png
camera_depth/*.png
depth_mask/sock_depth/*.png
depth_mask/leg_depth/*.png
metadata.json
robot_contact.jsonl
```

Depth PNGs are RCareWorld's uint8 linear depth between the configured near/far
planes; masked pixels are zero.
Metadata records the exact column order and schema. This layout is
ShareSet-compatible without importing or modifying ShareSet.

To check a smoke episode with the existing residual-data audit:

```bash
cd ../dress_regrasping
python3 -m residual_flow.audit \
  --data-root ../sock_dressing_simulation/artifacts/data \
  --manifest ../sock_dressing_simulation/artifacts/data/dataset_smoke.yaml \
  --dataset-name data_sock_sim_smoke \
  --output ../sock_dressing_simulation/artifacts/smoke_audit.json
```

Phase 0 smoke does not create `features/`, so the audit may mark the episode
unusable for residual training even when all raw modalities are aligned.

## Phase 4 — AIREC inference demo

`config/real_speed_autonomous40.yaml` approximates the real robot's configured
0.2 s command interval and speed overrides (arms 0.25, grippers 0.5). It uses
URDF velocity limits multiplied by those overrides, URDF acceleration limits,
and an acceleration-limited target update on each 0.02 s fixed step. Native
articulation velocity caps and moderate drive stiffness/damping are applied
and read back to limit coupled-joint transient overshoot. This profile replaces
the per-command `max_delta` limiter during rollout; initialization is unchanged.
At a prismatic hard stop, outward physical velocity is reconciled to zero
without changing joint position. Unconstrained velocity is never clipped by
this reconciliation; raw stop velocities and corrections are retained in the
fixed-step motion log. `SockDressing.Editor.RealSpeedStopChecks.Run` validates
both stops, inward motion, and free motion in Unity.
It does not reproduce a measured robot timing trace or the proprietary online
controller's interpolation. Observation and recording are read-only, so every
inference action advances exactly ten fixed steps.

Build a dedicated Player with `scripts/build_custom_player.sh development
Build/SockDressingPlayer/RealSpeed40`, then run `scripts/replay_foot_contact.py`
with `--config config/real_speed_autonomous40.yaml --autonomous --device cuda --steps 40
--physics-video --surface-mode local --transport local --fixed-grasp-offsets
--post-preparation-material --player-executable
Build/SockDressingPlayer/RealSpeed40/Player.x86_64 --output-root
artifacts/phase4/real-speed-autonomous40 --candidate-name <new-run-id>`.
The existing scene, checkpoint, renderer-mask configuration and task QA remain
in effect. `scripts/validate_real_speed_demo.py <run-directory>` validates live
commands, measured joint speeds, 400 physical steps, the 8 s duration, and all
three decoded MP4 recordings. `joint-motion.jsonl` retains every fixed-step
target, velocity, acceleration, actual joint state and physics clock.

`config/grasp_following_autonomous30.yaml` uses the same low-speed profile for
30 live closed-loop commands (300 fixed steps, 6 s of physics). Native virtual
pins and supplemental material/contact constraints sample the same original
anchor trajectory per substep, including the transformed grasp offsets during
rotation. Virtual anchor handles do not collide; physical jaws and feet retain
their collision geometry. The local rollout does not translate the entire sock.
Gripper-local anchor poses remain fixed during manipulation. The rectangular
cuff/plane preparation templates release at contact arming; passive cloth/hem
constraints and all material/contact bounds remain active. A stationary grasp
settles under gravity before inference, with its own video and diagnostics;
failure to meet the configured speed thresholds stops the run. The 300-step
policy recording starts after settling, with prompts from the settled frame.
The sock keeps vertex-to-vertex self contact and surface-to-collider external
contact as separate routes. Extra self-only point simplices never add duplicate
contacts with feet/jaws; all surface triangles retain their external contacts.
Within this CPU contact profile, convex MeshColliders use a cached outward
convex hull and one nearest contact per cloth simplex/collider, rather than
independent two-sided source-mesh faces. Inside/outside classification uses
the contact material point, including triangles that cross a solid boundary.
The inserted cuff patch excludes only its own jaw collision category; free
cloth, the opposite jaw and foot contacts remain active. Release/disarm restores
the original filters. Contact diagnostics are emitted after collision callbacks
in `grasp-substeps.contacts.jsonl`, with the completed step number and world
positions reconstructed from barycentric coordinates.
Mass, radius, friction, damping and strain limits stay unchanged. Real-mesh
paired probes cover the energy injection caused by surface self contacts;
the native suite also tests an obstacle hitting a triangle interior and routing
removal restoring the ordinary actor representation.
The Obi CPU backend has an optional external pin-target hook; unregistered
pins, non-external solvers, and rigidbody pins retain the standard Obi behavior.

Run `SockDressing.Editor.GraspMotionChecks.Run` in Unity before building a
dedicated Player. The paired tests include rotation, gravity sag and release,
as well as stationary/moving/stop/reversal and anchor-collision cases. Run
`scripts/replay_foot_contact.py --config config/grasp_following_autonomous30.yaml
--autonomous --device cuda --steps 30 --physics-video --foot-views --surface-mode local
--transport local --fixed-grasp-offsets --post-preparation-material
--player-executable <dedicated-player>/Player.x86_64 --output-root <new-output>
--candidate-name <new-run-id>`. Use `--prepared-state <saved-state.json>` to
restore a validated existing preparation without replaying policy commands.
Validate with `scripts/report_grasp_following.py --directory <run-directory>
--regressions <grasp-motion-checks.json>` and
`scripts/validate_real_speed_demo.py <run-directory> --commands 30`.
For the current ankle-total25deg, ten-physics-step profile, add
`--camera-alignment-output config/autonomous30_current_ankle_total25deg_physics10_camera_centered.yaml`
to the autonomous replay command using
`config/autonomous30_current_ankle_total25deg_physics10.yaml`. Calibration samples
the freshly prepared scene without advancing physics, adjusts only mounted
camera yaw/roll, and requires a toe-center error of at most 25 pixels and a
leg-centerline tilt of at most 3 degrees before inference starts. The resulting
config can also be used directly for later runs. `camera_alignment/alignment.json`
retains every trial image, angle measurement, and before/after physics clock.
Run `scripts/report_camera_centered_demo.py <run-directory> --baseline <original-episode>`
from the package directory after completion to validate the 30 live commands,
300 fixed steps, camera alignment and unchanged human/chair/foot pose, and to
save synchronized multiview video and before/after previews beside the run.

For comparison with the real right-camera first frame, use
`scripts/match_real_camera_composition.py --name <new-candidate> --translation <x> <y> <z>
--position <x> <y> <z> --rotation <pitch> <yaw> <roll> --iterations 0`.
This performs fresh preparation, mounts the right head camera, translates the
human root, chair, anchor and fixed toe together, and records one autonomous
command with ten physics steps. Camera trials use read-only captures; resolution,
FOV, arm/ankle poses and cloth preparation remain inherited. With nonzero
iterations the camera fit uses the annotated real grasp-pad centers. Optional
`--shape-after-command` additionally fits foot/sock masks after the live command;
this is an exploratory run and its camera must be validated in a fresh run with
`--iterations 0` before selection. No previous policy commands are replayed.
Run `scripts/report_real_camera_composition.py <candidate-directory>` to compare
centroids, widths, pad positions/tilt and physical-clock continuity. Add
`--publish` to save the selected configuration and image comparison. Unmatched
composition criteria are retained explicitly in `validation.json`; successful
recording validation does not imply every composition criterion passes.

For ankle-to-toe size adjustment with the right camera fixed, run
`python3 -u scripts/search_real_foot_size.py`. It compares at most twelve fresh
preparations, varying only human/chair world Y (±15cm) and Z (±20cm) from the
saved real-composition profile. Each preview stops before policy creation.
Measurements cut the leg mask at the anatomical ankle, use the projected
ankle/toe axis, and retain border truncation and real ankle annotations.
The real target excludes the lower leg; a cropped ankle cannot pass matching.
After search, read `selected_delta_yz_m` from
`artifacts/phase4/real-right-camera-foot-size-20261014/search.json` and run
`python3 -u scripts/adjust_real_foot_size.py --name <new-validation-name>
--dy <selected-Y> --dz <selected-Z>` without `--preview-only`. This performs fresh
preparation and exactly one live command with ten physical steps. Validate with
`python3 scripts/report_real_foot_size.py <validation-directory> --publish`.
The dedicated configuration and comparison retain any width/length residuals;
this workflow does not run a thirty-command demo or move the inference camera.

The grasp-following report requires complete live inference, 1 mm original-anchor accuracy,
cloth/contact QA, toe approach, gravity tests and decoded recordings. Substep
logs include mass, center height, kinetic energy and unpinned vertical speeds.

The `train` command retrains the ShareSet `SAMDAMSARNN` without modifying
ShareSet. It reads the episode-level split from `dataset_sock.yaml`, validates
headerless 18-column `angle.csv` and `torque.csv`, and consumes:

```text
camera_right/*.png
depth_mask/sock_depth/*.png
depth_mask/foot_depth/*.png
```

Real episodes call the second object `foot`; simulation episodes call it
`leg`. The training adapter explicitly maps `foot_depth` to the model's leg
input. Both names are accepted, but missing frames, non-18-column signals, and
frame misalignment fail closed. Training writes `SARNN_latest.pth`,
`data.json`, `parameter.json`, and `loss.json` under
`artifacts/phase4/model`.

The original online AIREC controller uses SAM2 and Depth Anything V2 rather
than renderer object masks. The simulation demo now uses the same path:
RCareWorld RGB → two independent SAM2 trackers → Depth Anything V2 →
sock/leg masked depth → recurrent SAMDAMSARNN → bounded 18-D command.
Configure the official checkpoints under `inference.sam2.checkpoint` and
`inference.depth_anything.checkpoint`, then run:

```bash
python3 -m pip install -e '.[inference,test]'
python3 -m sock_dressing_simulation.cli doctor --inference
python3 -m sock_dressing_simulation.cli train --audit-only
python3 -m sock_dressing_simulation.cli train --epochs 10000 --device cuda
python3 -m sock_dressing_simulation.cli demo --graphics --max-steps 250
```

The checked-in synthetic-scene prompt profile is used when points are omitted.
Graphics mode opens one prompt window for the sock and one for the leg when a
profile has no points.
Left-click positive points, right-click negative points, and press Enter to
finish each object. Explicit positive and negative points override the profile:

```bash
python3 -m sock_dressing_simulation.cli demo --graphics \
  --sock-point 568 494 --sock-negative-point 630 520 \
  --leg-point 630 520 --leg-negative-point 568 494 --max-steps 250
```

The pinned Player selects Unity's Null graphics device in its default
headless mode and may return a constant gray RGB frame. In that deployment the
command intentionally stops at mask QA; use `--graphics` (or an off-screen
graphics backend that produces real RGB) for SAM2 inference. Pixel coordinates
above are examples only. Confirm the generated `prompt_frame.png` and masks
for the exact camera/scene; SAM2 can segment the wrong synthetic object even
when a mask is numerically nontrivial.

SAM logits use a synthetic-scene threshold and are reduced to connected
components containing positive prompts. Per-object area, overlap, prompt
containment, temporal continuity, and usable renderer-mask agreement are
fail-closed checks. Degenerate renderer masks are not accepted as ground
truth. Inference uses the camera at the initial right See3CAM pose, while
`demo.mp4` is recorded independently from the configured fixed overview
camera. Every demo writes a ShareSet-compatible episode plus
`predicted_action.csv`, `applied_action.csv`, checkpoint SHA-256, prompt
points, `mask_overlays/`, per-frame perception QA, coverage observations,
task-success criteria, and a stop reason. Empty, full-frame,
identical, or discontinuously changing SAM masks stop the controller before
another command is sent.

This closes the perception/policy software loop; it does not establish
physical dressing success. The distributed Player still exposes only
unverified static cloth anchors, has no verified robot-following cloth grasp,
and provides no direct contact force. A custom Unity Player with verified
gripper attachment remains necessary for a physically successful pull-up.

### Alternate DressingPlayer capability probe

The binary-only `phy-robo-care` runtime is kept in a separate worktree and
config; its DLLs and Python package are never mixed with the canonical player.

```bash
python3 -m sock_dressing_simulation.cli \
  --config config/dressing_player.yaml doctor
python3 -m sock_dressing_simulation.cli \
  --config config/dressing_player.yaml dressing-probe
```

The probe exercises `ClothGrasperAttr`, records cloth-particle displacement,
and requires the garment-held signal before and after pull-up. The distributed
scene exposes one Kinova arm and one gripper, not the required Dry-AIREC
bimanual 18-D contract; therefore it reports this capability gap and never
labels a one-gripper run as sock-dressing success.

## Environment API and limitations

### Dry-AIREC rigid contact feedback (custom Player)

The rebuilt native custom Player records PhysX rigid-contact impulses on each
robot articulation. It divides each pair impulse by the actual simulated step
duration and projects its force onto ancestor joints. Revolute loads use N m;
prismatic gripper loads use N. `external_torque.csv` now uses this measured
rigid-contact projection when available. `torque.csv` retains drive effort and
`angle.csv` retains the physical joint positions. Drive/inverse-dynamics vectors
are reordered into the same movable-joint order and sampled after physics.

`robot_contact.jsonl` records one row per episode frame: contact partner IDs,
world contact forces and moments, physics step/duration, source and availability,
and the original `external_torque_residual` for comparison. Residuals are not
added to contact loads because that would double count collision response.
The existing `contact_force` field continues to describe Obi cloth contacts;
rigid contacts are separately exposed as `rigid_contact`.

Use `config/rigid_contact_feedback.yaml` for drive-controlled gripper/human
collisions with required contact telemetry. Historical configurations that
explicitly ignore human collisions or teleport rollout joints retain those
settings; diagnostics expose their policy. Older Players retain the labelled
residual fallback, while the required-feedback profile rejects missing telemetry.

Build and validate with:

```bash
bash scripts/build_custom_player.sh development
python3 scripts/validate_rigid_contact_feedback.py \
  --output artifacts/rigid_contact_feedback/validation.json
```

Unity 2022.3 supplies the total impulse for a contact pair
([Unity reference](https://docs.unity.cn/2022.3/Documentation/ScriptReference/Collision-impulse.html)).
For multiple contact points, moments are approximated by placing that resultant
at the manifold centroid. This is a simulator load estimate, not a calibrated
hardware torque sensor. PhysX applies the physical response already; the sensor
does not apply another force. Obi cloth/grasp tension transfer to ArticulationBody
remains a separate, unimplemented path.

### Pinned distributed Player

`SockDressingEnv` imports `pyrcareworld` only when connecting. It loads the
portable URDF and cloth OBJ, accesses configured HumanBodyIK and camera
objects, and requests robot/human/cloth state, RGB, uint8 depth, amodal masks,
and current collision pairs.

At the pinned commit, `HumanbodyAttr` addresses IK targets (indices 2 and 3 are
feet) but does **not** expose exact foot-collider object IDs. Configure
`scene.human_foot_collider_ids` only after inspecting the actual Unity scene.
Without them, the configured whole-human ID is explicitly labeled as a leg
mask proxy.

The current Python API also has no direct contact-force vector query. The
wrapper returns `contact_force: null` and diagnostics state this limitation.
`scene.contact_proxy_ids` can document scene-specific collision/robot-effort
proxies, but those must not be described as measured contact forces.

The distributed player cannot provide verifiable robot `AddObiCollider`
registration from this repository. The Python request is retained, while
metadata reports `robot_obi_collider_verified: false`. With no Unity project,
Phase 1 does not patch the player. Whole-human masks remain diagnostic proxies
and fail the live learning-readiness gate until scene-derived foot IDs exist.

## Tests

```bash
python3 -m pytest -q
```

Tests cover mapping/bounds/mimic behavior, mesh URI vendoring, tube topology,
scenario application, null-renderer rejection, episode output, and a synthetic
feature→visualization→audit regression. They do not launch Unity.

足の剛体接触形状を実皮膚に合わせる変更と検証: [docs/rigid_skin_contact.md](docs/rigid_skin_contact.md)。

低速閉鎖・保持・材料点の移動の計測方法と出力: [docs/grasp_closure_measurement.md](docs/grasp_closure_measurement.md)。

現在の閉鎖評価は摩擦保持を使わない幾何学的な方式です。[設定・結果・再現手順](artifacts/geometric-closure/README.md) を参照してください。

新合格候補5配置の接近・閉鎖の実画像と、1mm・4mm²を幾何タスクの基準として固定した判断は[画像レビュー](artifacts/geometric-closure-photos-clean/README.md)にあります。布の変形を伴う把持成功の検証とは区別しています。

No.3候補8・No.12候補9の旧開始姿勢での事前検証は[腕リーチング検証](artifacts/arm-reaching/README.md)にあります。目標IK不成立と、開始姿勢の衝突形状の重なりを確認した記録です。開始姿勢とモデル固有の除外については以下の更新を参照してください。

更新：[腕IKによる60候補の再選別・開始姿勢の修正・足裏側の追加探索](artifacts/arm-reaching-reselection/README.md)。新しい開始姿勢は `config/arm_reaching_safe_start.yaml` に保存しました。SRDFのモデル固有の除外を反映し、検査対象の手首干渉を解消しています。既存候補と追加した足裏側候補は、今回の条件では採用0件です。

開口から15mmを中心に半径を5mm増やした32初期形状での再探索は[膨らみ追加後の再検証](artifacts/bulged-cuff/README.md)にあります。胴体IKは無効のまま、34配置が布・手先の幾何条件を通過しましたが、右腕の関節余裕・自己干渉・接近前IKの条件まで満たす採用候補は得られていません。

中央配置と独立した手首のひねりを含む追加探索は[中央配置のリーチング再探索](artifacts/centered-reaching/report.html)にあります。32足首状態で22,708,224姿勢を調べ、布・足の幾何条件を190姿勢が通過しました。この段階では従来の腕の開始姿勢を固定し、足全体を開始時の両グリッパー間に収める条件で、追加の左右余白を0mmにした探索も含めて採用軌道は得られませんでした。


torsoを従来の角度 [−45°, 100°, 0°] に固定し、両腕の開始姿勢・人体位置・手首姿勢を再探索した結果、No.30で15cm・約6.15秒のリーチング軌道を1本生成しました。[初期姿勢・人体配置・検証結果](artifacts/fixed-torso-start-search/report.html)にまとめています。足全体は従来の両グリッパー間の範囲内（左寄り）です。登録済み3,688部品による137点の軌道検査と、PhysXの関節ドライブによる再生に合格しました。対象は新しい開始姿勢からの接近で、旧開始姿勢からの移行や、変形する布の把持保持は含みません。


指定写真を基準にした更新：[写真に近い初期姿勢と靴下への短い接近](artifacts/photo-reference-reaching/report.html)。設定は [config/photo_reference_initial_pose.yaml](config/photo_reference_initial_pose.yaml) に保存しました。両肘を外側に張り、肘から手首に向かって下がる構えと、両グリッパーを靴下側に向ける条件を加えています。足全体は新しい両グリッパー間にも収まります。torso固定、初期の最小関節余裕10°、人体の高さ変更−120mm、左足の床貫通約96.6mmです。34mm・2.1秒の接近が69点の衝突検査とPhysXの関節ドライブ再生に合格しました。この右手の短い接近では、終端の閉鎖把持条件は未達です。上記No.30の把持目標への軌道とは別の配置・評価です。


閉鎖時の足との干渉については、同じ写真の初期配置を維持し、左手で靴下の上縁へ回り込む経路で解消しました。[接近・閉鎖までの検証](artifacts/photo-grasp-adjustment/report.html)。右腕は初期姿勢に保持し、左手を40→50mmに開いて接近し、50→1mmまで閉じます。1,361点の補間検査と101点の閉鎖検査、PhysXによる閉鎖までの再生に合格しました。実測関節角度と閉じ幅での再検査でも皮膚離隔約6.70mm、布捕捉面積約4.17mm²です。保存布形状での検証であり、布の変形・摩擦保持・牽引の成功判定は含みません。

共通初期姿勢からの複数条件・複数経路の結果は[左手リーチングの多様化](artifacts/diverse-left-reaching/final/report.html)に保存します。共通設定は [config/diverse_left_reaching_shared.yaml](config/diverse_left_reaching_shared.yaml) です。写真基準の成功例から両腕の第2関節のみ−3°調整し、人体・台座の移動量、torso・head・baseは維持します。条件ごとの開始姿勢の変更を検証時に拒否します。Excelの足首角度と開口位置を参照し、素材・しわ・折り返し等は適用しません。足首だけを変えた追加条件と、開口位置までExcelと一致する条件は結果に別々に記録します。

```bash
# Python 3.11 と geometry extras、opencv-python が必要。
# 既存の成功例・32足首状態・登録済み衝突形状を読み取り専用の入力として使用。
python scripts/search_diverse_left_endpoints.py --stages workbook \
  --output artifacts/diverse-left-reaching/workbook-shared-probe
python scripts/generate_diverse_left_reaching.py \
  --probe artifacts/diverse-left-reaching/workbook-shared-probe
python scripts/replay_diverse_left_reaching.py
python scripts/report_diverse_left_reaching.py
```

同一条件の別経路は、弧長を正規化した手先経路のRMS差5mm以上を要求します。平滑化後の補間検査、実再生、実測角度・開き幅での閉鎖検査を別々に保存します。実再生合格分には18列の実測 `angle.csv` と位相・時刻の `frames.csv` を出力します。布は保存形状に固定され、摩擦保持と牽引、同期画像・トルクの収集は未検証のため、`learning_ready` は `false` です。写真基準と同じ人体左足の床貫通と明示的な許容も継承しています。再生スクリプトはMesaによるソフトウェア描画を既定で使います。

生成時の軌道JSONは変更せず、実行結果は `replays/*/native-route-initial/replay-result.json` と `summary.json` に保存します。容量を抑えるため、再生後の中間シーンは `.json.gz`、生成アセットは `assets.tar.gz`、Playerのログは `.log.gz` として保持します。

開口位置Bの最新版は[つま先の外周を保ったB専用形状](artifacts/diverse-left-reaching/b-natural-toe/shape-report.html)を参照してください。
前回は皮膚のつま先断面に追従して外周が急減し、その先の細い自由部分へ接続していました。
今回は、Bの前足部から基準外周を求め、差し込み深さと余り長さに合わせて、足先を包む部分から扁平な自由部分へ滑らかに接続します。
人体のつま先と靴下自体の閉じた端を区別し、閉じた端の25mmだけを丸く絞ります。
Aの54%という自然長は適用しません。B専用の凍結した幾何形状として構築しており、布変形や重力釣合いの物理検証を意味しません。
30mm帯・中心15mmの半径+5mm、共通のロボット初期姿勢と配置は維持します。

足首10条件（No.3/43は重複、No.10は追加角度）、保存形状11件で、皮膚・自己交差と外周連続性を検査しました。
つま先付近から閉じた端の丸みの手前までは前足部外周の80%以上、隣接断面間の外周比85%以上を要求します。
現在の生成設定では自由部分へ90%まで滑らかに移行させています。
比較写真、外周曲線、JSON/CSVの検査値、形状一式ZIPを同じレポートから参照できます。

```bash
# 既存の足首ごとのネイティブ足形状を参照し、B専用の基準メッシュを生成する。
python scripts/prepare_native_b_sock.py
python scripts/report_fitted_b_geometry.py
python scripts/search_diverse_left_endpoints.py --stages B \
  --numbers 8 10 3 15 19 22 23 31 38 41 43 \
  --maximum-joint-distance-rad 2 \
  --output artifacts/diverse-left-reaching/b-natural-toe/probe
python scripts/generate_diverse_left_reaching.py \
  --probe artifacts/diverse-left-reaching/b-natural-toe/probe \
  --output artifacts/diverse-left-reaching/b-natural-toe --variants 1
python scripts/replay_diverse_left_reaching.py \
  --output artifacts/diverse-left-reaching/b-natural-toe
python scripts/report_diverse_left_reaching.py \
  --output artifacts/diverse-left-reaching/b-natural-toe
python scripts/package_corrected_reaching_collection.py
```

新形状のB把持軌道は現在未採用です。従来の大きい楕円B、v3の12軌道、つま先にくびれのあるv4は現行データから除外します。
採用済みの実測CSVは `artifacts/diverse-left-reaching/corrected-collection/demonstrations.zip`、
対応表は同じフォルダの `manifest.json` と `manifest.csv` です。現在はAの既存12軌道を収録しています。

### 既存12経路のShareSet同期収集（照明補正前の記録）

採用元は `artifacts/diverse-left-reaching/corrected-collection/manifest.json` のAの12経路です。
元の軌道・実測CSV・形状は変更しません。初回の頭部データは `artifacts/reaching-shareset-head-v2/` に保持します。
現行の照明補正版は `artifacts/reaching-shareset-lit-v3/`、従来パス `artifacts/reaching-shareset-v1/` は
全12経路の検査を終えた補正版へのリンクです。補正版の生成手順は次節「頭部画像の照明・深度マスクの修正」を参照してください。
以下は初回の同期収集の仕様と履歴です。

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/sock-collection-mpl \
  /tmp/sock-diversity-venv/bin/python scripts/run_head_reaching_shareset.py \
  --collection-python /tmp/sock-diversity-venv/bin/python --ml-python /usr/bin/python3
```

`camera_right` と `camera_left` はDry-AIRECのURDFにある
`head/see3cam_right/camera_color_frame` と `head/see3cam_left/camera_color_frame`
に配置します。両方とも追加の位置・角度オフセットは0、左右間隔は約84 mmです。
ROSのoptical frameをUnityカメラに直接適用しません。RGBは1280×960、垂直視野角92度で、
実機レンズの精密な内部較正・歪みを再現したものではありません。
固定外部カメラの画像は `camera_overview/` に確認用として別保存します。

物理刻み20 ms、関節角・関節力ログ50 Hz、RGBと学習用CSVは10 Hzです。
3台の画像を同一物理ステップで取得し、`simulate=False` の撮影前後で物理クロックが
進まないことを検査します。各フレームのカメラ姿勢・ステップ・左右間隔は
`camera_poses.jsonl`、同期検査は `raw-validation.json` に保存します。
`angle.csv` の回転関節はrad、直動関節はm、`torque.csv` は対応してNmとNです。
mimic角度は実測主関節の0.5倍、関節力列は同じ駆動DOFの複製です。
関節力は凍結布・高剛性ドライブでの測定値で、布の摩擦保持力ではありません。

データ構成は `/mnt/robot_ssd/ShareSet-from-Trash-20261106-151926/ShareSet.tar`
内の靴下データを読み取り専用で確認しました。左右RGB、右画像由来の `camera_depth/`、
`camera_right_mask/{sock_mask,foot_mask}/`、`depth_mask/{sock_depth,foot_depth}/`、
ヘッダーなし18列の `angle.csv`・`torque.csv`・`external_torque.csv` を保存します。
参照データにはrawと派生画像で1フレームの差がありますが、今回の全モダリティは同じID・同じ件数に揃えます。
参照結果は `reference-format.json` に記録します。

SAM2 smallは頭部右画像の全体を内部512で追跡します。可視の青い開口縁と
保存形状を頭部画像へ投影した領域から正例・負例を選び、保存マスクはSAM2出力を使用します。
布・足の投影支持率、クラス重複、空マスクを全フレームで検査します。
投影支持率の検査は画像上2 pxのラスター許容を使い、遮蔽を含む正解領域との一致率ではありません。
再初期化が必要な場合は現在画像からプロンプトを取り直し、元マスクと理由を
`perception/before-prompt-reset/` と `perception/validation.json` に保持します。
再初期化後も支持率を満たさない孤立フレームは `scripts/repair_head_reaching_perception.py`
で内部1024のSAM2を使い、追跡ずれ領域への負例を追加して自動再処理します。
元のマスク・マスク付き深度・検査結果と変更後のハッシュを
`perception/isolated-repair-originals/`・`head-repairs.json` に保持します。
RGB、右画像の深度推定値、関節信号は変更しません。
Depth Anything V2 vitsの正規化範囲は、訓練8経路（No.10/13/25/38）の全頭部右画像だけで固定します。
下限はモデルのReLU下限0、上限は全訓練画像の最大値に量子化1段分の余裕を加えます。
疎な24枚だけの較正では訓練途中の深度ピークを逃すため、この旧較正は使用しません。
近くのロボットや背景がこの範囲外になる場合も較正範囲は固定し、全画面のクリップ率を記録します。
`scripts/audit_head_reaching_shareset.py` は保存画像の復号、左右画像の重複、
マスクと深度の整合、学習対象の靴下・足の上限飽和率（各2%以下）を独立に検査します。
Depth Anything V2はReLUで0を出力し、8bit量子化でも0が生じるため、0を欠測とは扱いません。
訓練較正の下限は0から量子化1段未満であることを確認し、対象領域の0値率も記録します。
俯瞰画像用のROI・プロンプト・深度較正・学習済み重みは流用しません。

分割はNo.10/13/25/38がtrain（8経路）、No.20がvalidation（2経路）、No.50がtest（2経路）です。
ShareSetの頭部用設定は `../ShareSet/share_folder/configs/reaching_head_simulation.yaml` です。
従来の `reaching_simulation.yaml` も同じ頭部用設定に更新しています。
`expected_camera_profile` により俯瞰データの混入を拒否します。現行モデル入力は頭部右画像で、
頭部左画像は同期したステレオ対として保持します。CSVの先頭行を保持し、正規化はtrainだけで求め、
可変長系列のパディング部分を損失から除外します。`skip_num=2` の学習刻みは0.2秒です。
代表経路と全12経路の各1 epochの学習確認を行います。

旧画像の保存と従来パスの切り替えは、全検査完了後に実行します。

```bash
/tmp/sock-diversity-venv/bin/python scripts/activate_head_reaching_shareset.py
```

旧出力一式（俯瞰の深度・マスク・学習モデル・ログを含む）は
`artifacts/reaching-shareset-overview-v1-archive/` に退避します。元の `camera_right` は
現行の各episodeの `confirmation/camera_overview_original/` にもそのままコピーし、
全画像のSHA-256を `camera-migration.json` に記録して照合します。
旧メタデータ内の絶対パスは履歴として保持し、退避先への対応は `RELOCATION.json` に示します。
旧俯瞰向けスクリプトは頭部データに対して停止し、誤ったROIでの再処理を防ぎます。

合格済みの生収集と画像処理は再利用します。失敗した生収集は上書きせず停止するため、
ログを確認して失敗episodeを退避してから再開してください。
実行状況は現行出力の `collection.log`・`pipeline.log`、成果物は
`report.html`・`report.json`・`episodes.csv`、分割は `data/dataset_reaching.yaml` です。
対象は固定布への接近・幾何学的閉鎖です。`reaching_learning_ready` はこの範囲のデータ品質を示し、
`learning_ready` と `dynamic_grasp_validated` はfalseのままです。
1 epochの学習確認は動作成功率の検証を含みません。

### 頭部画像の照明・深度マスクの修正

照明補正版は `artifacts/reaching-shareset-lit-v3/` に作成します。
12経路・全6,607フレームの保存データ検査とShareSetの代表・全12経路の各1 epochの学習確認に合格しています。
可視布とのIoUは平均0.890→0.959、最小0.259→0.933です。実機例と修正前後は
[比較レポート](artifacts/reaching-shareset-lit-v3/report.html)で確認できます。
靴下の表裏が同じ頂点を使う描画メッシュでは法線が打ち消され、布が黒く描画されていました。
`unity/RuntimeExtensions/SockLearningLighting.cs` は描画用の表裏頂点を分離して法線を復元します。
物理粒子・布の三角形の位置・保存形状は変えません。
`balanced` 設定は環境光0.20、主光0.65・回転[55,180,0]、影の強さ0.25、
左右の補助光0.25・回転[40,90,0]/[40,270,0]です。開発プレイヤーは従来版とは別に保存します。

```bash
bash scripts/build_lighting_player.sh
/usr/bin/python3 scripts/render_lit_reaching_dataset.py --workers 6
/usr/bin/python3 scripts/finish_lit_reaching_dataset.py --workers 6
/usr/bin/python3 scripts/finalize_lit_reaching_dataset.py
```

描画と前処理は別プロセスで同時実行できます。描画は旧頭部データの10Hz実測姿勢を配置する処理で、
新たな力・接触計測ではありません。Unityの関節座標を描画へ同期する1µsの処理を使い、
姿勢のずれを防ぐため画像配置中のロボット衝突判定を停止します。途中再開した画像については
各 `camera_poses.jsonl` に衝突判定停止の有無を記録します（旧記録で欄がない場合は有効）。
全フレームで元の関節座標・頭部カメラ姿勢・固定布の位置との誤差を検査し、
`angle.csv`・`torque.csv`・`external_torque.csv`・`frames.csv`・50Hzログは元ファイルと同じ内容を保持します。

学習マスクは補正版RGBに対するSAM2出力です。布と足の投影支持率を各0.95以上、
独立した可視布ID描画とのIoUを0.90以上として全フレームを検査します。
ID描画は `qa/visible_sock/` に検査専用で保存し、学習入力や正解マスクへの置換には使いません。
追跡が外れた場合は現在画像から再初期化し、必要なら内部1024のSAM2で再推定します。
深度較正は補正版のtrain全4,542画像だけで固定し、validation/testは較正に使いません。

最終処理は保存された全画像の独立検査と代表・全12経路のShareSet学習確認を必須とし、
合格後に従来パスとShareSetの両設定を補正版へ切り替えます。
`expected_lighting_profile: balanced` が照明修正前データの混入を防ぎます。
修正前の頭部データは `reaching-shareset-head-v2/` のまま保持し、各新episodeの
`confirmation/previous_head_data/` から参照できます。元の俯瞰画像も同じく保持します。
`report.html` には指定tarから抽出した実機例と同じ記録姿勢での修正前後を並べます。
IoUはシミュレーション内の可視形状との一致率であり、実機との形状一致率ではありません。

### 実機アーカイブから作成した重力負荷の基準モデル

重力だけの基準モデルは `artifacts/gravity-calibration-v1/gravity-model.json` と
`calibrated_gravity.urdf` に保存しています。
[較正・検証レポート](artifacts/gravity-calibration-v1/report.html)には抽出画像、関節別誤差、
実機と同じ角度列で計算した重力負荷、Unityでの検証をまとめています。
指定tarの靴下11経路から、両グリッパーが開いて布から離れた終端の低速区間135フレームを抽出しました。
画像上の離隔を確認していますが、独立した力センサーによる無接触証明ではありません。
元の9 train経路114フレームでモデル選択・較正し、2 test経路21フレームを評価専用に保持します。

```bash
/usr/bin/python3 scripts/calibrate_gravity_reference.py
bash scripts/build_gravity_player.sh
bash scripts/run_gravity_reference.sh
/usr/bin/python3 scripts/export_gravity_reference.py
MPLCONFIGDIR=/tmp/sock-gravity-mpl /usr/bin/python3 scripts/report_gravity_reference.py
```

アーカイブの `torque.csv` は `ToroboState.link_effort`、`external_torque.csv` は
`external_effort` の記録です。`gravity_effort` 自体は保存されていないため、
低速区間の `link_effort - external_effort` を制御器の重力成分の推定値として使用しました。
この推定値に対する較正モデルのテストRMSEは0.0292 Nm、生のlink_effortに対しては1.5015 Nmです。
生トルクの静止残差は別項に保存し、物理的な重力には加えていません。
対象は腕14関節であり、グリッパー力の較正・摩擦・慣性テンソル・布の反力は含みません。
64パラメータに対する回帰行列のランクは28で、質量・重心の個々の値を一意に同定した結果ではありません。
小さい補正を元URDFへ正則化した重力の基準モデルです。

専用プレイヤー `RegisteredGravityV1` と `StaticGravityReference.configure_unity(env)` は、
URDFに質量指定のない補助リンクへ残る既定1kgを修正します。質量ゼロのリンクに設定する
数値計算上の下限は1mgです。重心のROS→Unity座標変換とグリッパー連動軸の直接姿勢同期も検査します。
重力のみの静止保持として、関節摩擦・剛体抵抗は0、保持用の駆動剛性1e5・駆動減衰5e3を使います。
SDKの `SetJointDamping` は関節摩擦と剛体抵抗も変えるため、この専用設定では使用しません。
これは実機サーボの同定値ではありません。

```python
from sock_dressing_simulation.gravity_reference import StaticGravityReference
reference = StaticGravityReference("artifacts/gravity-calibration-v1/gravity-model.json")
gravity_nm = reference.predict(q18)  # N×18のrad/m入力 → N×14の腕重力負荷[Nm]
# 専用プレイヤー上で、重力のみの静止保持を明示的に設定する場合:
# reference.configure_unity(env)
```

135姿勢でのUnity重力計算と解析モデルの最大差は0.000088 Nm以下、
重力を有効にした静止保持の駆動力との差のRMSEは0.0045 Nm以下です。
実機CSVの微小な関節上限超過はUnity検証時のみURDF上限へ合わせ、元CSVは保持しています。
既存12経路・6,607姿勢の結果は `artifacts/gravity-calibration-v1/simulation-routes/` の
`gravity_arm.csv` と列順を明記した `metadata.json` に出力します。
この14列の重力成分は、従来の18列 `torque.csv` と区別してください。
元のShareSet学習データ・画像・トルク記録は保持し、学習対象の自動切り替えは行っていません。

### 最新版Bの把持軌道探索

最新版Bの全足首条件に対する追加探索は[把持軌道探索の結果](artifacts/diverse-left-reaching/b-natural-toe/grasp-search/report.html)にまとめています。
共通初期姿勢・配置と保存形状を固定し、10条件で合計4,258,552姿勢候補を評価しました。
従来格子、周方向64点×帯深さ13点、実際の評価窓に対する手先の三次元位置、連続した姿勢修正、開口幅の細分化、局所連続最適化を含みます。
閉じ幅1mm・皮膚離隔1mm・捕捉面積4mm²・固定8×8mm窓の条件は維持しています。
一部に閉鎖時の離隔と面積を満たす候補がありましたが、把持前の布・足との非干渉を同時に満たす把持目標は得られず、新規合格軌道は0本です。
これは有限の探索範囲での結果であり、不可能性の証明ではありません。

```bash
python scripts/search_diverse_left_endpoints.py --stages B \
  --numbers 8 10 3 15 19 22 23 31 38 41 --seconds 90 \
  --output artifacts/diverse-left-reaching/b-natural-toe/grasp-search/probe
python scripts/search_natural_b_grasps.py --seconds 120 \
  --normal-offset-mm 0.3 --tangent-offset-mm 4 \
  --output artifacts/diverse-left-reaching/b-natural-toe/grasp-search/dense-probe
python scripts/search_natural_b_grasps.py --numbers 19 38 8 --seconds 150 \
  --normal-offset-mm 2 --tangent-offset-mm 6 \
  --output artifacts/diverse-left-reaching/b-natural-toe/grasp-search/wide-pose-probe
python scripts/polish_natural_b_grasps.py --numbers 19 38 8 --seconds 90
python scripts/report_natural_b_grasp_search.py
```

時間制限を使うため、実行環境によって試行数は変わります。
探索記録には形状と共通初期設定のハッシュ、補間点、乱数種、位置範囲、段階別の不合格理由を保存しています。
密な補間点から候補が見つかった場合は、保存した補間点を軌道生成でもそのまま使い、実際のIK姿勢で再評価します。
数値的に不確定な幾何判定は不合格として記録します。
