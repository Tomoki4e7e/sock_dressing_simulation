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

開口位置C〜Eは、Aと同じnative外周フィット・断面質点モデルによる重力下垂・開口帯の膨らみを適用します。
各足首姿勢で自然外周を縮め、通常重力で10秒の保持を検査した後、足に掛かる部分をつま先の材料位置まで保持して下垂を準備します。
最後にAと同じメッシュ細分化と30mm帯・中心15mm・半径+5mmの膨らみを適用し、皮膚・自己交差を検査します。
自然外周倍率はAの0.54を固定でコピーせず、Aと同じ張力条件で各段階をフィットします。
足首からすねまでを囲むため、C〜Eの保存形状は足首姿勢ごとに作り、足の座標系で一括回転しません。
下垂にはAの材料断面への対応付け・曲率制限・境界補間の探索を使います。準備は既定150ステップです。
支持部の誤差5µm未満、自由部速度0.05m/s未満、直近1秒間の全頂点変動2mm未満を要求します。
C〜Eの短い自由端には、Aの長い自由端を前提とした質量5g・落下量20mmの下限を強制しません。
外周フィット時にも重力を掛けているため、その後の追加落下量や先端の鉛直性は合否条件にせず、計測値として残します。
外周縮小の停止目標は0.08・0.04・0.02Nを順に試します。保持・張力・交差の合格条件は共通です。
外周フィットの接触補正は、Aの局所接触と同じく摩擦・減衰より前に実行し、減衰後の速度再計算による振動を避けます。
自己接触の隣接判定にもAと同じ材料座標を使い、足首の折れ曲がりで離れた布同士の接触が除外される問題を防ぎます。
失敗した候補は`fit-attempts/`に保存し、物理検査を通過した候補だけを採用します。
探索時に必要な形状がない場合は自動準備し、従来の楕円形状や古い探索キャッシュへのフォールバックは行いません。
最終的な膨らみ付き形状は、Aと同じくリーチング用に凍結します。

外周をさらに足へ近づけるため、native準備後の支持部を現在の実皮膚の断面に沿って調整します。
既定では断面の余裕を75%減らす方向へ動かし、目標の皮膚余裕を2.5mmとします。
各断面の移動は辺・面交差を確認して段階的に小さくし、全体の皮膚・自己交差検査も必須とします。
つま先側の最後の2支持断面と下垂部分の座標は完全に維持し、記録済みの下垂形状をそのまま使います。
その後に30mm帯・中心15mmの半径+5mmを加えます。
これは支持された凍結形状の追加調整です。調整後の形状で、支持なしの摩擦保持や牽引が成立するとは判定しません。

```bash
bash scripts/build_held_stage_player.sh
# ExcelでC〜Eに該当する行を準備。各物理段階は新しいPlayerで実行する。
python scripts/prepare_held_stage_shapes.py
# 検証済みの代表条件（C/EはNo.10、DはNo.45）
python scripts/prepare_held_stage_shapes.py --numbers 10 --stages C E
python scripts/prepare_held_stage_shapes.py --numbers 45 --stages D
python scripts/preview_held_stage_shapes.py --number 10 --stages C E
python scripts/preview_held_stage_shapes.py --number 45 --stages D
# ExcelのC〜Eを開口位置・pitch・rollでまとめ、失敗条件も記録して全条件を試す。
python3 scripts/expand_held_stage_patterns.py --workers 8
# 保存形状と物理レポートを照合し、全30行の対応と重複除外後の数を出力。
python3 scripts/report_held_stage_expansion.py
```

出力は `artifacts/cde-tight-fit/pose-<番号>-<段階>/` です。以前の `artifacts/cde-a-method/` は比較用に残します。
ExcelのCとEは各10姿勢、Dは8姿勢です。DのNo.2/11とNo.30/42はそれぞれ同姿勢として扱います。
一括試行では各条件のログを `expansion.log` に残し、失敗しても他条件を継続します。
初期配置を粗い凸包の面交差だけで拒否した場合は、現在の実皮膚でも検査します。
凸包への粒子侵入・布辺の交差、実皮膚との辺・面交差、布の自己交差のいずれかが残る場合は開始しません。
初期配置が通らない場合は、フィット前の余裕を15・17.5・20・25・30mmで試し、最初に通ったものを使います。
変更するのは縮小前の配置だけで、材料・最終フィット・保持・交差の合格条件は共通です。
最終結果は [追加試行の一覧](artifacts/cde-tight-fit/expansion-report.html) と
同フォルダの `expansion-report.csv`・`expansion-report.json` に保存します。
既存のC/No.10・E/No.10はExcelの当該開口行とは別の姿勢として、全体の種類数に含めます。
2026-11-08の追加試行では、ExcelのC〜Eの30行（重複除外28姿勢）のうち26姿勢が合格しました。
Cは1→11種類、Dは1→6種類、Eは1→11種類となり、Aの32種類・Bの10種類と合わせて45→70種類です。
足首角度だけでは引き続き32種類です。合格したC〜Eの全28保存形状でUnityの粒子・皮膚の読み戻しも確認しました。
D/No.21は3候補とも形状検査を通らず、最後の0.02N候補でも自己交差が15辺残りました。
D/No.36は最後の0.02N候補で交差は解消しましたが、保持の直近1秒間の開口粒子変動が9.46mm（上限5mm）あり未採用です。
両条件とも0.08・0.04・0.02Nの候補と物理・交差検査の結果を `fit-attempts/` に保持しています。
既存の合格済みnative基準がある場合は、その形状・材料・検査結果を入力として再利用します。
`fit/report.json` は外周フィットと保持検査、`settling-report.json` は支持部と下垂部の物理検査、
`surface-fit-report.json` は追加の断面調整と隙間・外周の比較、`state.json` は膨らみ適用後の凍結形状です。
`preview-<番号>/` にUnity画像と粒子・皮膚の読み戻し検査を保存します。
`preview_held_stage_shapes.py --compare-root artifacts/cde-a-method` は、調整前後を同じカメラで撮影します。
足の凸包が実皮膚より張り出す箇所の交差は診断値として残し、Aと同じ実皮膚の辺・面交差検査で最終判定します。
保持・質量・張力・侵入量の物理条件は引き続き必須です。
No.10のDは検証した候補で保持時の振動が残り、採用していません。全50行の物理合格を意味するものではありません。

下垂の重力は `--drape-gravity-scale`（既定1、0より大きく1以下）で調整できます。
短い自由端が開口側の振動に影響しているかは、以下の比較で確認できます。
同一の保存済み状態を復元し、足に掛かる部分の重力・質量・材料を保ち、つま先より先の自由端だけを弱めます。

```bash
python scripts/diagnose_stage_tail_gravity.py \
  --reference artifacts/cde-a-method-material-force02/pose-10-D/fit --scale 1
# 同じ入力で --scale .75、.5、.25 を比較する。
```

結果は `artifacts/cde-tail-gravity-v2/gravity-<倍率>/` に、開口・支持部・自由端を分けた変動量と検査結果を保存します。
自由端の重力変更は、接触摩擦の法線荷重の計算にも反映します。足に掛かる部分の重力は通常のままです。
この短時間の比較結果を、10秒間の保持試験に合格した新しい初期状態として自動採用することはありません。

保存済みの追加調整では、C/No.10の外周39.71→36.30cm、D/No.45は28.82→23.46cm、E/No.10は32.04→26.01cmです。
前後の同一カメラ画像は `artifacts/cde-tight-fit/fit-comparison.jpg`、検査値は同じフォルダの `validation.json` に保存します。
D/No.10の自由端重力100・75・50・25%の3秒比較では、どの条件も静止検査に合格しませんでした。
25%を10秒まで延長した試行でも、最後の1秒の開口粒子変動が13.87mm（基準5mm）あり、採用していません。
延長試験は `artifacts/cde-tail-gravity-v2-long/gravity-0.25/` に保存しています。通常設定の重力倍率は1のままです。

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

70初期形状に共通する左右の新しい開始姿勢と接近IKは、
[全140軌道の検査結果・画像・CSV](artifacts/common-opening-approach/negative-wrist-start/report.html)に保存しました。
設定は [config/common_opening_initial_pose.yaml](config/common_opening_initial_pose.yaml) です。
A32・B10・C11・D6・E11の全70条件で、右腕70本・左腕70本が補間後の衝突検査を通過しています。
片腕ずつ接近し、他方の腕は全条件共通の初期姿勢に保持します。人体・台座・胴体・頭部の配置は維持します。
左腕は既存A/No.10の開口部直前の通過点（元の3区間経路の0始まり添字278）を中心にし、
右腕は足の矢状面に対する鏡映位置から約28mmの位置を選びました。右手首の別解を使い、初期の最小関節余裕は20.92°です。
共通初期姿勢のUnity復元も合格しています。全140軌道の動力学再生と従来Aの把持条件の評価は[追加検証](artifacts/common-opening-dynamics/report.html)に保存しています。接近再生は115本が基準内、把持条件の合格は0本です。自由な布変形・摩擦保持・牽引は未検証です。

目標は開口縁から材料長0〜15mmの帯に対し、物理指先の中央面が表面の2mm手前に来る姿勢です。
TCPから指先までの10mmを補正し、指を含む実形状と登録済みの全ロボット形状で検査します。
全軌道の関節余裕5°以上、位置IK誤差0.1mm以下、URDFの速度・加速度上限の10%以下を確認しました。
CSVには時刻、右腕7関節、左腕7関節、左右開き幅、移動腕のTCP位置を保存します（rad・m・s）。
開き幅を変更する場合は、初期位置での開閉区間を含みます。従来の実測18列データとは別のIK指令形式です。
一式は同レポートの `ik-trajectories-70-states.zip`、元形状への対応とハッシュは `states.json` と `summary.json` にあります。

```bash
# 新規出力先で開始姿勢を探索する場合。候補16は別の右手首配置を使う。
python scripts/search_common_opening_starts.py starts --right-candidate-index 16 \
  --output artifacts/common-opening-approach/new-search
python scripts/search_common_opening_starts.py routes --guided --workers 12 \
  --output artifacts/common-opening-approach/new-search
# 接続未達条件では、検証済みの肩境界と同じ側にある別の目標IKを作って再試行する。
python scripts/seed_same_component_opening_goals.py \
  --output artifacts/common-opening-approach/new-search
python scripts/search_common_opening_starts.py routes --guided --workers 12 \
  --output artifacts/common-opening-approach/new-search
python scripts/report_common_opening_starts.py \
  --root artifacts/common-opening-approach/new-search
```

衝突判定の条件は共通です。新探索でのみ大きな部品群の空間検索を有効にし、従来の総当たりAABB候補を取りこぼさないことを検査しました。
既存Aの軌道と実測データは変更していません。連続時間の衝突証明ではなく、補間曲線と中間点の離散検査です。

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
重力基準モデル作成時は元のShareSet学習データ・画像・トルク記録を保持し、学習対象は切り替えませんでした。
現在の `reaching_head_simulation.yaml` と `reaching_simulation.yaml` は
`dataset.torque.source: recorded_drive_plus_calibrated_gravity` を指定し、読み込み時に
元の駆動力の腕14列へ同じ姿勢の較正済み重力を加算します。グリッパー4列と元CSVは保持します。
平滑化・間引き・train正規化は加算後の値に対して行い、入力と次時刻の教師信号の両方に反映します。
`gravity_root` は `dataset.root` 基準、較正モデルは `gravity_model_sha256` で固定します。
重力が有効な記録、列順・件数や角度/駆動力/時刻のハッシュが一致しない参照は拒否します。
旧モデルの正規化や重みは更新されないため、補正後の学習は別の出力先で実行します。
この補正は重力無効の運動記録への解析的な荷重加算で、摩擦・布反力の較正や、重力下での動的な再収集は含みません。

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

### 共通初期姿勢からの140軌道の動力学・把持検証

[検証レポート](artifacts/common-opening-dynamics/report.html)と
[全件の集計](artifacts/common-opening-dynamics/summary.json)に、70状態×左右腕の結果を記録します。
接近再生の合格は右腕65/70本、左腕50/70本です。元の計画終点は全140本で捕捉面積0mm²となり、従来Aの把持条件は全件不合格でした。
入力は `artifacts/common-opening-approach/negative-wrist-start/` の保存軌道です。
保存形状、共通初期姿勢、登録衝突形状のハッシュと、補間・時間配分を照合してから、各軌道を独立したPhysXシーンで再生します。
実測関節角度・開き幅のCSV、接触と追従の時系列、全身の離隔、固定側の変位、速度・加速度、終端誤差を保存します。
接近再生の合否と、終点での閉鎖・把持条件の合否は別に集計します。

把持条件は従来Aと同じ `config/geometric_closure.yaml` と判定関数を使用します。
閉じ幅1mm、皮膚離隔1mm、捕捉面積・最大連結面積4mm²以上、面積比5%以上、両方向の捕捉幅0.5mm以上です。
計画終点の静的検査は全140本に実施し、動力学再生を途中停止した場合も元の終点を別途評価します。
閉鎖中に許容する重なりは、動かしているグリッパーと保存布の組み合わせだけです。
把持の幾何条件を満たさない場合は閉鎖を省略し、理由を記録します。初期の試行には離隔を満たす場合の閉鎖再生も含みます。
閉鎖途中の離隔検査で不合格なら、危険な閉鎖は実行せずに理由を保存します。
以前合格したAの実測データで、面積・幅・皮膚離隔と合否の一致も確認します。
布は保存形状を保持した評価で、摩擦による保持、自由な布変形、引き上げの成功は検証対象に含みません。
ロボット関節の重力は従来Aと同じ無効で、重力荷重や駆動力上限の評価は含みません。

```bash
python scripts/validate_common_opening_dynamics.py --workers 8
python scripts/check_common_opening_grasp_goals.py --workers 3
python scripts/validate_common_opening_dynamics.py --report-only
python scripts/package_common_opening_dynamics.py
```

同じ入力・把持条件の完了済み結果は再利用します。新規の再試行には別の `--output` を指定します。
検証条件・ハッシュは `artifacts/common-opening-dynamics/protocol.json`、
Aの陽性対照は `a-criteria-equivalence.json` に保存します。

### 把持終点を条件にした共通初期姿勢の再探索

[再探索の結果](artifacts/common-grasp-start-search/report.html)と
[70状態の一覧](artifacts/common-grasp-start-search/cases.csv)を保存しています。
全70状態で把持へ到達する共通初期姿勢は未取得です。
グリッパー単体の把持終点は29状態（A28、C1）で得ましたが、B10状態を含む41状態では未取得でした。
候補初期姿勢1組に対する実際のIK・全身閉鎖検査では左腕16状態、右腕0状態が合格しています。
いずれも接近経路・動力学再生の合格を意味しません。

初期姿勢は各腕48候補を生成し、全70状態の布と32足首姿勢の人体を重ねた検査で24組が合格しました。
これらは初期干渉の検査のみを通った未採用候補です。公開済みの初期姿勢設定は更新していません。
把持の判定は従来Aの閉じ幅1mm・皮膚離隔1mm・捕捉面積4mm²等を維持しています。
開き幅の連続探索、成功姿勢の材料座標による移送、足の反対側への反射、指の入れ替え、接近方向反転、
B全10状態の連続最適化、腕の実際のFK姿勢を新たな目標とする探索を実施しました。
提案した位置・向きとの一致を緩めても、合格には実際のFK姿勢で元の把持条件を満たすことを要求します。
有限範囲・時間内の探索であり、解の非存在を証明する結果ではありません。

```bash
python scripts/search_common_grasp_geometry.py --workers 8 --output artifacts/new-grasp-search
python scripts/optimize_common_grasp_geometry.py --id B-19 --seconds 600
python scripts/check_common_grasp_arm_endpoints.py --guided --loose --pass-name arm-space-search
python scripts/report_common_grasp_search.py
```

探索入力・判定条件のハッシュ、未採用の初期姿勢候補、合格終点の独立した再検査、
旧Aの陽性対照、代表姿勢の開き幅診断を記録しています。
`programs/` は実行コードの控えで、実行には元のプロジェクト構成とモデル資産が必要です。

### 閉鎖しない両グリッパーの布・足接触条件による探索

`scripts/search_contact_pregrasp.py` は、保存済みの全70状態について、左右のグリッパーを
足の反対側に置く新しい接近課題を探索します。各グリッパーの一方の爪が靴下の
開口境界から材料座標で15〜30mm内側の表面、もう一方の爪が足の実皮膚に
表面間距離0〜0.5mmで接し、各グリッパー近傍の開口境界線の長さを持つ区間が
実際の有限な爪間に入ることを要求します。靴下の全円周を各グリッパーに入れる意味ではありません。
帯境界を横切る布三角形も切り分けて判定し、開口線には実メッシュの境界辺を使います。
皮膚の接触対象は足部から下腿までを含む対象脚の実皮膚面全体で、足首で切り分けません。

貫通は許可しません。接触を指定した足側の爪についてのみ、旧課題の皮膚離隔1mmを
0mmに変更し、他部品の皮膚離隔1mmと登録形状の全身離隔を維持します。
関節余裕の既定値は5度です。`--joint-margin-deg 0.5` で0.5度にも設定できます。実際のURDF関節範囲は越えません。
旧課題の閉じ幅1mm・捕捉面積4mm²・閉鎖掃引は、この新課題の合否に使用しません。
開き幅を合わせる動作は離れた共通初期姿勢で行い、接近中と接近終点では幅を固定します。
把持確認の閉鎖動作は生成しません。保存布形状・足姿勢・人体配置・ロボット基部と胴体を維持します。

```bash
PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/search_contact_pregrasp.py geometry --workers 8 --seconds 35
PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/search_contact_pregrasp.py endpoints --workers 8 --seconds 40
PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/search_contact_pregrasp.py audit
PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/search_contact_pregrasp.py starts --joint-margin-deg 0.5
PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/search_contact_pregrasp.py routes --workers 4 --pairs 4 --seconds 60
PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/search_contact_pregrasp.py collect --joint-margin-deg 0.5
```

出力先は `artifacts/contact-pregrasp-start-search/` です。`protocol.json`、状態ごとの接触点・
距離・境界線区間を含む `geometry/` と `endpoints/`、全70状態を重ねた初期干渉検査、
初期姿勢候補ごとの両腕同時終点・順次接近経路を保存します。
左右両方の初期位置候補を検査し、採用候補に対する結果を `routes/` と
`candidate-initial.yaml` にまとめます。成功経路のCSVには両腕14関節と両爪間隔を共通時刻で記録します。
`starts` は足の外側・上方の待機姿勢を追加し、全70状態の障害物を重ねて検査します。
`collect` は保存済みの探索結果を再集計し、到達できる状態数が多い初期姿勢を採用します。
同数なら初期姿勢の関節限界からの余裕が大きい候補を選びます。
接触姿勢への接近では初期姿勢側からの逆運動学も試し、保存済み終点と異なる肘の配置に
到達した場合には、その実際の終点について接触条件と両腕の同時干渉を再検査します。
`endpoints --right-joint-1-min-deg 111` のように探索範囲を絞ることもできます。
これは接触の合否条件ではなく、共通初期姿勢と同じ肩関節側で別の終点を探すための指定です。
`--right-elbow-min-deg 30` では、右腕第4関節を30度以上として、肘が伸び切る付近を避けた終点を探せます。
既存の別範囲の終点を保持するため、この指定での探索には別の出力ディレクトリを使います。
全70状態で同一の初期設定から両腕の接近が成立した場合に限り `reachability_verified` が真になります。
有限探索の未取得を不可能性とはみなしません。軌道の干渉検査はサンプル検査であり、
連続時間での非干渉、動力学、接触力、摩擦保持、引き上げの成功は未認証です。
