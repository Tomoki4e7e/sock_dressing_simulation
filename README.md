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
`change_pose.py` `ka` posture and the three torso joints are set to its `kb`
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
The report requires complete live inference, 1 mm original-anchor accuracy,
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
