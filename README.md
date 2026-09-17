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

## Commands

```bash
python3 -m sock_dressing_simulation.cli doctor
python3 -m sock_dressing_simulation.cli prepare-assets
python3 -m sock_dressing_simulation.cli smoke --offline
python3 -m sock_dressing_simulation.cli smoke --headless --frames 3
python3 -m sock_dressing_simulation.cli calibrate
python3 -m sock_dressing_simulation.cli collect --graphics --frames 30 --seed 0
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
