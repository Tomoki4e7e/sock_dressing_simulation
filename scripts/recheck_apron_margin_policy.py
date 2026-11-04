#!/usr/bin/env python3
"""Reevaluate the explicit-start margin policy without rerunning unchanged collisions."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from apron_reaching_common import start_margin_transition,overlay_robot
from search_translated_reaching import load
from sock_dressing_simulation.arm_reaching import Kinematics
OUT=ROOT/'artifacts/apron-reaching';robot=json.loads((OUT/'robot-scene.json').read_text());k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',overlay_robot(load(3),robot))
for path in (OUT/'routes').glob('*-validation.json'):
 r=json.loads(path.read_text());plan_path=path.with_name(path.name.replace('-validation.json','-trajectory.json'));plan=json.loads(plan_path.read_text());policy=start_margin_transition(plan['q_rad'],k.lower,k.upper,k.start);r.setdefault('previous_uniform_margin_failures',r['failures']);r['failures']=[v for v in r['failures'] if not v['check']['passed']];r['joint_margin_policy']=policy;r['offline_passed']=bool(not r['failures'] and policy['passed'] and all(v['passed'] for v in r['opening_checks']+r['virtual_closure_checks']) and r['endpoint_position_error_m']<=1e-4 and r['endpoint_orientation_error_deg']<=.5);plan['joint_margin_policy']=policy;plan['offline_passed']=r['offline_passed'];path.write_text(json.dumps(r,indent=2));plan_path.write_text(json.dumps(plan,indent=2));print(r['number'],r['offline_passed'],policy['first_full_margin_index'])
