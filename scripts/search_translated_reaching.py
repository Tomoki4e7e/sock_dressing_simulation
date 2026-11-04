#!/usr/bin/env python3
"""Offline rigid scene translation search; retains URDF and native collision shapes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import json,sys,copy,argparse
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree,ConvexHull
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene,shape_vertices

OUT=ROOT/'artifacts/translated-reaching'
def load(n):
    folder=ROOT/f'artifacts/bulged-cuff/pose-{n:02d}'
    scene=json.loads((ROOT/'artifacts/bulged-cuff/native/shared-robot-scene.json').read_text());scene.update(json.loads((folder/'native-scene.json').read_text()))
    return scene

def translated(scene,d,points):
    s=copy.deepcopy(scene);s['human_skin']['vertices']=(np.array(s['human_skin']['vertices'])+d).tolist();s['particles_world']=(np.array(points)+d).tolist()
    for r in s['obstacles']:
        if 'Ground' in r['name']:continue
        if r['kind']=='MeshCollider':r['vertices_world']=(np.array(r['vertices_world'])+d).tolist()
        else:r['position']=(np.array(r['position'])+d).tolist()
    return s

def audit(k,scene):
    rows=[];frames=k.fk(k.start)
    for r in scene['robot_colliders']:
        name=r['link']
        if name not in k.hand.meshes:continue
        raw=k.hand.meshes[name][1];T=frames[name];expected=(raw@T[:3,:3].T+T[:3,3])@k.A.T+k.b;actual=shape_vertices(r)
        error=max(cKDTree(actual).query(expected)[0].max(),cKDTree(expected).query(actual)[0].max())
        hull=ConvexHull(raw);nativeh=ConvexHull(actual)
        # Signed plane distance of original vertices: convexification fills recesses.
        depth=np.maximum(0,-(raw@hull.equations[:,:3].T+hull.equations[:,3]).max(axis=1))
        rows.append(dict(link=name,native_convex=r['convex'],vertex_hausdorff_m=float(error),urdf_hull_volume_m3=hull.volume,native_hull_volume_m3=nativeh.volume,maximum_vertex_inward_plane_distance_m=float(depth.max())))
    assert len(rows)==5
    return dict(passed=all(r['vertex_hausdorff_m']<2e-5 for r in rows),parts=rows,scope='Saved native MeshCollider input vertices versus URDF collision STL with scale, origin and FK; cooked PhysX/Obi hull not extracted',calibration=k.calibration)

def margin(k,q):return float(np.rad2deg(np.minimum(q-k.lower,k.upper-q)).min())
def good(r):return r['position_error_m']<=1e-4 and r['orientation_error_deg']<=.5

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=12);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--shard-index',type=int,default=0);args=ap.parse_args();OUT.mkdir(exist_ok=True)
    scene=load(39);urdf=ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf';k=Kinematics(urdf,scene)
    (OUT/'gripper-audit.json').write_text(json.dumps(audit(k,scene),indent=2))
    targets=[]
    for path in sorted((ROOT/'artifacts/bulged-cuff').glob('pose-*/search/geometry-*.json')):
        r=json.loads(path.read_text());state=json.loads((ROOT/r['source_state']).read_text());targets.append((str(path.relative_to(ROOT)),r['number'],r['candidate'],[v['gap_m'] for v in r['valid_openings']],state))
    for path in sorted((ROOT/'artifacts/geometric-closure').glob('pose-*/report.json')):
        seen=set()
        for r in json.loads(path.read_text())['candidates']:
            if not r['accepted']:continue
            c=r['candidate'];key=json.dumps(c,sort_keys=True)
            if key in seen:continue
            seen.add(key);state=json.loads((ROOT/r['initial_state_source']).read_text());targets.append((str(path.relative_to(ROOT))+':'+str(r['candidate_index']),r['number'],c,[r['opening_gap_m']],state))
    for target in targets:
        state=target[-1]
        if 'particles_world' not in state:state['particles_world']=state['initial_particles_world']
    counts=Counter();results=[];rng=np.random.default_rng(20261105);bound=np.array([.3,.15,.3]);pad=np.deg2rad(5)
    for index,(source,n,c,gaps,state) in enumerate(targets):
        if index%args.shards!=args.shard_index:continue
        cached=OUT/f'target-{index:03d}.json'
        if cached.exists():
            row=json.loads(cached.read_text());assert row['source']==source;results.append(row);continue
        p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix']);attempts=[];accepted=[]
        for seed in range(args.seeds):
            q=k.start if seed==0 else rng.uniform(k.lower+pad,k.upper-pad)
            # Translation is a decision variable, while orientation remains exact.
            def fun(x):
                actual,rot=k.pose(x[:7]);return np.r_[actual-p-x[7:],.15*Rotation.from_matrix(rot@R.T).as_rotvec()]
            def jac(x):
                _,rot,jp,ja=k.fast_pose(x[:7]);phi=Rotation.from_matrix(rot@R.T).as_rotvec();theta=np.linalg.norm(phi);a,b,c=phi;skew=np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])
                v=1/12 if theta<1e-5 else (1-.5*theta/np.tan(.5*theta))/theta**2
                inv=np.eye(3)-.5*skew+v*skew@skew
                return np.block([[jp,-np.eye(3)],[.15*inv@ja,np.zeros((3,3))]])
            x0=np.r_[q,np.clip(k.pose(q)[0]-p,-bound+1e-8,bound-1e-8)]
            fit=least_squares(fun,x0,bounds=(np.r_[k.lower+pad,-bound],np.r_[k.upper-pad,bound]),jac=jac,max_nfev=180,gtol=1e-10,ftol=1e-10,xtol=1e-10)
            q=fit.x[:7];d=fit.x[7:];actual,rot=k.pose(q);entry=dict(translation_m=d.tolist(),q=q.tolist(),position_error_m=float(np.linalg.norm(actual-p-d)),orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(rot@R.T).magnitude())),joint_margin_deg=margin(k,q));attempts.append(entry)
            if not good(entry):counts['translated_ik_fail']+=1;continue
            counts['translated_ik_pass']+=1
            moved=translated(load(n),d,state['particles_world']);checker=CollisionScene(k,moved,state['triangles'])
            for gap in gaps:
                end=checker.check(q,gap);entry['endpoint_collision']=end
                if not end['passed']:counts['endpoint_collision_fail']+=1;continue
                # Follow backwards in Cartesian space, retaining one IK branch.
                qs=[q];failure=None
                for distance in np.linspace(.0025,.05,20):
                    sol=k.solve(p+d-R[:,0]*distance,R,qs[-1]);qq=np.array(sol['q'])
                    if not good(sol) or margin(k,qq)<5:failure='approach_ik_or_margin';break
                    if np.max(abs(qq-qs[-1]))>np.deg2rad(10):failure='approach_branch_jump';break
                    check=checker.check(qq,gap)
                    if not check['passed']:failure='approach_collision';break
                    qs.append(qq)
                if failure:counts[failure]+=1;entry['approach_failure']=failure;continue
                closure=[checker.check(q,float(g),True) for g in np.linspace(gap,.001,21)]
                if not all(v['passed'] for v in closure):counts['closure_collision_fail']+=1;continue
                # Straight joint-space route from corrected start; densely sampled.
                path=[];pre=qs[-1];steps=max(2,int(np.ceil(np.max(abs(pre-k.start))/np.deg2rad(.5))))
                for t in np.linspace(0,1,steps+1):
                    qq=k.start*(1-t)+pre*t;check=checker.check(qq,gap)
                    if not check['passed']:break
                    path.append(qq.tolist())
                route_pass=len(path)==steps+1
                counts['approach_path_pass']+=1;counts['start_route_pass' if route_pass else 'start_route_collision']+=1
                accepted.append(dict(translation_m=d.tolist(),gap_m=gap,endpoint_q=q.tolist(),approach_q=[v.tolist() for v in qs[::-1]],start_route_pass=route_pass,start_route_q=path if route_pass else [],closure_checks=closure,minimum_joint_margin_deg=min(margin(k,v) for v in qs)))
                break
            if accepted:break
        row=dict(source=source,number=n,attempts=attempts,accepted=accepted);results.append(row)
        (OUT/f'target-{index:03d}.json').write_text(json.dumps(row,indent=2))
        print(index+1,len(targets),dict(counts),flush=True)
    aggregate=Counter()
    for row in results:
        aggregate['targets']+=1
        aggregate['trials']+=len(row['attempts'])
        valid=[a for a in row['attempts'] if good(a)]
        aggregate['endpoint_ik_trials']+=len(valid)
        aggregate['targets_with_endpoint_ik']+=bool(valid)
        aggregate['targets_with_approach']+=bool(row['accepted'])
        aggregate['targets_with_start_route']+=any(a['start_route_pass'] for a in row['accepted'])
    (OUT/('summary.json' if args.shards==1 else f'summary-shard-{args.shard_index}.json')).write_text(json.dumps(dict(aggregate=dict(aggregate),counts_scope='Counters for current invocation only; validation.json recomputes all cached and new trials',counts=dict(counts),targets=len(targets),seeds=args.seeds,translation_bounds_m=bound.tolist(),results=[dict(source=r['source'],number=r['number'],accepted=r['accepted']) for r in results],executed=False,continuous_collision_certified=False),indent=2))
if __name__=='__main__':main()
