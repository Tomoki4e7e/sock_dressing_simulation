#!/usr/bin/env python3
"""Reconstruct cooked convex supporting planes with adaptive local ray probes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull,HalfspaceIntersection
import trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_compound_geometry import conditioned_part,vector
from audit_compound_rays import ray_interval
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.convex_distance import point_convex_distance


def solid_vertex_distance(vertices,other):
    """Distance to a closed convex solid, zero for interior vertices."""
    return point_convex_distance(vertices,other)


def reconstruct(row,shape,normal_decimals=7):
    """Fit observed faces; stable means sampled checks pass, not full recovery.

    Tiny unseen faces and finite precision can leave an incorrect closure even
    after witnesses pass. Prefer serialized native planes for a final gate.
    """
    hits=[r for r in row['rays'] if r['hit']];points=np.array([r['point'] for r in hits]);normals=np.array([r['normal'] for r in hits]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    if len(points)<4:raise ValueError('Fewer than four ray hits')
    # PhysX reports a single face normal per convex hit. Group repeated normals
    # and estimate each plane from multiple observations of that same face.
    groups={}
    for i,n in enumerate(normals):groups.setdefault(tuple(np.round(n,normal_decimals)),[]).append(i)
    planes=[]
    for ids in groups.values():
        n=normals[ids].mean(0);n/=np.linalg.norm(n);planes.append(np.r_[n,-np.median(points[ids]@n)])
    planes=np.array(planes);fit=linprog([0,0,0,-1],A_ub=np.c_[planes[:,:3],np.ones(len(planes))],b_ub=-planes[:,3],bounds=[(None,None)]*3+[(0,None)],method='highs')
    if not fit.success:raise ValueError('Plane recovery LP: '+fit.message)
    if fit.x[3]<1e-10:raise ValueError('Recovered planes have no strict interior')
    vertices=HalfspaceIntersection(planes,fit.x[:3]).intersections;vertices=vertices[ConvexHull(vertices).vertices];scales=np.array([shape['scale'][a] for a in 'xyz']);original=np.array(row['vertices']);physical=vertices*scales;expected=original*scales
    outside_input=solid_vertex_distance(physical,expected);outside_cooked=solid_vertex_distance(expected,physical)
    # Convert plane residuals to physical perpendicular distances under scale.
    physical_norm=np.linalg.norm(planes[:,:3]/scales,axis=1);support_residual=(points@planes[:,:3].T+planes[:,3])/physical_norm
    maximum_support_violation=float(np.maximum(support_residual,0).max())
    checks=[];target_hits=[r for r in hits if r['kind']=='reconstructed_vertex'];targets=np.array([[v[a] for a in 'xyz'] for v in shape.get('targets',[])])
    target_rays=[r for r in row['rays'] if r['kind']=='reconstructed_vertex'];witness_groups={};direct_errors=[]
    references=np.array([[v[a] for a in 'xyz'] for v in shape.get('targetReferences',shape.get('targets',[]))]);groups=shape.get('targetGroups',list(range(len(targets))))
    for target,reference,group,ray in zip(targets,references,groups,target_rays):
        error=float(np.linalg.norm((np.array(ray['point'])-reference)*scales)) if ray['hit'] else 1.
        witness_groups[group]=min(witness_groups.get(group,float('inf')),error)
        if np.array_equal(target,reference):direct_errors.append(error)
    witness_errors=list(witness_groups.values())
    for ray in row['rays']:
        if ray['kind']!='independent':continue
        direction=np.array(ray['direction'],dtype=float);direction/=np.linalg.norm(direction);interval=ray_interval(planes,np.array(ray['origin']),direction)
        error=None if interval is None or not ray['hit'] else abs(ray['distance']-interval[0])*np.linalg.norm(direction*scales)
        checks.append(dict(hit=ray['hit'],expected_hit=interval is not None,error_m=None if error is None else float(error)))
    return dict(index=row['index'],variant=row['variant'],reconstructed_vertices=vertices.tolist(),planes=planes.tolist(),plane_count=len(planes),hit_count=len(hits),miss_count=len(row['rays'])-len(hits),maximum_support_violation_m=maximum_support_violation,maximum_reconstructed_vertex_witness_error_m=max(witness_errors,default=None),maximum_exact_vertex_ray_error_m=max(direct_errors,default=None),independent_ray_mismatches=sum(r['hit']!=r['expected_hit'] for r in checks),maximum_independent_ray_error_m=max((r['error_m'] for r in checks if r['error_m'] is not None),default=None),input_outside_reconstructed_m=float(outside_cooked.max()),reconstructed_outside_input_m=float(outside_input.max()),reconstructed_solid_hausdorff_m=float(max(outside_input.max(),outside_cooked.max())),stable=bool(witness_errors and max(witness_errors)<1e-6 and maximum_support_violation<1e-6 and all(r['hit']==r['expected_hit'] and r['error_m'] is not None and r['error_m']<1e-5 for r in checks)))


def witness_targets(vertices,scales,near_faces=False):
    targets=[];references=[];groups=[];vertices=np.array(vertices);hull=ConvexHull(vertices)
    for i,vertex in enumerate(vertices):
        candidates=[vertex]
        if near_faces:
            for face in hull.simplices:
                if i not in face:continue
                delta=vertices[face].mean(0)-vertex;distance=np.linalg.norm(delta*scales)
                if distance>0:candidates.append(vertex+delta*min(.05,2.5e-7/distance))
        for candidate in candidates:targets.append(vector(candidate));references.append(vector(vertex));groups.append(i)
    return dict(targets=targets,targetReferences=references,targetGroups=groups)


def support_seeds(vertices,stretch):
    """Aim within every native vertex normal cone in the query's metric."""
    vertices=np.array(vertices);query_vertices=vertices*stretch;hull=ConvexHull(query_vertices);seeds=[]
    for index in hull.vertices:
        faces=np.any(hull.simplices==index,axis=1)
        normals=np.unique(np.round(hull.equations[faces,:3],8),axis=0)
        direction=normals.sum(0);direction/=np.linalg.norm(direction)
        seeds.append(dict(anchor=vector(vertices[index]),direction=vector(direction/stretch)))
    return seeds


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--indices',type=int,nargs='+');ap.add_argument('--all',action='store_true');ap.add_argument('--variants',nargs='+',default=['uniform100','pca1']);ap.add_argument('--rounds',type=int,default=3);ap.add_argument('--normal-decimals',type=int,default=7);ap.add_argument('--seed-summary',type=Path);ap.add_argument('--face-witnesses',action='store_true');ap.add_argument('--contact-support',action='store_true');ap.add_argument('--support-stretch',action='store_true');ap.add_argument('--seed-cooked',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/compound-reaching/followup/cooked-plane-cycles');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    originals=json.loads((ROOT/'artifacts/compound-reaching/native-compound-payload.json').read_text())['parts'];previous=json.loads((ROOT/'artifacts/compound-reaching/followup/remaining-vertex-rays.json').read_text());audit=json.loads((ROOT/'artifacts/compound-reaching/followup/ray-audit.json').read_text());indices=args.indices if args.indices is not None else list(range(len(originals))) if args.all else sorted(set([r['index'] for r in previous['checks']]+[r['index'] for r in audit['parts'] if r['counts']['physx_misses']]))
    shapes=[conditioned_part(originals[i],i,v) for i in indices for v in args.variants]
    if args.seed_cooked:
        cooked={p['name']:p for p in json.loads(args.seed_cooked.read_text())['parts']}
        for shape in shapes:
            vertices=np.array([[p[a] for a in 'xyz'] for p in shape['vertices']]);stretch=1/np.ptp(vertices,axis=0) if args.support_stretch else np.ones(3)
            shape['supportSeeds']=support_seeds(cooked[f"cook_{shape['index']}_{shape['variant']}"]['vertices'],stretch)
    if args.seed_summary:
        seeds={(r['index'],r['variant']):r for r in json.loads(args.seed_summary.read_text())['parts']}
        for shape in shapes:
            seed=seeds.get((shape['index'],shape['variant']),{})
            if 'reconstructed_vertices' in seed:shape.update(witness_targets(seed['reconstructed_vertices'],np.array([shape['scale'][a] for a in 'xyz']),args.face_witnesses))
    if not args.all:
        for index,half in [(-1,[.005,.005,.005]),(-2,[.005,.005,.00005])]:
            original=dict(link='right_arm/link_5',vertices=[vector(np.array([x,y,z])*half) for x in [-1,1] for y in [-1,1] for z in [-1,1]])
            shapes.extend(conditioned_part(original,index,v) for v in ['box','uniform100','pca1'])
    for shape in shapes:shape.update(measureSupport=args.contact_support,supportStretch=args.support_stretch)
    config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');config.pop('pose_pattern',None);config['rcareworld'].update(executable='scripts/run_compound_reaching_player.sh',graphics=True,port=15970,log_level=0);config['assets']['output_dir']=str(args.output/'assets');scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario);rng=np.random.default_rng(20261104)
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),Path(config['assets']['output_dir'])/'sock.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario)
        final=[]
        for cycle in range(args.rounds):
            payload=copy.deepcopy(shapes)
            if cycle:
                for shape in payload:
                    directions=rng.normal(size=(256,3));directions/=np.linalg.norm(directions,axis=1)[:,None];shape['directions']=[vector(d) for d in directions]
            (args.output/f'cycle-{cycle}.payload.json').write_text(json.dumps(dict(parts=payload)));raw=[];results=[]
            for offset in range(0,len(payload),16):
                env.sock_cloth._send_data('ReadSockCompoundCookedPlanes',json.dumps(dict(parts=payload[offset:offset+16])));env._env.step(3,simulate=False);batch=env.sock_cloth.data['sock_compound_cooked_planes']['parts'];raw.extend(batch)
                for row,shape in zip(batch,payload[offset:offset+16]):
                    try:result=reconstruct(row,shape,args.normal_decimals)
                    except Exception as error:result=dict(index=row['index'],variant=row['variant'],stable=False,error=str(error))
                    results.append(result)
                print('planes cycle',cycle,'parts',len(results),len(payload),'stable',sum(r['stable'] for r in results),flush=True)
            (args.output/f'cycle-{cycle}.raw.json').write_text(json.dumps(dict(parts=raw,physics_executed=False)));(args.output/f'cycle-{cycle}.summary.json').write_text(json.dumps(dict(parts=results,physics_executed=False),indent=2));final=results
            for shape,result in zip(shapes,results):
                if 'reconstructed_vertices' in result:shape.update(witness_targets(result['reconstructed_vertices'],np.array([shape['scale'][a] for a in 'xyz']),args.face_witnesses))
        summary=dict(parts=final,cycles=args.rounds,normal_decimals=args.normal_decimals,face_witnesses=args.face_witnesses,seed_summary=str(args.seed_summary),stable_parts=sum(r['stable'] for r in final),shape_discrepancies_over_01mm=sum(r.get('reconstructed_solid_hausdorff_m',0)>1e-4 for r in final if r['stable']),scope='Sampled ray plane estimate, not native contact geometry; stable does not prove every face was recovered',physics_executed=False,operational_geometry_changed=False)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('plane cycles done',summary['stable_parts'],'/',len(final),'discrepancies',summary['shape_discrepancies_over_01mm'],flush=True)


if __name__=='__main__':main()
