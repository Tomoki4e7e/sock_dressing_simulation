#!/usr/bin/env python3
"""Review tables and figures for the skin/proxy grasp-geometry comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def plane_segments(vertices,faces,x):
    segments=[]
    for triangle in np.asarray(vertices)[np.asarray(faces).reshape(-1,3)]:
        hits=[]
        for i,j in ((0,1),(1,2),(2,0)):
            a,b=triangle[i],triangle[j]
            if (a[0]-x)*(b[0]-x)<0:hits.append((a+(b-a)*(x-a[0])/(b[0]-a[0]))[1:])
        if len(hits)==2:segments.append(hits)
    return np.asarray(segments)


def plot_reports(output,reports,example):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    from sock_dressing_simulation.grasp_geometry import RightHand,ConvexFootProxy,SkinBody
    import fcl
    fig,axes=plt.subplots(8,4,figsize=(14,19),sharex=True,sharey=True)
    for ax,r in zip(axes.ravel(),reports):
        values=np.zeros((7,32))
        for s in r['samples']:
            if not s['native_contact_face']:continue
            opening=s.get('best_orientation',{}).get('maximum_checked_approach_opening')
            if opening:values[round(s['depth_m']/.005),s['theta_index']]=opening['gap_m']*1000
        values[6]=np.nan
        cm=plt.get_cmap('viridis').copy();cm.set_under('white');cm.set_bad('lightgray')
        mesh=ax.imshow(values,extent=[-5.625,354.375,32.5,-2.5],vmin=4.99,vmax=50,cmap=cm,aspect='auto',interpolation='nearest')
        ax.set_title(f'No.{r["number"]}  pitch {r["pitch_deg"]:g}, roll {r["roll_deg"]:g}',fontsize=9)
        ax.set_xticks([0,90,180,270]);ax.set_yticks([0,10,20,30])
    for ax in axes[-1]:ax.set_xlabel('Material theta [deg]')
    for ax in axes[:,0]:ax.set_ylabel('Depth [mm]')
    fig.suptitle('Actual-skin-clear samples: maximum checked approach opening\nGray: inactive boundary faces. Current convex foot proxy rejects all candidates.',fontsize=12)
    fig.subplots_adjust(top=.94,bottom=.06,left=.06,right=.91,hspace=.55,wspace=.25)
    fig.colorbar(mesh,cax=fig.add_axes([.93,.10,.015,.78]),label='Opening [mm]')
    fig.savefig(output/'opening-map.png',dpi=150);plt.close(fig)

    r,c=example;state=json.loads((ROOT/r['input_state']).read_text());settings=r['settings']
    hand=RightHand(ROOT/settings['urdf'],settings['tip_inset_m'],settings['maximum_gap_m']);parts=hand.bank(settings['nominal_pinch_gap_m'])
    proxy=ConvexFootProxy(state['source']['surfaces']);body=SkinBody(state['source']['skin_surface'])
    rotation=np.asarray(c['rotation_matrix']);point=np.asarray(c['grasp_center_unity_world_m'])
    contact=None
    for p in parts:
        p.obj.setTransform(fcl.Transform(rotation,point));result=fcl.CollisionResult()
        if fcl.collide(p.obj,proxy.objects[0],fcl.CollisionRequest(num_max_contacts=1,enable_contact=True),result):
            contact=result.contacts[0];break
    if contact is None:raise ValueError('Example does not intersect the foot proxy')
    plane_x=float(((contact.pos-point)@rotation)[0])
    skin_vertices=(np.asarray(state['source']['skin_surface']['vertices'])-point)@rotation
    skin_faces=np.asarray(state['source']['skin_surface']['triangles']).reshape(-1,3)
    foot_vertices=(np.asarray(state['source']['surfaces'][0]['vertices'])-point)@rotation
    foot_faces=np.asarray(state['source']['surfaces'][0]['triangles']).reshape(-1,3)
    fig=plt.figure(figsize=(13,6));ax=fig.add_subplot(121,projection='3d')
    ax.add_collection3d(Poly3DCollection(skin_vertices[skin_faces][:,:,[0,2,1]],facecolor='gray',alpha=.14,edgecolor='none'))
    ax.add_collection3d(Poly3DCollection(foot_vertices[foot_faces][:,:,[0,2,1]],facecolor='crimson',alpha=.08,edgecolor='none'))
    for p in parts:
        ax.add_collection3d(Poly3DCollection(p.vertices[p.faces][:,:,[0,2,1]],facecolor='darkorange',edgecolor='saddlebrown',linewidth=.1,alpha=.5))
    ax.scatter(0,0,0,color='magenta',s=40,label='Grasp center')
    ax.set_xlim(-.31,.06);ax.set_ylim(-.09,.09);ax.set_zlim(-.12,.07);ax.set_box_aspect((.37,.18,.19));ax.view_init(20,-60)
    ax.set_xlabel('Approach X [m]');ax.set_ylabel('Width Z [m]');ax.set_zlabel('Closing Y [m]');ax.legend(fontsize=8)
    ax.set_title('Actual skin: clear / convex foot proxy: intersecting',fontsize=10)
    ax2=fig.add_subplot(122)
    foot_lines=plane_segments(foot_vertices,foot_faces,plane_x)
    poly_points=foot_lines.reshape(-1,2);poly_points=poly_points[ConvexHull(poly_points).vertices]
    ax2.fill(poly_points[:,0]*1000,poly_points[:,1]*1000,color='crimson',alpha=.16,label='Convex foot contact proxy')
    lines=plane_segments(skin_vertices,skin_faces,plane_x)
    ax2.add_collection(LineCollection(lines*1000,colors='black',linewidths=1.5,label='Actual skin section'))
    for p in parts:
        lines=plane_segments(p.vertices,p.faces,plane_x)
        if len(lines)<3:continue
        pts=lines.reshape(-1,2);pts=pts[ConvexHull(pts).vertices]
        ax2.fill(pts[:,0]*1000,pts[:,1]*1000,color='darkorange',alpha=.7,edgecolor='saddlebrown')
    ax2.axvline(0,color='magenta',ls=':',lw=.8);ax2.axhline(0,color='magenta',ls=':',lw=.8)
    ax2.set_xlim(-65,45);ax2.set_ylim(-55,55);ax2.set_aspect('equal');ax2.grid(alpha=.25)
    ax2.set_xlabel('Closing Y [mm]');ax2.set_ylabel('Width Z [mm]')
    ax2.set_title(f'Finger/proxy contact section: X={plane_x*1000:.1f} mm',fontsize=10);ax2.legend(fontsize=8,loc='lower left')
    fig.suptitle(f'No.{r["number"]}: 10 mm physical-tip inset, 5 mm nominal pinch gap, tilt {c["tilt_deg"]} deg',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(output/'foot-proxy-section.png',dpi=170);plt.close(fig)
    sphere=fcl.CollisionObject(fcl.Sphere(.000001),fcl.Transform(np.eye(3),point))
    skin_gap=float(fcl.distance(sphere,body.obj,fcl.DistanceRequest(),fcl.DistanceResult())+.000001)
    proxy_gap=float(fcl.distance(sphere,proxy.objects[0],fcl.DistanceRequest(),fcl.DistanceResult())+.000001)
    details=dict(number=r['number'],sample_index=c['sample_index'],candidate=c,
        material_sample=r['samples'][c['sample_index']],target_to_actual_skin_m=skin_gap,
        target_to_convex_foot_proxy_m=proxy_gap,proxy_contact_penetration_depth_m=float(contact.penetration_depth))
    (output/'example-candidate.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/right-grasp-geometry')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();output=args.output
    summary=json.loads((output/'summary.json').read_text());reports=[];rows=[];tilts=Counter();depths=Counter()
    for t in summary['trials']:
        r=json.loads((output/f'pose-{t["number"]:02d}'/'report.json').read_text());reports.append(r)
        candidates=[c for c in r['candidates'] if c['native_contact_face']]
        approached=[c for c in candidates if c['maximum_checked_approach_opening']]
        shallow=[c for c in approached if abs(c['tilt_deg'])<=30]
        for c in candidates:
            tilts[c['tilt_deg']]+=1;depths[round(r['samples'][c['sample_index']]['depth_m']*1000)]+=1
        pc=r['simulator_proxy_check']['counts']
        rows.append(dict(number=r['number'],pitch_deg=r['pitch_deg'],roll_deg=r['roll_deg'],tested_poses=r['counts']['tested'],
            active_surface_points=r['sampling']['native_contact_surface_points'],skin_clear_poses=len(candidates),skin_clear_approach_poses=len(approached),
            skin_clear_surface_points=r['native_contact_points_with_clear_endpoint'],shallow_approach_poses=len(shallow),
            maximum_shallow_approach_opening_mm=max((c['maximum_checked_approach_opening']['gap_m']*1000 for c in shallow),default=0),
            full_50mm_approach_poses=sum(c['maximum_checked_approach_opening']['gap_m']>=.05 for c in approached),
            current_convex_foot_only_nonintersecting_poses=pc.get('foot_only_no_intersection',0),
            current_foot_calf_proxy_clear_poses=pc.get('proxy_clear_candidates',0)))
    rows.sort(key=lambda r:(r['pitch_deg'],r['roll_deg']))
    with (output/'per-pose-summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary.update(total_native_skin_clear_candidates=sum(r['skin_clear_poses'] for r in rows),
        total_native_skin_clear_approach_candidates=sum(r['skin_clear_approach_poses'] for r in rows),
        native_clear_points_range=[min(r['skin_clear_surface_points'] for r in rows),max(r['skin_clear_surface_points'] for r in rows)],
        native_skin_clear_candidates_range=[min(r['skin_clear_poses'] for r in rows),max(r['skin_clear_poses'] for r in rows)],
        native_skin_clear_counts_by_tilt_deg=dict(tilts),native_skin_clear_counts_by_material_depth_mm=dict(depths),
        table=rows,conclusion='Actual skin permits sampled candidates, but the current convex foot contact proxy intersects all of them at the nominal 5 mm pinch gap.')
    full_grid=output/'full-grid-foot-proxy-summary.json'
    if full_grid.exists():
        full=json.loads(full_grid.read_text())
        summary['full_grid_foot_proxy_tested_poses']=full['tested']
        summary['full_grid_foot_proxy_poses_with_nonintersection']=full['poses_with_native_nonintersection']
        summary['full_grid_foot_proxy_poses_with_1mm_clearance']=full['poses_with_native_1mm_clearance']
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if not args.no_plots:
        r=next(r for r in reports if r['number']==1)
        candidates=[c for c in r['candidates'] if c['native_contact_face'] and abs(c['tilt_deg'])<=30
            and c['maximum_checked_approach_opening'] and c['maximum_checked_approach_opening']['gap_m']>=.010]
        plot_reports(output,reports,(r,max(candidates,key=lambda c:c['endpoint_clearance_m'])))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['settings','hand','trials','table']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
