#!/usr/bin/env python3
"""Plot recorded cloth/foot geometry in orthographic diagnostic views."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def plot(record, output):
    safety=record['dressing_qa']['foot_collision_safety']
    hull=next(shape for shape in safety['foot_collider_geometry'] if shape.get('mesh_vertices_world'))
    origin=np.asarray(hull['position']);x,y,z,w=hull['rotation']
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    def local(points):return (np.asarray(points)-origin)@rotation*1000
    foot=local(hull['mesh_vertices_world']);cloth=local(record['particles_world'])
    foot_faces=np.asarray(hull['mesh_triangles'],dtype=int).reshape(-1,3)
    cloth_faces=np.asarray(record['particle_surface_triangles'],dtype=int).reshape(-1,3)
    rim=np.asarray(record['opening_particle_indices'],dtype=int);rim=np.r_[rim,rim[0]]
    figure,axes=plt.subplots(1,3,figsize=(15,5))
    for axis,(horizontal,vertical,title) in zip(axes,[(0,1,'Transverse'),(0,2,'Top'),(2,1,'Side')]):
        axis.add_collection(PolyCollection(foot[foot_faces][:,:,[horizontal,vertical]],facecolor='gray',edgecolor='gray',alpha=.25,linewidth=.2))
        axis.add_collection(PolyCollection(cloth[cloth_faces][:,:,[horizontal,vertical]],facecolor='#008f96',edgecolor='#008f96',alpha=.13,linewidth=.12))
        axis.plot(cloth[rim,horizontal],cloth[rim,vertical],color='#224de7',linewidth=2)
        axis.autoscale_view();axis.set_aspect('equal');axis.set_title(title)
        axis.set_xlabel('foot local '+ 'XYZ'[horizontal]+' (mm)');axis.set_ylabel('foot local '+ 'XYZ'[vertical]+' (mm)')
        axis.grid(alpha=.2)
    figure.suptitle('Recorded geometry, frame '+str(record['frame']+1)+' — gray: foot collider; teal: cloth; blue: opening')
    figure.tight_layout();output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=160);figure.savefig(output.with_suffix('.pdf'));plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('diagnostics',type=Path)
    parser.add_argument('--frame',type=int,help='Zero-based frame; defaults to latest');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();records=[json.loads(line) for line in args.diagnostics.read_text().splitlines() if line]
    record=records[-1] if args.frame is None else next(row for row in records if row['frame']==args.frame)
    plot(record,args.output)
