#!/usr/bin/env python3
"""Plot measured circumference contraction and gravitational retention."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args=p.parse_args();folder=args.directory
    report=json.loads((folder/'report.json').read_text());rows=report['telemetry']
    t=np.array([r['physics_time_s'] for r in rows]);t-=t[0]
    circumference=np.array([r['cuff_perimeter_m'] for r in rows])*100
    forces=np.array([r['cuff_force_median_n'] for r in rows])
    hold=np.array([r['phase']=='gravity_hold' for r in rows])
    first=int(np.flatnonzero(hold)[0]);last=rows[-11:]
    centers=np.array([r['cuff_center_world'] for r in last])
    points=np.array([r['cuff_particles_world'] for r in last])
    tail_t=np.array([r['physics_time_s'] for r in last]);tail_t-=tail_t[0]
    fig,axes=plt.subplots(3,1,figsize=(10,9),constrained_layout=True)
    axes[0].plot(t,circumference,label='Actual cuff circumference')
    if 'initial' in report:
        natural=np.array([r['rest_circumference_scale'] for r in rows])*report['initial']['cuff_perimeter_m']*100
        axes[0].plot(t,natural,'--',label='Natural cuff circumference')
    axes[0].set(ylabel='Circumference (cm)',xlabel='Simulated time (s)')
    axes[0].legend()
    axes[1].plot(t,forces,label='Median hoop tension')
    axes[1].set(ylabel='Model force (N)',xlabel='Simulated time (s)')
    for axis in axes[:2]:
        axis.axvspan(t[first],t[-1],color='green',alpha=.10,label='Full gravity')
        axis.axvline(t[first],color='green',linestyle=':')
        axis.grid(alpha=.25)
    center_drift=np.linalg.norm(centers-centers[0],axis=1)*1000
    point_motion=np.max(np.linalg.norm(points-points[0],axis=2),axis=1)*1000
    axes[2].plot(tail_t,center_drift,label='Cuff center drift')
    axes[2].plot(tail_t,point_motion,label='Largest cuff particle displacement')
    axes[2].axhline(5,color='gray',linestyle='--',label='5 mm bound')
    axes[2].set(ylabel='Displacement (mm)',xlabel='Last second of gravity hold (s)')
    axes[2].grid(alpha=.25);axes[2].legend()
    fig.suptitle(f"Stage {report['stage']}: {'PASS' if report['passed'] else 'NOT ACCEPTED'}; no gripper pins; 50 g; gravity 9.81 m/s^2")
    fig.savefig(folder/'retention.png',dpi=160);plt.close(fig)
    print(folder/'retention.png')


if __name__=='__main__':main()
