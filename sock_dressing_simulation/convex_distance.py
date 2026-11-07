"""Convex distance without tolerance-sensitive thin-triangle barycentrics."""
import numpy as np
from scipy.spatial import ConvexHull


def point_convex_distance(points,vertices,*,surface=False):
    points=np.asarray(points,dtype=float);vertices=np.asarray(vertices,dtype=float)
    center=vertices.mean(0);extent=np.ptp(vertices,axis=0).max()
    if extent<=0:raise ValueError('Degenerate convex vertices')
    v=(vertices-center)/extent;p=(points-center)/extent;hull=ConvexHull(v);A=hull.equations[:,:3];b=hull.equations[:,3];signed=p@A.T+b;inside=signed.max(1)<=1e-12;distance=np.zeros(len(p))
    if surface:distance[inside]=np.maximum(0,-signed[inside].max(1))
    outside=np.flatnonzero(~inside)
    if len(outside):
        q=p[outside];s=signed[outside];best=np.full(len(q),np.inf)
        # The closest point is on a face, an edge, or a vertex. Only accept a
        # face projection if it lies in every supporting half-space.
        for normal,offset in zip(A,b):
            d=q@normal+offset;projection=q-d[:,None]*normal
            valid=(projection@A.T+b).max(1)<=1e-11
            best[valid]=np.minimum(best[valid],np.abs(d[valid]))
        edges=np.unique(np.sort(hull.simplices[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1),axis=0)
        for edge in edges:
            a,z=v[edge];direction=z-a;length=direction@direction
            if length==0:continue
            fraction=np.clip((q-a)@direction/length,0,1);projection=a+fraction[:,None]*direction
            best=np.minimum(best,np.linalg.norm(q-projection,axis=1))
        distance[outside]=best
    return distance*extent
