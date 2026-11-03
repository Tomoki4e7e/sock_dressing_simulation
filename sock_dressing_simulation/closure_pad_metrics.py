"""Finite-pad proximity from recorded cloth faces (geometry proxy, not force)."""
import numpy as np
from scipy.spatial import ConvexHull
from .grasp_cloth_geometry import ray_interval


def sections(hand,gap,half=.004):
 fingers=[v for name,v in hand.vertices(gap).items() if name.endswith('/left_finger_link') or name.endswith('/right_finger_link')]
 fingers=sorted(fingers,key=lambda v:v[:,1].mean());hulls=[ConvexHull(v).equations for v in fingers]
 rays=np.array([[x,z] for x in np.linspace(-half,half,5) for z in np.linspace(-half,half,5)])
 limits=[]
 for x,z in rays:
  a,b=[ray_interval(h,x,z) for h in hulls]
  limits.append([a[1],b[0]] if a is not None and b is not None else [np.nan,np.nan])
 return rays,np.asarray(limits)


def local_pad_proximity(local,faces,section,envelope=.0025):
 rays,limits=section;t=local[faces];uv=t[:,:,[0,2]];e1=uv[:,1]-uv[:,0];e2=uv[:,2]-uv[:,0]
 det=e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0];good=np.abs(det)>1e-14
 if not good.any():return dict(negative_near=False,positive_near=False,minimum_negative_distance_m=None,minimum_positive_distance_m=None)
 t=t[good];uv=uv[good];e1=e1[good];e2=e2[good];det=det[good]
 d=rays[:,None,:]-uv[None,:,0,:]
 b=(d[:,:,0]*e2[None,:,1]-d[:,:,1]*e2[None,:,0])/det
 c=(e1[None,:,0]*d[:,:,1]-e1[None,:,1]*d[:,:,0])/det
 valid=(b>=-1e-7)&(c>=-1e-7)&(b+c<=1+1e-7)&np.isfinite(limits[:,0,None])
 y=t[None,:,0,1]+b*(t[None,:,1,1]-t[None,:,0,1])+c*(t[None,:,2,1]-t[None,:,0,1])
 distances=[np.where(valid,np.abs(y-limits[:,i,None]),np.inf).min() for i in (0,1)]
 return dict(negative_near=bool(distances[0]<=envelope),positive_near=bool(distances[1]<=envelope),
  minimum_negative_distance_m=float(distances[0]) if np.isfinite(distances[0]) else None,
  minimum_positive_distance_m=float(distances[1]) if np.isfinite(distances[1]) else None)
