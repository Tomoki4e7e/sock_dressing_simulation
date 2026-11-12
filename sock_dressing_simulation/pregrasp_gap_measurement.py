"""Clip actual cloth triangles to the finite, measured opposing jaw aperture."""
import numpy as np
from .contact_pregrasp import aperture_planes,rim_intervals,opening_edges
from .geometric_closure import clip_polygon


def outside_union(polygons,planes):
    # Disjoint first-satisfied partition of the union of outside halfspaces.
    result=[]
    for polygon in polygons:
        remaining=polygon
        for plane in planes:
            if len(remaining)<3:break
            outside=clip_polygon(remaining,-plane)
            if len(outside)>=3:result.append(outside)
            remaining=clip_polygon(remaining,plane)
    return result


def area(polygon):
    return sum(np.linalg.norm(np.cross(polygon[i,:3]-polygon[0,:3],polygon[i+1,:3]-polygon[0,:3]))*.5 for i in range(1,len(polygon)-1))


def measure(points,faces,material_depth,fingers,rim):
    fingers=sorted(fingers,key=lambda v:v[:,1].mean());hulls,projected=aperture_planes(*fingers)
    lo=np.maximum(fingers[0][:,[0,2]].min(0),fingers[1][:,[0,2]].min(0));hi=np.minimum(fingers[0][:,[0,2]].max(0),fingers[1][:,[0,2]].max(0))
    triangles=points[faces];uv=triangles[:,:,[0,2]];possible=(uv.min(1)<=hi).all(1)&(uv.max(1)>=lo).all(1)
    captured=[];depths=[];total=band=0.
    if (lo<=hi).all():
        for face in faces[possible]:
            polygon=np.c_[points[face],material_depth[face]]
            for plane in projected:
                polygon=clip_polygon(polygon,np.r_[plane[:3],0.,plane[3]])
                if len(polygon)<3:break
            if len(polygon)<3:continue
            lower=np.c_[hulls[0][hulls[0][:,1]>1e-9,:3],np.zeros(sum(hulls[0][:,1]>1e-9)),hulls[0][hulls[0][:,1]>1e-9,3]]
            upper=np.c_[hulls[1][hulls[1][:,1]<-1e-9,:3],np.zeros(sum(hulls[1][:,1]<-1e-9)),hulls[1][hulls[1][:,1]<-1e-9,3]]
            pieces=outside_union(outside_union([polygon],lower),upper)
            for part in pieces:
                value=area(part)
                if value<=1e-14:continue
                total+=value;captured.append(part[:,:3].tolist());depths.extend(part[:,3].tolist())
                bounded=clip_polygon(part,np.array([0.,0.,0.,-1.,.015]));bounded=clip_polygon(bounded,np.array([0.,0.,0.,1.,-.030])) if len(bounded)>=3 else bounded
                if len(bounded)>=3:band+=area(bounded)
    rim_parts=rim_intervals(rim,*fingers);length=sum(p['length_m'] for p in rim_parts)
    rim_total=float(np.linalg.norm(rim[:,1]-rim[:,0],axis=1).sum())
    return dict(cloth_surface_area_between_fingers_mm2=float(total*1e6),material_15_30mm_band_area_between_fingers_mm2=float(band*1e6),opening_rim_length_between_fingers_mm=float(length*1000),total_opening_rim_length_mm=rim_total*1000,opening_rim_fraction_percent=length/rim_total*100 if rim_total else 0.,material_depth_range_mm=None if not depths else [float(min(depths)*1000),float(max(depths)*1000)],cloth_present=bool(total>1e-12),opening_rim_present=bool(length>=1e-6),captured_polygons_local_m=captured,rim_segments=rim_parts,method='Actual cloth triangle surface area clipped to shared finite convex-jaw footprints and opposing jaw envelopes; no infinite parallel-plane approximation',scope='Geometric occupancy only; does not establish grip closure, frictional retention or pulling success')
