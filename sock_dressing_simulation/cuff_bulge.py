"""Material-coordinate cuff refinement and a bounded radial bulge.

Refinement preserves the original piecewise-linear surface exactly before
deformation. Radius is measured about the ORIGINAL section centreline in its
normal plane; this is well-defined for the existing noncircular sock.
"""
import numpy as np


def refine_sections(rest,points,faces,depths=(.0075,.015,.0225,.030)):
    rest=np.asarray(rest,dtype=float);points=np.asarray(points,dtype=float)
    rr=list(rest.copy());pp=list(points.copy());minimum=rest[:,2].min()
    indices={tuple(np.round(v,11)):i for i,v in enumerate(rest)}
    def intersect(a,b,level):
        ra,rb=rr[a],rr[b];weight=(minimum+level-ra[2])/(rb[2]-ra[2])
        if weight<1e-9:return a
        if weight>1-1e-9:return b
        r=ra+(rb-ra)*weight;key=tuple(np.round(r,11))
        if key not in indices:
            indices[key]=len(rr);rr.append(r);pp.append(pp[a]+(pp[b]-pp[a])*weight)
        return indices[key]
    def clip(poly,level,below):
        result=[];a=poly[-1];da=(rr[a][2]-minimum-level)*(1 if below else -1)
        for b in poly:
            db=(rr[b][2]-minimum-level)*(1 if below else -1)
            if (da<=1e-12)!=(db<=1e-12):result.append(intersect(a,b,level))
            if db<=1e-12:result.append(b)
            a,da=b,db
        unique=[]
        for i in result:
            if not unique or unique[-1]!=i:unique.append(i)
        if len(unique)>1 and unique[0]==unique[-1]:unique.pop()
        return unique
    refined=[]
    for face in np.asarray(faces).reshape(-1,3):
        polygons=[face.tolist()]
        for level in depths:
            next_polygons=[]
            for poly in polygons:
                z=np.array([rr[i][2]-minimum for i in poly])
                if z.min()<level-1e-10 and z.max()>level+1e-10:
                    next_polygons.extend([clip(poly,level,True),clip(poly,level,False)])
                else:next_polygons.append(poly)
            polygons=next_polygons
        for poly in polygons:
            for j in range(1,len(poly)-1):
                tri=[poly[0],poly[j],poly[j+1]]
                if len(set(tri))==3:refined.append(tri)
    return np.asarray(rr),np.asarray(pp),np.asarray(refined,dtype=int)


def radial_bulge(original_rest,original_points,refined_rest,refined_points,*,width=.030,peak=.005):
    if not np.isfinite(width) or not np.isfinite(peak) or width<=0 or peak<=0:raise ValueError('Positive finite width and radius required')
    rest=np.asarray(original_rest);points=np.asarray(original_points);depth=rest[:,2]-rest[:,2].min()
    levels=np.unique(np.round(depth,8));centers=np.array([points[np.isclose(depth,z,atol=1e-7,rtol=0)].mean(0) for z in levels])
    z=np.asarray(refined_rest)[:,2]-rest[:,2].min()
    center=np.column_stack([np.interp(z,levels,centers[:,i]) for i in range(3)])
    gradients=np.gradient(centers,levels,axis=0)
    tangent=np.column_stack([np.interp(z,levels,gradients[:,i]) for i in range(3)])
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
    relative=np.asarray(refined_points)-center
    radial=relative-tangent*np.sum(relative*tangent,axis=1)[:,None]
    radius=np.linalg.norm(radial,axis=1);active=(z>1e-10)&(z<width-1e-10)
    if np.any(radius[active]<1e-8):raise ValueError('Degenerate cuff radius')
    amount=np.zeros(len(z));amount[active]=peak*np.sin(np.pi*z[active]/width)**2
    moved=np.asarray(refined_points).copy();moved[active]+=radial[active]/radius[active,None]*amount[active,None]
    return moved,dict(depth_m=z.tolist(),radius_increment_m=amount.tolist(),reference_center_world=center.tolist(),reference_axis_world=tangent.tolist())
