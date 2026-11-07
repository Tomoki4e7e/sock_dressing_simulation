"""Static pregrasp screening: cloth collision and finite opposing pad footprint.

This is a sampled finite-patch test, not proof of capture or retention.
"""
import numpy as np
import fcl
from scipy.spatial import ConvexHull


def ray_interval(planes,x,z):
    a=planes[:,1];b=-(planes[:,0]*x+planes[:,2]*z+planes[:,3])
    flat=np.abs(a)<1e-10
    if np.any(b[flat]<-1e-9):return None
    low=np.max(b[a< -1e-10]/a[a< -1e-10]);high=np.min(b[a>1e-10]/a[a>1e-10])
    return (float(low),float(high)) if low<=high else None


def pad_patch(vertices,faces,finger_vertices,*,half_extent_m=.004,minimum_area_m2=4e-6,edge_clearance_m=.0005):
    """Cast a 5x5 grid through the true convex finger cross-sections.

    Every counted cloth point must lie between facing surfaces of both finite
    fingers, with clearance. Require a two-dimensional patch, not one point.
    x is approach, y closing, z jaw width; origin is the calibrated TCP.
    """
    hulls=[ConvexHull(v).equations for v in finger_vertices]
    centers=[v[:,1].mean() for v in finger_vertices];negative=int(np.argmin(centers));positive=1-negative
    triangles=vertices[faces];xz=triangles[:,:,[0,2]];hits=[]
    for x in np.linspace(-half_extent_m,half_extent_m,5):
        for z in np.linspace(-half_extent_m,half_extent_m,5):
            intervals=[ray_interval(h,x,z) for h in hulls]
            if any(i is None for i in intervals):continue
            lower=intervals[negative][1];upper=intervals[positive][0]
            if upper-lower<=2*edge_clearance_m:continue
            found=False
            for face,uv in zip(triangles,xz):
                matrix=np.column_stack([uv[1]-uv[0],uv[2]-uv[0]])
                if abs(np.linalg.det(matrix))<1e-14:continue
                b,c=np.linalg.solve(matrix,np.array([x,z])-uv[0]);weights=np.array([1-b-c,b,c])
                if weights.min()<-1e-7:continue
                y=float(weights@face[:,1])
                if lower+edge_clearance_m<=y<=upper-edge_clearance_m:found=True;break
            if found:hits.append([float(x),float(z)])
    area=0.
    if len(hits)>=3 and np.linalg.matrix_rank(np.asarray(hits)-hits[0],tol=1e-9)==2:
        area=float(ConvexHull(hits).volume)
    # Count fully supported grid cells, rather than bridging separated hits
    # with their convex hull. This remains an explicit sampled criterion.
    step=half_extent_m/2
    indices={tuple(np.rint((np.asarray(h)+half_extent_m)/step).astype(int)) for h in hits}
    cells=sum(all((i+di,j+dj) in indices for di,dj in ((0,0),(1,0),(0,1),(1,1))) for i in range(4) for j in range(4))
    supported_area=cells*step*step
    return dict(passed=supported_area+1e-12>=minimum_area_m2,supported_cell_area_m2=supported_area,supported_cells=cells,grid_hits=len(hits),projected_patch_hull_area_m2=area,
        hits_xz_m=hits,half_extent_m=half_extent_m,minimum_area_m2=minimum_area_m2,edge_clearance_m=edge_clearance_m,
        method='5x5 opposing actual convex finger ray sections; fully supported grid cell area')


class ClothScreen:
    def __init__(self,points,triangles):
        self.points=np.asarray(points,dtype=float);self.faces=np.asarray(triangles,dtype=np.int32).reshape(-1,3)
        mesh=fcl.BVHModel();mesh.beginModel(len(self.faces),len(self.points));mesh.addSubModel(self.points,self.faces);mesh.endModel()
        self.mesh=mesh;self.obj=fcl.CollisionObject(mesh)

    def intersections(self,parts,R,p):
        hits=[]
        for part in parts:
            part.obj.setTransform(fcl.Transform(R,p))
            if fcl.collide(part.obj,self.obj,fcl.CollisionRequest(),fcl.CollisionResult()):hits.append(part.label)
        return hits

    def inspect(self,hand,gap,R,p,**settings):
        hits=self.intersections(hand.bank(gap),R,p)
        if hits:return dict(passed=False,reason='hand_cloth_intersection',intersecting_parts=hits)
        # Limit work to the local target footprint; collision always uses full cloth.
        local=(self.points-p)@R
        half=settings.get('half_extent_m',.004)
        # Triangle bounding-box overlap also covers large triangles spanning the footprint.
        uv=local[self.faces][:,:,[0,2]]
        eligible=np.all(uv.min(1)<=half,axis=1)&np.all(uv.max(1)>=-half,axis=1)
        fingers=[v for name,v in hand.vertices(gap).items() if name.endswith('/left_finger_link') or name.endswith('/right_finger_link')]
        patch=pad_patch(local,self.faces[eligible],fingers,**settings)
        return dict(passed=patch['passed'],reason='accepted' if patch['passed'] else 'insufficient_pad_patch',patch=patch)
