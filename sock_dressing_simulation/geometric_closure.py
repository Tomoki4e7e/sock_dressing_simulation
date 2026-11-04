"""Force-free closure evaluation using finite real jaw surfaces and cloth area."""
import heapq
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon,box
from shapely.ops import unary_union
from shapely.geometry.polygon import orient


def validate_settings(settings):
    names={'closing_gap_m','body_clearance_m','approach_distance_m','half_extent_m','edge_clearance_m','material_radius_m','band_width_m','minimum_area_m2','minimum_area_fraction','minimum_component_area_m2','minimum_span_m','topology_tolerance_m'}
    if set(settings)!=names or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or v<=0 for v in settings.values()):raise ValueError('Known positive finite geometric parameters required')
    if settings['closing_gap_m']<=2*settings['edge_clearance_m']:raise ValueError('Positive aperture must exceed both cloth-side margins')
    if settings['minimum_area_fraction']>1:raise ValueError('Area fraction exceeds one')
    area=4*settings['half_extent_m']**2
    if max(settings['minimum_area_m2'],settings['minimum_component_area_m2'])>area or settings['minimum_span_m']>2*settings['half_extent_m']:raise ValueError('Required patch exceeds the finite window')
    if settings['topology_tolerance_m']>=min(settings['edge_clearance_m'],settings['minimum_span_m'])/100:raise ValueError('Topology tolerance must stay much smaller than geometric margins')


def clip_polygon(vertices,plane,tolerance=1e-12):
    """Clip an ordered polygon in arbitrary dimensions by a linear halfspace."""
    if len(vertices)==0:return np.empty((0,len(plane)-1))
    result=[];previous=vertices[-1];before=float(previous@plane[:-1]+plane[-1])
    for current in vertices:
        after=float(current@plane[:-1]+plane[-1]);inside=after<=tolerance;was_inside=before<=tolerance
        if inside!=was_inside:
            result.append(previous+(current-previous)*(before/(before-after)))
        if inside:result.append(current)
        previous=current;before=after
    return np.asarray(result) if result else np.empty((0,vertices.shape[1]))


def material_distances(rest,faces,target_face,weights):
    """Rest-mesh edge geodesic, seeded at the requested material point."""
    rest=np.asarray(rest);target=np.asarray(weights)@rest[target_face]
    graph=[{} for _ in rest]
    for face in faces:
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            length=float(np.linalg.norm(rest[a]-rest[b]));graph[a][b]=length;graph[b][a]=length
    distance=np.full(len(rest),np.inf);queue=[]
    for index in target_face:
        distance[index]=np.linalg.norm(rest[index]-target);heapq.heappush(queue,(distance[index],int(index)))
    while queue:
        value,index=heapq.heappop(queue)
        if value>distance[index]:continue
        for neighbor,length in graph[index].items():
            proposed=value+length
            if proposed<distance[neighbor]:distance[neighbor]=proposed;heapq.heappush(queue,(proposed,int(neighbor)))
    if not np.isfinite(distance).all():raise ValueError('Disconnected material mesh')
    return distance


def jaw_cells(finger_vertices,half_extent,clearance):
    """Piecewise convex aperture cells bounded by actual opposing hull facets."""
    fingers=sorted(finger_vertices,key=lambda v:np.mean(v[:,1]));hulls=[ConvexHull(v) for v in fingers]
    window=box(-half_extent,-half_extent,half_extent,half_extent);surfaces=[]
    for i,(vertices,hull) in enumerate(zip(fingers,hulls)):
        selected=[]
        for indices,plane in zip(hull.simplices,hull.equations):
            if (i==0 and plane[1]<=1e-10) or (i==1 and plane[1]>=-1e-10):continue
            polygon=Polygon(vertices[indices][:,[0,2]])
            if polygon.area<=1e-16 or not polygon.intersects(window):continue
            polygon=polygon.intersection(window)
            if polygon.area>1e-16:selected.append((polygon,plane))
        surfaces.append(selected)
    cells=[]
    for a,negative in surfaces[0]:
        for b,positive in surfaces[1]:
            domain=a.intersection(b)
            if domain.area<=1e-16:continue
            lower=-negative/negative[1];lower[3]+=clearance
            upper=positive/positive[1];upper[3]+=clearance
            # Remove places where the facing surfaces cross or leave no gap.
            points=np.asarray(orient(domain,sign=1).exterior.coords)[:-1]
            feasibility=lower+upper
            points3=np.c_[points[:,0],np.zeros(len(points)),points[:,1]]
            points3=clip_polygon(points3,feasibility)
            if len(points3)<3:continue
            domain=Polygon(points3[:,[0,2]])
            if domain.area<=1e-16:continue
            points=np.asarray(orient(domain,sign=1).exterior.coords)[:-1];planes=[]
            for start,end in zip(points,np.roll(points,-1,axis=0)):
                dx,dz=end-start
                planes.append(np.array([dz,0.,-dx,dx*start[1]-dz*start[0]]))
            cells.append((domain,planes+[lower,upper]))
    return cells


def connected_groups(polygons,tolerance):
    """Join numerical edge cracks without dilating the measured union area.

    Point-only contact does not qualify. Buffered overlap must have enough
    extent to indicate a shared edge, rather than a single touching vertex.
    """
    parents=list(range(len(polygons)))
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    buffers=[p.buffer(tolerance) for p in polygons]
    for i,a in enumerate(polygons):
        for j in range(i+1,len(polygons)):
            if a.distance(polygons[j])<=tolerance and buffers[i].intersection(buffers[j]).area>16*tolerance*tolerance:
                parents[root(j)]=root(i)
    groups={}
    for i in range(len(polygons)):groups.setdefault(root(i),[]).append(i)
    return sorted(groups.values(),key=lambda ids:sum(polygons[i].area for i in ids),reverse=True)


def projected_cloth_area(local,faces,fingers,distances,depth,*,half_extent_m=.004,edge_clearance_m=.0001,material_radius_m=.020,band_width_m=.030,topology_tolerance_m=1e-9):
    """Clip cloth in 3D, then take the exact planar polygon union (floating point)."""
    cells=jaw_cells(fingers,half_extent_m,edge_clearance_m)
    if not cells:return dict(area_m2=0.,effective_pad_area_m2=0.,area_fraction=0.,largest_component_area_m2=0.,span_x_m=0.,span_z_m=0.,polygons_xz_m=[])
    triangles=[]
    for face in faces:
        v=local[face]
        if np.any(v[:,[0,2]].min(0)>half_extent_m) or np.any(v[:,[0,2]].max(0)<-half_extent_m):continue
        augmented=np.c_[v,np.asarray(distances)[face],np.asarray(depth)[face]]
        augmented=clip_polygon(augmented,np.array([0,0,0,1,0,-material_radius_m]))
        augmented=clip_polygon(augmented,np.array([0,0,0,0,1,-band_width_m]))
        if len(augmented)>=3:triangles.append(augmented[:,:3])
    projected=[]
    for _,planes in cells:
        for triangle in triangles:
            polygon=triangle
            for plane in planes:
                polygon=clip_polygon(polygon,plane)
                if len(polygon)<3:break
            if len(polygon)>=3:
                p=Polygon(polygon[:,[0,2]])
                if not p.is_valid:p=p.buffer(0)
                if p.area>1e-16:projected.append(p)
    pad=unary_union([domain for domain,_ in cells]);union=unary_union(projected)
    polygons=[]
    if not union.is_empty:
        polygons=[union] if union.geom_type=='Polygon' else list(union.geoms)
        polygons=[p for p in polygons if p.geom_type=='Polygon'];polygons.sort(key=lambda p:p.area,reverse=True)
    groups=connected_groups(polygons,topology_tolerance_m);largest=[polygons[i] for i in groups[0]] if groups else []
    bounds=[min(p.bounds[0] for p in largest),min(p.bounds[1] for p in largest),max(p.bounds[2] for p in largest),max(p.bounds[3] for p in largest)] if largest else [0.,0.,0.,0.]
    return dict(area_m2=float(union.area),effective_pad_area_m2=float(pad.area),area_fraction=float(union.area/pad.area) if pad.area else 0.,
        largest_component_area_m2=float(sum(p.area for p in largest)),
        span_x_m=float(bounds[2]-bounds[0]),span_z_m=float(bounds[3]-bounds[1]),
        topology_tolerance_m=topology_tolerance_m,component_groups=groups,polygons_xz_m=[dict(exterior=np.asarray(p.exterior.coords).tolist(),holes=[np.asarray(r.coords).tolist() for r in p.interiors]) for p in polygons])
