#!/usr/bin/env python3
"""Synchronized real simulator views and independent, diagnostic-only containment."""
from __future__ import annotations

from collections import Counter
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

from report_natural_opening import ordered_ring


def cross2(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def closed_faces(row):
    """Ear-clip the projected cuff. Never modify physical or rendered geometry."""
    vertices = np.asarray(row['particles_world'], dtype=float)
    faces = np.asarray(row['particle_surface_triangles'], dtype=int).reshape(-1, 3)
    ring = ordered_ring(row)
    points = vertices[ring]
    _, _, axes = np.linalg.svd(points-points.mean(axis=0), full_matrices=False)
    xy = (points-points.mean(axis=0)) @ axes[:2].T
    n = len(ring)
    for i in range(n):
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            a,b,c,d = xy[i],xy[(i+1)%n],xy[j],xy[(j+1)%n]
            if cross2(b-a,c-a)*cross2(b-a,d-a) < 0 and cross2(d-c,a-c)*cross2(d-c,b-c) < 0:
                raise ValueError('Projected opening self-intersects; containment is ambiguous')
    orientation = np.sign(sum(cross2(xy[i], xy[(i+1)%n]) for i in range(n)))
    if not orientation:
        raise ValueError('Degenerate opening')
    remaining = list(range(n)); cap = []
    while len(remaining) > 3:
        for k,b in enumerate(remaining):
            a,c = remaining[k-1],remaining[(k+1)%len(remaining)]
            if orientation*cross2(xy[b]-xy[a],xy[c]-xy[b]) <= 1e-12:
                continue
            others = [q for q in remaining if q not in (a,b,c)]
            if any(all(orientation*cross2(xy[v]-xy[u],xy[q]-xy[u]) >= -1e-12
                       for u,v in ((a,b),(b,c),(c,a))) for q in others):
                continue
            cap.append([ring[a],ring[b],ring[c]])
            remaining.pop(k)
            break
        else:
            raise ValueError('Opening cannot be triangulated reliably')
    cap.append([ring[q] for q in remaining])
    cap = np.asarray(cap, dtype=int)
    all_faces = np.vstack([faces,cap])
    counts = Counter(tuple(sorted((int(a),int(b)))) for t in all_faces
                     for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])))
    if any(n != 2 for n in counts.values()):
        raise ValueError('Diagnostic sock mesh is not closed and manifold')
    return vertices, all_faces, cap


def ray_distances(point, direction, triangles):
    a,b,c = triangles[:,0],triangles[:,1],triangles[:,2]
    e1,e2 = b-a,c-a
    h = np.cross(np.broadcast_to(direction,e2.shape),e2)
    det = np.einsum('ij,ij->i',e1,h)
    valid = np.abs(det) > 1e-12
    inv = np.divide(1,det,out=np.zeros_like(det),where=valid)
    s = point-a
    u = inv*np.einsum('ij,ij->i',s,h)
    q = np.cross(s,e1)
    v = inv*(q @ direction)
    distance = inv*np.einsum('ij,ij->i',e2,q)
    hits = np.sort(distance[valid & (u >= -1e-9) & (v >= -1e-9) &
                            (u+v <= 1+1e-9) & (distance > 1e-8)])
    return hits[np.r_[True,np.diff(hits)>1e-7]] if len(hits) else hits


def inside_points(points, triangles):
    directions = np.array([[.936329,.280899,.210674],[-.317,.863,.394],[.211,-.397,.893]])
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    votes = np.array([[len(ray_distances(p,d,triangles)) % 2 == 1
                       for d in directions] for p in points],dtype=bool)
    return votes.all(axis=1), (votes.any(axis=1) != votes.all(axis=1))


def analyze_row(row):
    result = {'command':row['frame']+1,'valid':False,'sections':{},
              'toe_center_inside':False,'reason':None}
    try:
        vertices, faces, cap = closed_faces(row)
        anatomy = row['dressing_qa']['anatomical_foot_conformity']
        if not anatomy.get('valid'):
            raise ValueError('Anatomical samples unavailable')
        for section in anatomy['sections']:
            samples = np.asarray(section['samples_world'],dtype=float)
            contained, ambiguous = inside_points(samples,vertices[faces])
            result['sections'][section['name']] = {
                'contained_samples':int(contained.sum()),'total_samples':len(samples),
                'ratio':float(contained.mean()),'ambiguous_samples':int(ambiguous.sum()),
                'legacy_ratio':section.get('containment_ratio')}
        toe = np.asarray(row['scene_geometry']['right_toe_position'])
        contained, ambiguous = inside_points([toe],vertices[faces])
        result['toe_center_inside'] = bool(contained[0])
        result['toe_center_ambiguous'] = bool(ambiguous[0])
        result['valid'] = not ambiguous[0] and all(
            s['ambiguous_samples']==0 for s in result['sections'].values())
        result['diagnostic_cap_triangles'] = len(cap)
    except (ValueError,KeyError) as error:
        result['reason'] = str(error)
    safety = row['dressing_qa']['foot_collision_safety']
    result['anchor_error_mm'] = 1000*safety['maximum_original_grasp_anchor_error_m']
    result['safety_ok'] = all(safety.get(k)==0 for k in (
        'remaining_surface_intersections','remaining_swept_intersections',
        'remaining_swept_surface_intersections')) and safety.get(
            'maximum_geometric_penetration_m',float('inf')) <= .00200001
    # V88 uses this named final geometric penetration field (not raw Obi contacts).
    if 'maximum_geometric_penetration_m' not in safety:
        result['safety_ok'] = all(safety.get(k)==0 for k in (
            'remaining_surface_intersections','remaining_swept_intersections',
            'remaining_swept_surface_intersections')) and row['dressing_qa'].get(
                'geometric_maximum_cloth_foot_penetration_m',float('inf')) <= .00200001
    result['stretch_ok'] = bool(row.get('stretch',{}).get('available')) and all(
        e.get('maximum_stretch') is None and e.get('edge_count')==0 or
        e.get('maximum_stretch',float('inf')) <= (2 if name=='opening_rim' else 1.1 if name=='opening_body' else 1.5)*1.01
        for name,e in row.get('stretch',{}).get('edge_classes',{}).items())
    result['foot_contained'] = result['valid'] and all(
        result['sections'].get(name,{}).get('ratio',0) >= .95 for name in ('toes','forefoot','heel'))
    return result


class FootViewRecorder:
    def __init__(self, environment, directory):
        self.env = environment; self.directory = Path(directory)
        self.cameras = {}; self.videos = {}; self.frames = 0

    def initialize(self, data):
        sections = data['dressing_qa']['anatomical_foot_conformity']['sections']
        samples = np.vstack([s['samples_world'] for s in sections if s['name'] in ('toes','forefoot','heel')])
        target = (samples.min(axis=0)+samples.max(axis=0))/2
        self.poses = {}
        for name,identifier,height,rotation in (
                ('top',1311,.35,[90,0,0]),('bottom',1312,-.30,[-90,0,180])):
            camera = self.env._env.InstanceObject(name='Camera',id=identifier,
                attr_type=type(self.env.recording_camera))
            position = target+np.array([0,height,0])
            camera.SetTransform(position=position.tolist(),rotation=rotation,scale=[1,1,1])
            self.cameras[name] = camera
            self.poses[name] = {'position':position.tolist(),'rotation':rotation,'fov':80}
        self.env._env.step(simulate=False)
        (self.directory/'foot-view-cameras.json').write_text(json.dumps(self.poses,indent=2)+'\n')
        initial = {'frame':-1,'dressing_qa':data['dressing_qa'],
                   **{k:data[k] for k in ('particles','particle_surface_triangles','opening_particle_indices','scene_geometry')}}
        initial['particles_world'] = initial.pop('particles')
        (self.directory/'foot-initial-state.json').write_text(json.dumps(initial)+'\n')

    def capture(self, cloth):
        before = dict(cloth['dressing_qa']['foot_collision_safety']['cloth_physics_clock'])
        for camera in self.cameras.values():
            camera.GetRGB(1280,960,80)
        self.env._env.step(simulate=False)
        # CameraAttr replaces its data on the next communication step.
        # Preserve RGB before requesting the post-capture clock.
        images = {name:self.env._decode(camera.data['rgb'],'RGB') for name,camera in self.cameras.items()}
        self.env.sock_cloth.request_dressing_qa()
        self.env._env.step(simulate=False)
        after = self.env.sock_cloth.data['dressing_qa']['foot_collision_safety']['cloth_physics_clock']
        if before != after:
            raise RuntimeError('Foot-view capture advanced cloth physics')
        return {'images':images,
                'clock_before':before,'clock_after':dict(after)}

    def record(self, capture, cloth, frame):
        from sock_dressing_simulation.demo import _open_video, _write_video_frame
        for name,rgb in capture['images'].items():
            folder = self.directory/('camera_'+name); folder.mkdir(exist_ok=True)
            Image.fromarray(rgb).save(folder/f'{frame:04d}.png')
            if name not in self.videos:
                self.videos[name] = _open_video(self.directory/(name+'_demo.mp4'),rgb.shape,5.0)
            _write_video_frame(self.videos[name],rgb,frame)
        with (self.directory/'foot-view-sync.jsonl').open('a') as stream:
            stream.write(json.dumps({'frame':frame,'clock_before':capture['clock_before'],
                                    'clock_after':capture['clock_after']})+'\n')
        self.frames += 1

    def close(self):
        for video in self.videos.values():
            video.release()


def opening_crossing(previous, current):
    """Sample the moving diagnostic aperture at <=1 mm boundary displacement.

    This is endpoint-interpolated evidence, not native substep trajectory proof.
    Native point/face/swept CCD diagnostics are checked separately.
    """
    try:
        old,_,cap = closed_faces(previous)
        new = np.asarray(current['particles_world'],dtype=float)
        p0 = np.asarray(previous['scene_geometry']['right_toe_position'])
        p1 = np.asarray(current['scene_geometry']['right_toe_position'])
        steps = max(1,int(np.ceil(np.linalg.norm((new-old)[np.unique(cap)]-(p1-p0),axis=1).max()/.001)))
        for j in range(steps):
            t0,t1 = j/steps,(j+1)/steps
            tri0 = (old+(new-old)*t0)[cap]-(p0+(p1-p0)*t0)
            tri1 = (old+(new-old)*t1)[cap]-(p0+(p1-p0)*t1)
            for first,second in zip(tri0,tri1):
                normals = np.array([np.cross(t[1]-t[0],t[2]-t[0]) for t in (first,second)])
                signed = np.array([np.dot(n,-t[0])/max(np.linalg.norm(n),1e-15)
                                   for n,t in zip(normals,(first,second))])
                if signed[0]*signed[1] > 0 or abs(signed[0])+abs(signed[1]) < 1e-12:
                    continue
                f = abs(signed[0])/(abs(signed[0])+abs(signed[1]))
                triangle = first+(second-first)*f
                a,b,c = triangle
                uv = np.linalg.lstsq(np.column_stack((b-a,c-a)),-a,rcond=None)[0]
                if min(uv)>=-1e-6 and sum(uv)<=1+1e-6 and np.linalg.norm(a+(b-a)*uv[0]+(c-a)*uv[1])<=.001:
                    return {'between_commands':[previous['frame']+1,current['frame']+1],
                            'fraction':t0+(t1-t0)*f,'interpolation_displacement_m':.001,
                            'method':'moving_diagnostic_cap_endpoint_interpolation'}
    except (ValueError,KeyError):
        pass
    return None


def finalize(directory, metadata_path, required_steps):
    from report_natural_opening import records
    directory = Path(directory)
    rows = records(directory/'live_frame_diagnostics.jsonl')
    values = [analyze_row(row) for row in rows]
    initial = json.loads((directory/'foot-initial-state.json').read_text())
    previous = initial; crossings = []
    for row in rows:
        event = opening_crossing(previous,row)
        if event is not None:
            event['toe_inside_after'] = values[row['frame']]['toe_center_inside']
            crossings.append(event)
        previous = row
    streak = 0; first = None
    for value in values:
        streak = streak+1 if value['foot_contained'] and value['safety_ok'] and value['stretch_ok'] else 0
        if streak>=3 and first is None:
            first = value['command']
    sync = records(directory/'foot-view-sync.jsonl')
    clocks = [row['dressing_qa']['foot_collision_safety']['cloth_physics_clock'] for row in rows]
    metadata = json.loads(Path(metadata_path).read_text())
    initial_clock = metadata['rollout_initial_cloth_physics_clock']
    # The unchanged replay consumes 18 fixed steps per recorded command.
    increments = np.diff([initial_clock['fixed_steps']]+[c['fixed_steps'] for c in clocks])
    sync_ok = len(sync)==len(rows) and all(s['clock_before']==s['clock_after']==c
        for s,c in zip(sync,clocks)) and all(increments==18)
    crossing_supported = any(e['toe_inside_after'] for e in crossings)
    autonomous = metadata.get('mode')=='samdamsarnn_closed_loop'
    command_integrity = (metadata.get('policy_source')=='SAMDAMSARNNPolicy' and
        metadata.get('frames_replayed')==0 and metadata.get('frames_inferred')==len(rows) and
        metadata.get('reference_action_projection',{}).get('path') is None and
        metadata.get('reference_action_projection',{}).get('blend')==0 and
        metadata.get('reference_action_projection',{}).get('cartesian_pull_m')==0) if autonomous else metadata['replay_command_maximum_error']==0
    report = {'completed_commands':len(rows),'required_commands':required_steps,
        'mode':metadata.get('mode','fixed_motion_replay'),
        'original_commands_unchanged':None if autonomous else command_integrity,
        'command_source_verified':command_integrity,
        'recording_complete':len(rows)==required_steps and (
            metadata['stop_reason']=='max_steps' or
            (metadata['stop_reason']=='user_requested_stop' and
             metadata.get('user_requested_prefix_complete',False) and
             metadata.get('effective_requested_commands')==required_steps)),
        'capture_did_not_advance_physics':sync_ok,
        'opening_crossings':crossings,'opening_passage_supported':crossing_supported,
        'first_three_command_containment_completed_at':first,
        'foot_containment_confirmed_by_geometry':first is not None and crossing_supported and sync_ok,
        'final_foot_contained':bool(values and values[-1]['foot_contained']),
        'all_frame_safety_ok':all(v['safety_ok'] for v in values),
        'all_frame_stretch_ok':all(v['stretch_ok'] for v in values),
        'visual_review_complete':False,'visual_confirmation':None,
        'ankle_calf_pull_up_required':False,
        'grasp_repair_deferred':True,'containment_threshold':.95,'consecutive_commands':3,
        'sampling_limit':'Existing anatomical skin envelope: 8 samples per section; not exhaustive surface coverage.',
        'passage_limit':'Endpoint interpolation plus native CCD diagnostics; camera endpoints alone do not prove the continuous trajectory.',
        'diagnostic_cap_used_in_physics_or_rendering':False,
        'diagnostics_sha256':hashlib.sha256((directory/'live_frame_diagnostics.jsonl').read_bytes()).hexdigest(),
        'rows':values}
    (directory/'foot-containment.json').write_text(json.dumps(report,indent=2)+'\n')
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(directory/'.plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,1,figsize=(10,6),sharex=True)
    for name in ('toes','forefoot','heel','ankle','calf'):
        axes[0].plot([v['command'] for v in values],[v['sections'].get(name,{}).get('ratio',np.nan) for v in values],label=name)
    axes[0].axhline(.95,color='gray',linestyle='--'); axes[0].set_ylabel('Contained sample fraction');axes[0].legend()
    axes[1].plot([v['command'] for v in values],[v['anchor_error_mm'] for v in values])
    axes[1].set_ylabel('Maximum grasp error (mm)');axes[1].set_xlabel('Recorded command')
    for axis in axes:axis.grid(alpha=.25)
    fig.tight_layout();fig.savefig(directory/'containment-and-grasp.png',dpi=160);plt.close(fig)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(directory/'top_demo.mp4'),
        '-i',str(directory/'bottom_demo.mp4'),'-filter_complex','[0:v][1:v]hstack=inputs=2',
        '-c:v','libx264','-crf','18',str(directory/'top-bottom-demo.mp4')],check=True)
    videos = {'front':Path(metadata_path).parent/'demo.mp4','side':directory/'side_demo.mp4',
              'top':directory/'top_demo.mp4','bottom':directory/'bottom_demo.mp4',
              'top_bottom':directory/'top-bottom-demo.mp4'}
    video_info = {}
    for name,path in videos.items():
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,avg_frame_rate',
            '-of','json',str(path)],text=True))['streams'][0]
        if int(info['nb_read_frames'])!=len(rows):
            raise RuntimeError('Unsynchronized video frame count: '+name)
        video_info[name] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**info}
    (directory/'video-provenance.json').write_text(json.dumps(video_info,indent=2)+'\n')
    selected = {'final':len(rows)-1}
    if crossings:selected['opening_passage']=crossings[0]['between_commands'][1]-1
    if first is not None:selected['foot_containment']=first-1
    for label,index in selected.items():
        for name,path in videos.items():
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(path),'-vf',f'select=eq(n\\,{index})',
                            '-frames:v','1',str(directory/f'{label}-{name}.png')],check=True)
    conclusion = 'Geometric containment criterion reached; visual review pending.' if first is not None else 'Foot containment criterion was not reached.'
    (directory/'REPORT.md').write_text(('# V88 autonomous SAMDAMSARNN' if autonomous else '# V88 fixed-trajectory')+' foot containment demo\n\n'+conclusion+
        '\n\nAnkle/calf pull-up is optional. Grasp error remains unresolved and is reported separately.\n\n'+
        f'Commands: {len(rows)}/{required_steps}. Synchronized camera clocks: {sync_ok}. '+
        f'Opening passage evidence: {crossing_supported}. Final foot containment: {report["final_foot_contained"]}.\n\n'+
        'See foot-containment.json, containment-and-grasp.png, video-provenance.json and top-bottom-demo.mp4.\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','opening_crossings')}),flush=True)
