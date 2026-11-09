#!/usr/bin/env python3
"""Execute paired contact routes and retain only validated RGB/feedback episodes."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from replay_contact_pregrasp import verify_source
from collect_reaching_shareset import check_raw
from replay_diverse_left_reaching import compress_intermediates
from sock_dressing_simulation.reaching_collection import sha256


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def planning_proofs(path, initial):
    """Require certificates for these exact route bytes, including repaired routes."""
    digest=sha256(path);root=path.parent.parent.parent
    bases=[root/'robust-verification',root/'skin-far-verification']
    matches={}
    for kind,folders in [('continuous',['continuous','continuous-shoulder-bound']),('endpoint',['endpoint-audit'])]:
        for base in bases:
            for folder in folders:
                certificate=base/folder/path.name
                if not certificate.exists():continue
                result=json.loads(certificate.read_text());binding=result.get('binding',result)
                if result.get('passed') and binding.get('route_sha256')==digest:
                    initial_hash=binding.get('initial_sha256')
                    if initial_hash is None:
                        initial_hash=json.loads((root/'result-summary.json').read_text())['source_initial_sha256']
                    if initial_hash!=sha256(initial):continue
                    matches[kind]=dict(path=str(certificate.relative_to(ROOT)),sha256=sha256(certificate));break
            if kind in matches:break
        if kind not in matches:raise ValueError('No passed bound '+kind+' certificate for '+str(path))
    return matches


def collect(path, args, index):
    source = path.parent.parent
    plan = json.loads(path.read_text())
    state = ROOT / plan['condition']['state']
    initial = source / 'candidate-initial.yaml'
    verify_source(path, initial, state)
    proofs=planning_proofs(path,initial)
    dest = args.output / 'attempts' / path.stem
    dest.mkdir(parents=True, exist_ok=True)
    episode = dest / 'episode'
    player = Path(os.environ.get('SOCK_REGISTERED_PLAYER', ROOT / 'Build/SockDressingPlayer/RegisteredLightingV1/Player.x86_64')).resolve()
    binding = {str(p.relative_to(ROOT)): sha256(p) for p in
               [path, initial, state, ROOT / 'scripts/capture_fixed_torso_initial.py',
                ROOT / 'scripts/replay_contact_pregrasp.py', ROOT / 'scripts/native_pregrasp_geometry.py',
                ROOT / 'sock_dressing_simulation/reaching_collection.py',
                ROOT / 'sock_dressing_simulation/continuous_pregrasp.py',
                ROOT / 'sock_dressing_simulation/head_cameras.py',
                ROOT / 'sock_dressing_simulation/learning_lighting.py', Path(__file__).resolve()]}
    binding['human_collision_model']=os.environ.get('SOCK_CONTACT_HUMAN_MODEL','native')
    binding['drive_profile']=os.environ.get('SOCK_CONTACT_DRIVE_PROFILE','legacy')
    binding['exact_human_contact_sha256']=sha256(ROOT/'sock_dressing_simulation/exact_human_contact.py')
    binding['image_size']=args.image_size
    binding['duration_scale']=args.duration_overrides.get(path.stem,args.duration_scale)
    binding['planning_proofs']=proofs
    binding.update(player_path=str(player), player_executable_sha256=sha256(player),
                   player_managed_sha256=sha256(player.parent / 'Player_Data/Managed/RCareWorld.dll'),
                   fixed_step_s=.02, head_cameras=True, lighting_profile='balanced')
    status_path = dest / 'status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get('binding') == binding and previous.get('finished'):
            return previous
        raise RuntimeError('Existing attempt has different inputs; choose a new output: ' + str(dest))
    if episode.exists():raise RuntimeError('Unfinished episode requires inspection: ' + str(episode))
    if shutil.disk_usage(dest).free < 2_000_000_000:
        raise RuntimeError('Less than 2 GB free; collection stopped')
    env = dict(os.environ, PYTHONUNBUFFERED='1', SOCK_REGISTERED_BATCHMODE='1', SOCK_REGISTERED_PLAYER=str(player),
               SOCK_REGISTERED_PLAYER_LOG=str(dest / 'player.log'), LIBGL_ALWAYS_SOFTWARE='1',
               __GLX_VENDOR_LIBRARY_NAME='mesa')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / '.deps/RCareWorld/pyrcareworld'),
                                       '/tmp/sock-grasp-deps', '/opt/ros/noetic/lib/python3/dist-packages'])
    command = [sys.executable, str(ROOT / 'scripts/capture_fixed_torso_initial.py'),
               '--initial-configuration', str(initial), '--state', str(state),
               '--contact-route', str(path), '--output', str(dest), '--port', str(args.port + index * 4),
               '--collect-episode', str(episode), '--head-cameras', '--lighting-profile', 'balanced',
               '--image-size', *map(str,args.image_size),'--duration-scale',str(binding['duration_scale'])]
    status = dict(id=path.stem, condition=plan['condition'], binding=binding, finished=False,
                  passed=False, episode=str(episode.relative_to(args.output)), final_closure_performed=False)
    write(status_path, status)
    began = time.monotonic()
    print('COLLECT CONTACT', path.stem, flush=True)
    with (dest / 'replay.log').open('w') as log:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:status['returncode'] = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL);process.wait()
            status.update(returncode=-15, error='Native replay timeout')
    result_path = dest / 'native-route-initial/replay-result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        status.update(native_passed=result['passed'], motion_completed=result['completed'],
                      final_contact_checks=result['final_contact_checks'],
                      maximum_tracking_error_m=result['maximum_tracking_error_m'],
                      unexpected_contact_steps=len(result['unexpected_contacts']),
                      geometric_failure_steps=len(result['actual_configuration_failures']))
    if (episode / 'metadata.json').exists():
        try:
            qa = check_raw(episode)
            status.update(raw_validation=qa, frames=qa['frames'])
            status['passed'] = bool(status['returncode'] == 0 and status.get('native_passed') and qa['passed'])
            metadata = json.loads((episode / 'metadata.json').read_text())
            native_initial = json.loads((dest / 'native-route-initial/result.json').read_text())
            metadata.update(lighting_profile='balanced', lighting=native_initial['learning_lighting'])
            metadata.update(human_collision_model=binding['human_collision_model'],exact_human_contact=native_initial.get('exact_human_contact'))
            metadata['duration_scale']=binding['duration_scale']
            metadata['drive_profile']=binding['drive_profile']
            metadata['planning_proofs']=proofs
            metadata.update(raw_validation_passed=qa['passed'], reaching_learning_ready=status['passed'],
                            contact_pregrasp_learning_ready=status['passed'])
            write(episode / 'metadata.json', metadata)
        except Exception as error:status['validation_error'] = str(error)
    compress_intermediates(dest)
    status.update(finished=True, elapsed_s=time.monotonic() - began)
    write(status_path, status)
    print('CONTACT EPISODE', path.stem, status['passed'], status.get('frames', 0), flush=True)
    return status


def report(args):
    rows = [json.loads(p.read_text()) for p in sorted((args.output / 'attempts').glob('*/status.json'))]
    good = [r for r in rows if r['passed']]
    value = dict(expected=70, attempted=len(rows), finished=sum(r['finished'] for r in rows),
                 accepted=len(good), all_70_passed=len(good) == 70,
                 final_closure_performed=False, cloth_frozen=True,
                 scope='Native joint-driven contact pregrasp with synchronized measured angles, effort and stereo RGB; no closure or retention test',
                 accepted_episodes=[dict(id=r['id'], path=r['episode'], frames=r['frames'],
                                         scene_number=r['condition']['scene_number']) for r in good], rows=rows)
    write(args.output / 'collection.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/contact-placement-search/skin-mesh-far')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/contact-pregrasp-learning')
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--port', type=int, default=29000)
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--image-size',nargs=2,type=int,default=[640,480],metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--duration-scale',type=float,default=1.)
    parser.add_argument('--duration-overrides',type=Path,help='JSON object mapping condition IDs to duration scales')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    args.duration_overrides=json.loads(args.duration_overrides.read_text()) if args.duration_overrides else {}
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.source / 'routes').glob('*.json'))
    if args.ids:
        paths = [p for p in paths if p.stem in args.ids]
        if {p.stem for p in paths} != set(args.ids):parser.error('Unknown route ID')
    elif len(paths) != 70:parser.error('Expected exactly 70 source routes')
    if not args.report_only:
        for path in paths:planning_proofs(path,args.source/'candidate-initial.yaml')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(collect, path, args, i) for i, path in enumerate(paths)]):
                future.result();report(args)
    value = report(args)
    print('COLLECTION', value['accepted'], '/', value['finished'], 'accepted', flush=True)


if __name__ == '__main__':main()
