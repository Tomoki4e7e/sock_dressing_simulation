#!/usr/bin/env python3
"""Close an already running recorded replay at a completed command boundary."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import yaml


def process_matches(pid, token):
    try:
        return token in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
    except FileNotFoundError:
        return False


def stop_and_finalize(directory, pid, commands):
    directory = Path(directory).resolve()
    token = 'scripts/replay_foot_contact.py'
    if not process_matches(pid, token):
        raise RuntimeError('Controller PID does not match the requested replay')
    actions = next(directory.glob('data_sock_sim_smoke/train/phase4_*/applied_action.csv'))
    request = dict(controller_pid=pid, completed_command_limit=commands,
                   mode='fixed_motion_replay', new_autonomous_run_separate=True)
    (directory/'user-stop-request.json').write_text(json.dumps(request, indent=2)+'\n')
    print(f'Waiting for {commands} flushed applied command rows.', flush=True)
    previous = -1
    while True:
        # This CSV is flushed after all four view frames have been recorded.
        with actions.open() as stream:
            completed = sum(len(row)==18 for row in csv.reader(stream))
        if completed != previous:
            print(f'Completed commands: {completed}', flush=True)
            previous = completed
        if completed >= commands:
            if completed != commands:
                raise RuntimeError('Requested command boundary has already been exceeded')
            if process_matches(pid, token):
                os.kill(pid, signal.SIGINT)
            break
        if not process_matches(pid, token):
            raise RuntimeError('Controller exited before the requested boundary')
        time.sleep(.01 if completed==commands-1 else .25)
    print('Sent SIGINT; waiting for video writers and native player to close.', flush=True)
    while process_matches(pid, token):
        time.sleep(.25)
    metadata_path = actions.parent/'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    (directory/'metadata-before-user-stop-annotation.json').write_text(json.dumps(metadata, indent=2)+'\n')
    predicted = np.loadtxt(actions.parent/'predicted_action.csv', delimiter=',', ndmin=2)
    applied = np.loadtxt(actions, delimiter=',', ndmin=2)
    if applied.shape != (commands,18):
        raise RuntimeError('Recorded command count changed during shutdown')
    config = yaml.safe_load((directory/'config.yaml').read_text())
    source = Path(config['inference']['reference_actions'])
    recorded = np.loadtxt(source, delimiter=',', ndmin=2)
    maximum_error = float(np.max(np.abs(applied-recorded[:commands])))
    if maximum_error != 0 or not np.array_equal(predicted[:commands], applied):
        raise RuntimeError('Existing replay command provenance failed')
    initial = json.loads((directory/'foot-initial-state.json').read_text())
    clock = initial['dressing_qa']['foot_collision_safety']['cloth_physics_clock']
    metadata.update(mode='fixed_motion_replay', frames=commands, frames_inferred=0,
        frames_replayed=commands, stop_reason='user_requested_stop',
        original_requested_commands=100, effective_requested_commands=commands,
        user_requested_prefix_complete=True, replay_source=str(source),
        replay_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        replay_command_maximum_error=maximum_error,
        rollout_initial_cloth_physics_clock=clock,
        policy_source='recorded_applied_commands',
        perception_source='renderer_measurements',
        next_command_may_have_started_before_interrupt=True)
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    (directory/'result.json').write_text(json.dumps(dict(metadata=str(metadata_path),
        completed_commands=commands, stop_reason='user_requested_stop'), indent=2)+'\n')
    from foot_containment_demo import finalize
    finalize(directory, metadata_path, commands)
    from finalize_foot_containment_demo import finalize as review
    review(directory)
    print('Closed and verified requested replay prefix.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--commands', type=int, default=40)
    args = parser.parse_args()
    stop_and_finalize(args.directory, args.pid, args.commands)
