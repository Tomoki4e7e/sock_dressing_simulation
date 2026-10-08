#!/usr/bin/env python3
"""Read partial trial diagnostics without changing or qualifying recordings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def summarize(directory: Path) -> dict:
    path = directory / 'live_frame_diagnostics.jsonl'
    rows = []
    incomplete_line = False
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            incomplete_line = True
    result = {'trial': str(directory), 'recorded_frames': len(rows),
              'partial_last_line': incomplete_line, 'qualifies_as_best': False}
    if not rows:
        return result
    safety = [row['dressing_qa'].get('foot_collision_safety', {}) for row in rows]
    violations = []
    for row, contact in zip(rows, safety):
        failed = []
        for field in ('remaining_swept_intersections', 'remaining_surface_intersections',
                      'remaining_swept_surface_intersections'):
            if contact.get(field) != 0:
                failed.append(field)
        binding = contact.get('native_grasp_binding', {})
        if (binding.get('all_original_targets_match') is not True or
            binding.get('expected_pin_count') != 10 or binding.get('matched_pin_count') != 10):
            failed.append('native_gripper_binding_missing_or_mismatched')
        for field, value, bound in (
            ('penetration', row['dressing_qa'].get('geometric_maximum_cloth_foot_penetration_m'), .002),
            ('anchor_error', contact.get('maximum_original_grasp_anchor_error_m'), .2),
            ('offset_drift', contact.get('maximum_grasp_offset_drift_m'), .002),
        ):
            if value is None or not math.isfinite(value) or value > bound:
                failed.append(field)
        if failed:
            violations.append({'frame': row['frame'], 'failed': failed})
    latest = rows[-1]
    conformity = latest['dressing_qa'].get('anatomical_foot_conformity', {})
    latest_stretch = {name: values.get('maximum_stretch') for name, values in
                      latest.get('stretch', {}).get('edge_classes', {}).items()}
    material_bad_frames = []
    for row in rows:
        classes = row.get('stretch', {}).get('edge_classes', {})
        values = [(name, values.get('maximum_stretch')) for name, values in classes.items()]
        if not any(value is not None for _, value in values) or any(
            value is not None and (not math.isfinite(value) or value >
                (2 if name == 'opening_rim' else 1.1 if name == 'opening_body' else 1.5) * 1.01)
            for name, value in values
        ):
            material_bad_frames.append(row['frame'])
    result.update(
        latest_frame=latest['frame'], recorded_safety_violations=violations,
        stretch_violation_frames_with_1pct_tolerance=material_bad_frames,
        latest_stretch=latest_stretch,
        latest_anatomical_surface_containment=conformity.get('surface_containment_ratio'),
        latest_section_containment={section['name']: section.get('containment_ratio')
                                    for section in conformity.get('sections', [])},
        latest_opening={key: latest.get('scene_geometry', {}).get(key) for key in (
            'opening_ring_area_retention', 'opening_span_m', 'sock_body_direction',
            'sock_tip_opening_depth_m', 'foot_to_opening_plane_m')},
        latest_contact={key: safety[-1].get(key) for key in (
            'surface_reference_policy', 'surface_bounds_policy', 'surface_reference_triangle_updates',
            'surface_reference_retained_triangles', 'maximum_retained_surface_reference_substeps',
            'native_grasp_binding', 'particle_contact_shell_policy', 'particle_contact_shell_m',
            'opening_rim_shape_weight', 'opening_body_barrier_effective_weight',
            'opening_body_barrier_phase_policy', 'material_opening_barrier_path_clip_count',
            'maximum_original_grasp_anchor_error_m', 'maximum_grasp_offset_drift_m',
            'cloth_physics_clock', 'stage_wall_time_s')},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    trials = []
    for directory in args.directories:
        candidates = [directory] if (directory / 'live_frame_diagnostics.jsonl').exists() else sorted(directory.iterdir())
        trials.extend(summarize(candidate) for candidate in candidates
                      if (candidate / 'live_frame_diagnostics.jsonl').exists())
    report = {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'partial recorded frames; final assessment and visual review required',
              'trials': trials}
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
