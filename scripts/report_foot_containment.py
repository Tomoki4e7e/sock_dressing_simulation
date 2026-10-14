"""Keep observed enclosure separate from material/contact validation."""


def first_streak(rows, predicate, length=3):
    streak = 0
    for row in rows:
        streak = streak+1 if predicate(row) else 0
        if streak >= length:
            return row['command']
    return None


def required_regions_contained(row):
    # Ankle/calf coverage and ambiguous optional-region samples are not part
    # of the user's toe/forefoot/heel criterion. A valid diagnostic cap and
    # unambiguous required-region samples remain necessary.
    return row.get('diagnostic_cap_triangles', 0) > 0 and all(
        row.get('sections', {}).get(name, {}).get('ratio', 0) >= .95 and
        row['sections'][name].get('ambiguous_samples', 1) == 0
        for name in ('toes', 'forefoot', 'heel'))


def outcome(report, visual_confirmation=None):
    rows = report['rows']
    first_geometry = first_streak(rows, required_regions_contained)
    first_with_constraints = first_streak(rows, lambda row:
        required_regions_contained(row) and row['safety_ok'] and row['stretch_ok'])
    supported = bool(first_geometry is not None and
                     report['opening_passage_supported'] and
                     report['capture_did_not_advance_physics'])
    return {
        'first_three_command_geometric_containment_completed_at':first_geometry,
        'first_three_command_containment_with_contact_and_stretch_completed_at':first_with_constraints,
        'regional_geometric_containment_supported':supported,
        'visual_confirmation':visual_confirmation,
        'foot_enclosure_confirmed':supported and visual_confirmation is True,
        'final_regional_geometric_containment':bool(rows and required_regions_contained(rows[-1])),
        'fixed_condition_validation_passed':bool(
            report['recording_complete'] and report.get('command_source_verified',report['original_commands_unchanged']) and
            report['capture_did_not_advance_physics'] and
            report['all_frame_safety_ok'] and report['all_frame_stretch_ok']),
        'all_frame_safety_ok':report['all_frame_safety_ok'],
        'all_frame_stretch_ok':report['all_frame_stretch_ok'],
        'ankle_calf_pull_up_required':False,
        'definition':'Regional enclosure and material/contact compliance are reported separately. No physical threshold is relaxed.'}
