from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from report_foot_containment import outcome


def test_enclosure_is_reported_even_when_material_validation_fails_and_optional_ankle_is_exposed():
    rows = [dict(command=i, diagnostic_cap_triangles=30, safety_ok=True, stretch_ok=False,
                 sections={name:{'ratio':1, 'ambiguous_samples':0}
                           for name in ('toes','forefoot','heel')}) for i in (38,39,40)]
    for row in rows:
        row['sections']['ankle'] = {'ratio':0, 'ambiguous_samples':1}
    report = dict(rows=rows, opening_passage_supported=True,
                  capture_did_not_advance_physics=True, recording_complete=True,
                  original_commands_unchanged=True, all_frame_safety_ok=True,
                  all_frame_stretch_ok=False)
    result = outcome(report, visual_confirmation=True)
    assert result['first_three_command_geometric_containment_completed_at'] == 40
    assert result['foot_enclosure_confirmed']
    assert result['final_regional_geometric_containment']
    assert not result['fixed_condition_validation_passed']
    assert result['first_three_command_containment_with_contact_and_stretch_completed_at'] is None
    assert not outcome(report)['foot_enclosure_confirmed']
    rows[1]['sections']['heel']['ratio'] = .5
    assert not outcome(report, visual_confirmation=True)['foot_enclosure_confirmed']
