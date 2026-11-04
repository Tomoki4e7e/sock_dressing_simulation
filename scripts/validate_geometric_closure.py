#!/usr/bin/env python3
"""Independently verify all geometric outputs and the stored projected union."""
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[1]
def main():
 root=ROOT/'artifacts/geometric-closure';summary=json.loads((root/'summary.json').read_text());cases=[]
 for pose in summary['poses']:
  report=json.loads((root/f"pose-{pose['number']:02d}/report.json").read_text())
  source=json.loads((ROOT/f"artifacts/grasp-cloth-refilter/pose-{pose['number']:02d}/report.json").read_text())
  assert len(report['candidates'])==len(source['accepted'])
  for index,row in enumerate(report['candidates']):
   assert row['candidate']==source['accepted'][index]
   assert row['accepted']==all(row['checks'].values())==row['execution_plan']['enabled']
   assert 0<row['closing_gap_m']<row['opening_gap_m']
   assert row['execution_plan']['mode']=='geometry_only' and not row['execution_plan']['friction_or_retention_condition']
   assert len(row['open_placements'])==len(row['closed_placements'])==len(row['approach_start_placements'])==5
   p=row['projection'];polygons=[Polygon(q['exterior'],q['holes']) for q in p['polygons_xz_m']]
   area=unary_union(polygons).area
   assert np.isclose(area,p['area_m2'],rtol=0,atol=1e-12)
   assert area<=p['effective_pad_area_m2']+1e-12
   assert p['largest_component_area_m2']<=area+1e-12
   if row['accepted']:
    assert row['closure_skin_clearance_m']>=summary['settings']['body_clearance_m']-1e-9
    assert row['approach_skin_clearance_m']>=summary['settings']['body_clearance_m']-1e-9
    assert not row['approach_cloth_intersections'] and row['insertion']['passed']
   cases.append(row)
 assert len(cases)==436 and sum(c['accepted'] for c in cases)==summary['accepted_count']
 result=dict(passed=True,checked_cases=len(cases),accepted_cases=sum(c['accepted'] for c in cases),input_identity=True,union_area_verified=True,positive_gap=True,force_or_retention_used=False)
 (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
