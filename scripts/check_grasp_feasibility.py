#!/usr/bin/env python3
"""Necessary graph-distance bound for unchanged cloth limits and grasp anchors.

A violation proves infeasibility. No violation does not prove feasibility:
obstacles and cloth-face contact can impose additional constraints.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path


def check_frame(frame: dict, grasp_reach: float = .2) -> dict:
    edges = frame['particle_edges']
    rest = frame['particle_rest_edge_lengths']
    if len(edges) != len(rest):
        raise ValueError('Edge and rest-length diagnostics disagree')
    opening = set(frame['opening_particle_indices'])
    graph: dict[int, list[tuple[int, float]]] = {}
    for (a, b), length in zip(edges, rest):
        if not math.isfinite(length) or length <= 0:
            raise ValueError('Invalid rest length')
        limit = 2.0 if a in opening and b in opening else 1.1 if (a in opening) != (b in opening) else 1.5
        bound = length * limit * 1.01
        graph.setdefault(a, []).append((b, bound))
        graph.setdefault(b, []).append((a, bound))
    anchors = frame['dressing_qa']['foot_collision_safety']['original_grasp_anchors']
    points = [(side['side'], int(index), position) for side in anchors
              for index, position in zip(side['particle_indices'], side['positions_world'])]
    pairs = []
    for i, (side, start, position) in enumerate(points):
        distances = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for neighbor, weight in graph.get(node, []):
                candidate = distance + weight
                if candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        for other_side, end, other_position in points[i + 1:]:
            bound = distances.get(end, math.inf)
            separation = math.dist(position, other_position)
            pairs.append(dict(first_side=side, first_particle=start, second_side=other_side,
                              second_particle=end, anchor_separation_m=separation,
                              maximum_material_path_m=bound,
                              exact_pin_excess_m=separation-bound,
                              grasp_reach_excess_m=separation-2*grasp_reach-bound))
    if not pairs:
        raise ValueError('No original grasp-anchor pairs')
    worst = max(pairs, key=lambda pair: pair['grasp_reach_excess_m'])
    return dict(frame=frame['frame'], grasp_reach_m=grasp_reach, numerical_tolerance_fraction=.01,
                infeasible_with_grasp_reach=worst['grasp_reach_excess_m'] > 1e-6,
                worst_pair=worst, pairs=pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grasp-reach', type=float, default=.2,
                        help='Original configured grasp slip threshold, not particle selection radius')
    args = parser.parse_args()
    frames = [check_frame(json.loads(line), args.grasp_reach)
              for line in args.diagnostics.read_text().splitlines() if line.strip()]
    result = dict(necessary_bound_only=True, frame_count=len(frames),
                  infeasible_frames=sum(frame['infeasible_with_grasp_reach'] for frame in frames),
                  frames=frames)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'frames'}))
    if frames:
        print(json.dumps(max(frames, key=lambda frame: frame['worst_pair']['grasp_reach_excess_m'])['worst_pair']))


if __name__ == '__main__':
    main()
