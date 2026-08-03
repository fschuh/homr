from typing import Any

import numpy as np

from homr import constants
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.models import VisualGroup

MAX_CHORD_NOTEHEAD_HORIZONTAL_GAP_RATIO = 0.25


def source_notehead_bounds(
    group: VisualGroup,
) -> tuple[float, float, float, float] | None:
    points = [point for contour in group.notehead_contours for point in contour]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def noteheads_can_share_chord_stem(first: VisualGroup, second: VisualGroup) -> bool:
    first_bounds = source_notehead_bounds(first)
    second_bounds = source_notehead_bounds(second)
    if first_bounds is None or second_bounds is None:
        return False
    first_left, _first_top, first_right, _first_bottom = first_bounds
    second_left, _second_top, second_right, _second_bottom = second_bounds
    first_width = max(first_right - first_left, 1.0)
    second_width = max(second_right - second_left, 1.0)
    horizontal_gap = max(
        0.0,
        max(first_left, second_left) - min(first_right, second_right),
    )
    return (
        horizontal_gap <= min(first_width, second_width) * MAX_CHORD_NOTEHEAD_HORIZONTAL_GAP_RATIO
    )


def token_moments(symbols: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
    """Group chord tokens without re-sorting members by predicted pitch."""
    moments: list[list[EncodedSymbol]] = []
    append_to_previous = False
    for symbol in symbols:
        if symbol.rhythm == "chord":
            append_to_previous = True
            continue
        if append_to_previous and moments:
            moments[-1].append(symbol)
            append_to_previous = False
        else:
            moments.append([symbol])
    return moments


def symbol_group_distance(symbol: EncodedSymbol, group: VisualGroup) -> float:
    if symbol.coordinates is None or group.transformer_center is None:
        return float("inf")
    return float(np.linalg.norm(np.subtract(symbol.coordinates, group.transformer_center)))


def diatonic_pitch_index(pitch: str) -> int | None:
    if len(pitch) < 2 or pitch[0] not in "CDEFGAB":
        return None
    try:
        octave = int(pitch[1:])
    except ValueError:
        return None
    return octave * 7 + "CDEFGAB".index(pitch[0])


def local_staff_unit(point: Any, line_index: int) -> float:
    lines_per_stave = constants.number_of_lines_on_a_staff
    stave_start = (line_index // lines_per_stave) * lines_per_stave
    stave_lines = point.y[stave_start : stave_start + lines_per_stave]
    differences = np.diff(stave_lines)
    return float(np.median(differences)) if len(differences) else 1.0
