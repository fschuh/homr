"""Bounded removal of locally implausible samples from an annotation-only grid.

A page loses its whole optional-annotation capability when a single sampled
column of a single staff has implausible line spacing. Across the measured
corpus that is fifteen columns out of roughly fifty thousand, and they cost five
pages of twenty-one every fingering they could have carried.

This module works on the *exported* annotation copy of the detected grid, never
on ``Staff.grid``, recognition geometry, note contours or associations. It drops
whole sample columns and nothing else: no knot is interpolated, no support sample
is synthesised, and every retained sample keeps its original coordinates exactly.

What makes a removal safe is not that the surviving grid passes the validator --
it is that the consumer's interpolation across the gap still follows the printed
staff. The limits below therefore bound the *interpolation error* the removal
introduces, measured against what the producer's own sampling already shows on
staffs the validator accepts. They are frozen constants, chosen from that
distribution before any challenge test, not tuned until a particular page passed.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from homr.visual_sidecar.annotation_geometry import (
    MAX_GAP_RATIO,
    MAX_STAFF_SPACE_IMAGE_FRACTION,
    MIN_GAP_RATIO,
    MIN_STAFF_SPACE_PIXELS,
    AnnotationGeometryError,
    GeometryDiagnostic,
    validate_physical_staff,
)

STAGE_RECOVERY = "recovery"


@dataclass(frozen=True)
class RecoveryLimits:
    """Frozen bounds on what a removal may cost.

    Every value is expressed in local staff spaces so that it means the same
    thing on a 1200 dpi scan and on a 300 dpi render. The basis given for each
    is the distribution measured over the accepted staffs of the pinned corpus
    (about 38,000 samples), recorded in ``plans/staff-geometry-evidence.md``.
    """

    #: Outer-line distance from the chord across the removed run. Accepted staffs
    #: show a leave-one-out interpolation error of 0.360 spaces at p99.9 and 0.554
    #: at its maximum; 0.40 sits above the former and below the latter.
    max_chord_deviation_spaces: float = 0.40
    #: Local spacing must agree either side of the run. Accepted staffs move by
    #: 0.127 spaces between neighbours at p99.9 and 0.166 at maximum.
    max_neighbour_unit_disagreement: float = 0.15
    #: How far the chord may depart from where the slope just outside was heading,
    #: integrated over the span. Accepted staffs reach 2.97 spaces at p99.9, so
    #: this catches a divergent neighbourhood without constraining ordinary curvature.
    max_slope_disagreement_spaces: float = 3.0
    #: Unsupported span an interior removal may bridge. A single accepted step is
    #: 1.67 spaces at p99.9 and 3.90 at maximum, so this permits a short run only.
    max_interior_span_spaces: float = 6.0
    #: Edge trimming has no chord to check against, so it is bounded more tightly.
    #: No defect in the measured corpus was at an edge: this bound rests on the
    #: neighbour-span distribution alone and has no observed case behind it.
    max_edge_trim_spaces: float = 4.0
    #: Hard stop on a single run. The span bound is the binding constraint.
    max_removed_run: int = 8
    max_removed_fraction: float = 0.05
    min_surviving_samples: int = 2


DEFAULT_LIMITS = RecoveryLimits()


@dataclass(frozen=True)
class StaffRepair:
    """Compact provenance for one repaired staff. Full grids stay in the capture."""

    staff_id: str
    staff_group_index: int
    staff_index: int
    kind: str
    removed: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "staff_id": self.staff_id,
            "staff_group_index": self.staff_group_index,
            "staff_index": self.staff_index,
            "kind": self.kind,
            "removed_count": len(self.removed),
            "removed": self.removed,
        }


@dataclass(frozen=True)
class RecoveryOutcome:
    staff: dict[str, Any] | None
    repair: StaffRepair | None = None
    diagnostics: list[GeometryDiagnostic] = field(default_factory=list)


def _declined(reason: str, message: str, staff: dict[str, Any], **extra: Any) -> RecoveryOutcome:
    return RecoveryOutcome(
        staff=None,
        diagnostics=[
            GeometryDiagnostic(
                reason=reason,
                stage=STAGE_RECOVERY,
                message=message,
                staff_id=staff.get("staff_id"),
                staff_group_index=staff.get("staff_group_index"),
                staff_index=staff.get("staff_index"),
                system_index=staff.get("system_index"),
                **extra,
            )
        ],
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _identity_is_unambiguous(staff: dict[str, Any]) -> bool:
    if not isinstance(staff.get("staff_id"), str) or not staff["staff_id"]:
        return False
    return all(
        isinstance(staff.get(key), int) and not isinstance(staff.get(key), bool) and staff[key] >= 0
        for key in ("staff_group_index", "staff_index", "system_index")
    )


def _shapes_are_consistent(staff: dict[str, Any]) -> bool:
    lines, spacing = staff.get("lines"), staff.get("spacing")
    if not isinstance(lines, list) or len(lines) != 5:
        return False
    if not all(isinstance(line, list) for line in lines):
        return False
    size = len(lines[0])
    if any(len(line) != size for line in lines):
        return False
    return isinstance(spacing, list) and len(spacing) == size


def _coordinates_are_finite(staff: dict[str, Any]) -> bool:
    for series in (*staff["lines"], staff["spacing"]):
        for point in series:
            if not isinstance(point, list) or len(point) != 2 or not all(_finite(v) for v in point):
                return False
    return True


def _samples_share_increasing_x(staff: dict[str, Any]) -> str | None:
    lines, spacing = staff["lines"], staff["spacing"]
    for i in range(len(lines[0])):
        x = lines[0][i][0]
        if any(line[i][0] != x for line in lines) or spacing[i][0] != x:
            return "recovery-misaligned-sample-x"
        if i and lines[0][i - 1][0] >= x:
            return "recovery-duplicate-or-reversed-x"
    return None


def _structurally_eligible(staff: dict[str, Any]) -> str | None:
    """Return a reason code when this staff must not be repaired at all.

    A locally implausible column is a detection defect in an otherwise coherent
    grid. Duplicate or reversed x, non-finite coordinates, an inconsistent line
    count or an ambiguous identity are different in kind: they mean the grid
    itself is not trustworthy, and removing columns would only hide that.
    """
    if not _identity_is_unambiguous(staff):
        return "recovery-ambiguous-identity"
    if not _shapes_are_consistent(staff):
        return "recovery-inconsistent-line-count"
    if not _coordinates_are_finite(staff):
        return "recovery-non-finite-coordinates"
    return _samples_share_increasing_x(staff)


def _implausible_indices(staff: dict[str, Any], height: float) -> list[int]:
    lines = staff["lines"]
    maximum = height / MAX_STAFF_SPACE_IMAGE_FRACTION
    bad = []
    for i in range(len(lines[0])):
        ys = [lines[j][i][1] for j in range(5)]
        unit = (ys[4] - ys[0]) / 4
        gaps = [ys[j + 1] - ys[j] for j in range(4)]
        if not MIN_STAFF_SPACE_PIXELS <= unit <= maximum or any(
            not MIN_GAP_RATIO * unit <= gap <= MAX_GAP_RATIO * unit for gap in gaps
        ):
            bad.append(i)
    return bad


def _runs(indices: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in indices:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _unit_at(lines: list[list[list[float]]], i: int) -> float:
    return (lines[4][i][1] - lines[0][i][1]) / 4


def _local_slope(lines: list[list[list[float]]], near: int, far: int) -> float | None:
    step = lines[0][near][0] - lines[0][far][0]
    if step == 0:
        return None
    return (lines[0][near][1] - lines[0][far][1]) / step


def _interior_run_is_safe(
    staff: dict[str, Any], run: list[int], limits: RecoveryLimits
) -> RecoveryOutcome | None:
    """Return a declining outcome, or ``None`` when this run may be removed."""
    lines = staff["lines"]
    left, right = run[0] - 1, run[-1] + 1
    unit_left, unit_right = _unit_at(lines, left), _unit_at(lines, right)
    reference = (unit_left + unit_right) / 2
    if reference <= 0:
        return _declined(
            "recovery-unusable-neighbours",
            "Surviving neighbours do not define a usable staff space",
            staff,
            sample_index=run[0],
        )
    span = lines[0][right][0] - lines[0][left][0]
    if span / reference > limits.max_interior_span_spaces:
        return _declined(
            "recovery-span-too-long",
            "Removing these samples would bridge too long an unsupported span",
            staff,
            sample_index=run[0],
            x=lines[0][run[0]][0],
            limits={
                "span_spaces": round(span / reference, 4),
                "max_interior_span_spaces": limits.max_interior_span_spaces,
            },
        )
    disagreement = abs(unit_right - unit_left) / reference
    if disagreement > limits.max_neighbour_unit_disagreement:
        return _declined(
            "recovery-neighbours-disagree",
            "Surviving neighbours disagree about the local staff spacing",
            staff,
            sample_index=run[0],
            x=lines[0][run[0]][0],
            limits={
                "unit_disagreement": round(disagreement, 4),
                "max_neighbour_unit_disagreement": limits.max_neighbour_unit_disagreement,
            },
        )
    chord_slope = (lines[0][right][1] - lines[0][left][1]) / span
    for near, far in ((left, left - 1), (right, right + 1)):
        if not 0 <= far < len(lines[0]):
            continue
        local = _local_slope(lines, near, far)
        if local is None:
            continue
        departure = abs(chord_slope - local) * span / reference
        if departure > limits.max_slope_disagreement_spaces:
            return _declined(
                "recovery-slope-disagrees",
                "The chord departs from the slope of the surrounding samples",
                staff,
                sample_index=run[0],
                x=lines[0][run[0]][0],
                limits={
                    "slope_departure_spaces": round(departure, 4),
                    "max_slope_disagreement_spaces": limits.max_slope_disagreement_spaces,
                },
            )
    # The decisive check: at every removed column, the outer lines the consumer
    # will interpolate must already sit close to where the chord puts them. A
    # grid that passes only because its survivors line up proves nothing about
    # the columns in between.
    for index in run:
        t = (lines[0][index][0] - lines[0][left][0]) / span
        for line in (0, 4):
            chord = lines[line][left][1] + (lines[line][right][1] - lines[line][left][1]) * t
            deviation = abs(lines[line][index][1] - chord) / reference
            if deviation > limits.max_chord_deviation_spaces:
                return _declined(
                    "recovery-displacement-too-large",
                    "A removed sample sits too far from the interpolation that replaces it",
                    staff,
                    sample_index=index,
                    x=lines[0][index][0],
                    limits={
                        "line": line,
                        "chord_deviation_spaces": round(deviation, 4),
                        "max_chord_deviation_spaces": limits.max_chord_deviation_spaces,
                    },
                )
    return None


def _edge_run_is_safe(
    staff: dict[str, Any], run: list[int], size: int, limits: RecoveryLimits
) -> RecoveryOutcome | None:
    lines = staff["lines"]
    leading = run[0] == 0
    survivor = run[-1] + 1 if leading else run[0] - 1
    if not 0 <= survivor < size:
        return _declined(
            "recovery-no-surviving-neighbour",
            "Trimming this edge would leave no surviving sample beside it",
            staff,
            sample_index=run[0],
        )
    reference = _unit_at(lines, survivor)
    if reference <= 0:
        return _declined(
            "recovery-unusable-neighbours",
            "The surviving edge sample does not define a usable staff space",
            staff,
            sample_index=survivor,
        )
    trimmed = abs(lines[0][survivor][0] - lines[0][run[0] if leading else run[-1]][0])
    if trimmed / reference > limits.max_edge_trim_spaces:
        return _declined(
            "recovery-edge-trim-too-long",
            "Trimming this edge would drop too much of the staff",
            staff,
            sample_index=run[0],
            x=lines[0][run[0]][0],
            limits={
                "trimmed_spaces": round(trimmed / reference, 4),
                "max_edge_trim_spaces": limits.max_edge_trim_spaces,
            },
        )
    return None


def _rebuilt(staff: dict[str, Any], keep: list[int]) -> dict[str, Any]:
    """A new staff from the surviving columns, with every kept value untouched."""
    lines = [[list(line[i]) for i in keep] for line in staff["lines"]]
    spacing = [list(staff["spacing"][i]) for i in keep]
    return {
        **{
            key: staff[key]
            for key in ("staff_id", "staff_group_index", "staff_index", "system_index")
        },
        "lines": lines,
        "spacing": spacing,
        "extent": [
            lines[0][0][0],
            min(p[1] for p in lines[0]),
            lines[0][-1][0],
            max(p[1] for p in lines[4]),
        ],
    }


def _removal_budget_is_met(
    staff: dict[str, Any], implausible: list[int], size: int, limits: RecoveryLimits
) -> RecoveryOutcome | None:
    """Whole-staff budgets, checked before any individual run is examined."""
    if not implausible:
        return _declined(
            "recovery-not-applicable", "This staff has no implausible samples to remove", staff
        )
    if len(implausible) / size > limits.max_removed_fraction:
        return _declined(
            "recovery-too-many-samples",
            "Too much of this staff is implausible to repair by removal",
            staff,
            limits={
                "removed_fraction": round(len(implausible) / size, 4),
                "max_removed_fraction": limits.max_removed_fraction,
            },
        )
    if size - len(implausible) < limits.min_surviving_samples:
        return _declined(
            "recovery-too-few-survivors",
            "Too few original samples would survive the removal",
            staff,
            limits={
                "surviving": size - len(implausible),
                "min_surviving_samples": limits.min_surviving_samples,
            },
        )
    return None


def _runs_are_safe(
    staff: dict[str, Any], implausible: list[int], size: int, limits: RecoveryLimits
) -> tuple[set[str], RecoveryOutcome | None]:
    """Check every run of adjacent defects, and report what kinds were found."""
    kinds: set[str] = set()
    for run in _runs(implausible):
        if len(run) > limits.max_removed_run:
            return kinds, _declined(
                "recovery-run-too-long",
                "Too many adjacent samples are implausible to repair by removal",
                staff,
                sample_index=run[0],
                limits={"run": len(run), "max_removed_run": limits.max_removed_run},
            )
        if run[0] == 0 or run[-1] == size - 1:
            kinds.add("leading-edge" if run[0] == 0 else "trailing-edge")
            declined = _edge_run_is_safe(staff, run, size, limits)
        else:
            kinds.add("interior")
            declined = _interior_run_is_safe(staff, run, limits)
        if declined is not None:
            return kinds, declined
    return kinds, None


def recover_staff(
    staff: dict[str, Any],
    *,
    width: float,
    height: float,
    limits: RecoveryLimits = DEFAULT_LIMITS,
) -> RecoveryOutcome:
    """Try to make one exported staff valid by dropping implausible columns.

    Returns an outcome whose ``staff`` is ``None`` when no safe removal exists.
    The caller keeps the original rejection in that case; recovery never widens
    what is publishable, it only restores grids that were already trustworthy
    apart from a bounded local defect.
    """
    ineligible = _structurally_eligible(staff)
    if ineligible is not None:
        return _declined(
            ineligible,
            "This staff's grid is not eligible for bounded sample removal",
            staff,
        )
    size = len(staff["lines"][0])
    implausible = _implausible_indices(staff, height)
    budget = _removal_budget_is_met(staff, implausible, size, limits)
    if budget is not None:
        return budget
    kinds, declined = _runs_are_safe(staff, implausible, size, limits)
    if declined is not None:
        return declined

    keep = [i for i in range(size) if i not in set(implausible)]
    rebuilt = _rebuilt(staff, keep)
    # Removing an extreme sample would shrink the staff's vertical extent, and the
    # consumer derives a system's placement boundary from the gap between extents.
    # A neighbouring system must not quietly gain room because a sample went away.
    if [rebuilt["extent"][1], rebuilt["extent"][3]] != [staff["extent"][1], staff["extent"][3]]:
        return _declined(
            "recovery-changes-vertical-extent",
            "Removing these samples would move this staff's placement boundary",
            staff,
            limits={"before": staff["extent"], "after": rebuilt["extent"]},
        )
    try:
        validate_physical_staff(rebuilt, width, height)
    except AnnotationGeometryError as error:
        return _declined(
            "recovery-still-invalid",
            "The staff is still out of contract after removing its implausible samples",
            staff,
            limits={"remaining_reason": error.diagnostic.reason},
        )
    repair = StaffRepair(
        staff_id=staff["staff_id"],
        staff_group_index=staff["staff_group_index"],
        staff_index=staff["staff_index"],
        kind="mixed" if len(kinds) > 1 else next(iter(kinds)),
        removed=[
            {"x": staff["lines"][0][i][0], "reason": "implausible-staff-spacing"}
            for i in implausible
        ],
    )
    return RecoveryOutcome(staff=rebuilt, repair=repair)
