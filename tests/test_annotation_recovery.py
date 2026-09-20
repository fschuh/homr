# Pytest assertions provide the diagnostic comparison output these tests need.
# ruff: noqa: S101
import copy
import json
import math
from pathlib import Path

import pytest

from homr.model import Staff, StaffPoint
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from homr.visual_sidecar.annotation_geometry import (
    AnnotationGeometryError,
    export_staff_geometry,
    validate_physical_staff,
)
from homr.visual_sidecar.annotation_recovery import (
    DEFAULT_LIMITS,
    RecoveryLimits,
    recover_staff,
)

WIDTH, HEIGHT = 2550, 3301
FIXTURE = Path(__file__).parent / "fixtures" / "fillmore_page2_rejected_staff.json"


def grid(
    count: int = 40,
    *,
    step: float = 13.0,
    unit: float = 18.0,
    top: float = 500.0,
    slope: float = 0.0,
    curvature: float = 0.0,
) -> dict:
    """A smooth, valid five-line staff, optionally skewed and curved."""
    xs = [100.0 + i * step for i in range(count)]
    lines = []
    for line in range(5):
        points = []
        for x in xs:
            offset = (x - xs[0]) / (xs[-1] - xs[0])
            y = top + line * unit + slope * (x - xs[0]) + curvature * offset * (1 - offset)
            points.append([x, y])
        lines.append(points)
    return {
        "staff_id": "staff-0-0",
        "staff_group_index": 0,
        "staff_index": 0,
        "system_index": 0,
        "lines": lines,
        "spacing": [[x, unit] for x in xs],
        "extent": [
            lines[0][0][0],
            min(p[1] for p in lines[0]),
            lines[0][-1][0],
            max(p[1] for p in lines[4]),
        ],
    }


#: Enough to push one adjacent gap past 1.5 times the sample's mean gap, which is
#: what the producer's validator actually tests, on the default unit of 18 pixels.
INNER_DEFECT = 11.0


def _refresh(staff: dict) -> None:
    """Restore the derived spacing and extent after moving whole lines about."""
    lines = staff["lines"]
    staff["spacing"] = [
        [lines[0][i][0], (lines[4][i][1] - lines[0][i][1]) / 4] for i in range(len(lines[0]))
    ]
    staff["extent"] = [
        lines[0][0][0],
        min(p[1] for p in lines[0]),
        lines[0][-1][0],
        max(p[1] for p in lines[4]),
    ]


def displace(staff: dict, index: int, line: int, amount: float = INNER_DEFECT) -> dict:
    """Move one inner line at one column, the way a detection defect does."""
    staff = copy.deepcopy(staff)
    staff["lines"][line][index][1] += amount
    return staff


def recovered(staff: dict, limits: RecoveryLimits = DEFAULT_LIMITS) -> dict | None:
    return recover_staff(staff, width=WIDTH, height=HEIGHT, limits=limits).staff


def reason(staff: dict, limits: RecoveryLimits = DEFAULT_LIMITS) -> str:
    outcome = recover_staff(staff, width=WIDTH, height=HEIGHT, limits=limits)
    assert outcome.staff is None
    return outcome.diagnostics[0].reason


def test_the_captured_fillmore_outlier_is_removed_and_the_staff_becomes_valid() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    staff = document["staff"]
    width, height = document["captured_from"]["source_image_size"]
    with pytest.raises(AnnotationGeometryError) as rejected:
        validate_physical_staff(staff, width, height)
    assert rejected.value.diagnostic.reason == "implausible-staff-spacing"

    outcome = recover_staff(staff, width=width, height=height)

    assert outcome.staff is not None
    validate_physical_staff(outcome.staff, width, height)
    assert outcome.repair is not None
    assert outcome.repair.kind == "interior"
    assert [round(entry["x"], 3) for entry in outcome.repair.removed] == [439.609]
    assert len(outcome.staff["lines"][0]) == len(staff["lines"][0]) - 1


def test_every_retained_sample_keeps_its_original_coordinates_exactly() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    staff = document["staff"]
    width, height = document["captured_from"]["source_image_size"]
    before = copy.deepcopy(staff)

    outcome = recover_staff(staff, width=width, height=height)

    assert staff == before, "recovery must not mutate the exported staff it was given"
    assert outcome.staff is not None and outcome.repair is not None
    removed = {entry["x"] for entry in outcome.repair.removed}
    kept = [x for x in (p[0] for p in before["lines"][0]) if x not in removed]
    assert [p[0] for p in outcome.staff["lines"][0]] == kept
    for line in range(5):
        original = {p[0]: p[1] for p in before["lines"][line]}
        for x, y in outcome.staff["lines"][line]:
            assert y == original[x]


def test_a_valid_grid_is_never_altered_and_recovery_reports_why_it_did_nothing() -> None:
    for staff in (grid(), grid(slope=0.04), grid(curvature=9.0), grid(slope=-0.03, curvature=-7.0)):
        validate_physical_staff(staff, WIDTH, HEIGHT)
        outcome = recover_staff(staff, width=WIDTH, height=HEIGHT)
        assert outcome.staff is None
        assert outcome.repair is None
        assert outcome.diagnostics[0].reason == "recovery-not-applicable"


def test_recovery_is_deterministic() -> None:
    staff = displace(grid(), 12, 2)
    first = recover_staff(staff, width=WIDTH, height=HEIGHT)
    second = recover_staff(copy.deepcopy(staff), width=WIDTH, height=HEIGHT)
    assert first.repair is not None and second.repair is not None
    assert first.staff == second.staff
    assert first.repair.to_dict() == second.repair.to_dict()


def test_a_skewed_and_curved_staff_still_recovers_from_one_bad_column() -> None:
    staff = displace(grid(slope=0.05, curvature=11.0), 20, 1)
    result = recovered(staff)
    assert result is not None
    validate_physical_staff(result, WIDTH, HEIGHT)


def test_a_nonuniformly_scaled_export_recovers_the_same_column() -> None:
    # The transform crops and scales x and y by different factors; the defect and
    # the limits are both expressed in staff spaces, so neither moves with it.
    transform = PredictionCoordinateTransform(
        source_image_size=(WIDTH, HEIGHT),
        autocrop_box=(40, 60, 2000, 2600),
        cropped_size=(2000, 2600),
        resized_size=(1000, 1600),
        resize_scale=(0.5, 1600 / 2600),
        prediction_size=(1000, 1600),
    )
    points = [
        StaffPoint(float(x), [200.0 + i * 12.0 for i in range(5)], 0) for x in range(0, 400, 9)
    ]
    points[15].y[2] += 8.0
    export = export_staff_geometry(Staff(points), 0, 0, transform)
    staff = export.staffs[0]
    with pytest.raises(AnnotationGeometryError):
        validate_physical_staff(staff, WIDTH, HEIGHT)

    outcome = recover_staff(staff, width=WIDTH, height=HEIGHT)

    assert outcome.staff is not None
    validate_physical_staff(outcome.staff, WIDTH, HEIGHT)
    assert outcome.repair is not None
    assert len(outcome.repair.removed) == 1


def test_bounded_runs_of_adjacent_defects_recover_and_longer_ones_do_not() -> None:
    close = grid(count=60)
    for index in (20, 21, 22):
        close["lines"][2][index][1] += INNER_DEFECT
    assert recovered(close) is not None

    tight = RecoveryLimits(max_removed_run=2)
    assert reason(close, tight) == "recovery-run-too-long"


def test_a_long_missing_interval_is_refused_rather_than_bridged() -> None:
    staff = grid(count=200, step=13.0)
    # Nine adjacent columns is well over half a bar of unsupported interpolation.
    for index in range(20, 29):
        staff["lines"][2][index][1] += INNER_DEFECT
    assert reason(staff) == "recovery-run-too-long"

    wide = grid(count=60, step=40.0)
    for index in (20, 21, 22):
        wide["lines"][2][index][1] += INNER_DEFECT
    assert reason(wide) == "recovery-span-too-long"


def test_a_removed_sample_whose_outer_lines_moved_too_far_is_refused() -> None:
    # The defect is in an outer line, so the interpolation that replaces the column
    # would no longer follow the printed staff. A geometric fit of the survivors
    # alone would not have noticed.
    staff = grid()
    staff["lines"][0][15][1] -= 20.0
    staff["spacing"][15][1] = (staff["lines"][4][15][1] - staff["lines"][0][15][1]) / 4
    assert reason(staff) == "recovery-displacement-too-large"


def test_neighbours_that_disagree_about_the_spacing_refuse_the_removal() -> None:
    # Every surviving column is internally plausible, but the staff space steps from
    # 18 to 25 pixels right across the defect, so nothing here says what the removed
    # column should have been.
    staff = grid(count=40)
    for index in range(16, 40):
        for line in range(5):
            staff["lines"][line][index][1] = 500.0 + line * 25.0
    staff["lines"][2][15][1] += INNER_DEFECT
    _refresh(staff)
    assert reason(staff) == "recovery-neighbours-disagree"


def test_an_edge_defect_is_trimmed_only_while_the_lost_span_stays_bounded() -> None:
    near = grid(count=40, step=13.0)
    near["lines"][2][0][1] += INNER_DEFECT
    result = recovered(near)
    assert result is not None
    assert result["extent"][0] == near["lines"][0][1][0]
    # The trimmed notes simply fall outside the staff extent from now on.
    assert result["extent"][0] > near["extent"][0]

    far = grid(count=200, step=13.0)
    for index in range(7):
        far["lines"][2][index][1] += INNER_DEFECT
    assert reason(far) == "recovery-edge-trim-too-long"


def test_trimming_that_would_move_the_placement_boundary_is_refused() -> None:
    # The first column carries this staff's topmost point, so dropping it would
    # raise the staff's own boundary and hand the system above extra room.
    staff = grid(count=40, slope=0.06)
    staff["lines"][2][0][1] += INNER_DEFECT
    assert staff["extent"][1] == staff["lines"][0][0][1]
    assert reason(staff) == "recovery-changes-vertical-extent"


def test_fewer_than_two_surviving_samples_is_refused() -> None:
    staff = grid(count=3)
    for index in (0, 2):
        staff["lines"][2][index][1] += INNER_DEFECT
    assert reason(staff) in {"recovery-too-many-samples", "recovery-too-few-survivors"}

    generous = RecoveryLimits(max_removed_fraction=1.0, max_edge_trim_spaces=100.0)
    assert reason(staff, generous) == "recovery-too-few-survivors"


@pytest.mark.parametrize(
    ("corrupt", "expected"),
    [
        ("nan", "recovery-non-finite-coordinates"),
        ("duplicate_x", "recovery-duplicate-or-reversed-x"),
        ("reversed_x", "recovery-duplicate-or-reversed-x"),
        ("ragged", "recovery-inconsistent-line-count"),
        ("identity", "recovery-ambiguous-identity"),
        ("misaligned_x", "recovery-misaligned-sample-x"),
    ],
)
def test_structural_corruption_is_never_repaired_by_removing_columns(
    corrupt: str, expected: str
) -> None:
    staff = displace(grid(), 12, 2)
    if corrupt == "nan":
        staff["lines"][0][3][1] = math.nan
    elif corrupt == "duplicate_x":
        for line in staff["lines"]:
            line[5][0] = line[4][0]
        staff["spacing"][5][0] = staff["spacing"][4][0]
    elif corrupt == "reversed_x":
        for line in staff["lines"]:
            line[5][0] = line[4][0] - 3
        staff["spacing"][5][0] = staff["spacing"][4][0] - 3
    elif corrupt == "ragged":
        del staff["lines"][3][-1]
    elif corrupt == "identity":
        staff["staff_index"] = -1
    else:
        staff["lines"][3][6][0] += 1
    assert reason(staff) == expected


def test_removal_matches_an_exactly_interpolated_knot_over_the_retained_domain() -> None:
    """Removal and an exact knot agree where the staff still has support.

    They differ only at the ends, which is why the extent and boundary effects of
    a removal are checked on their own rather than folded into this equivalence.
    """
    from homr.visual_sidecar.annotation_geometry import (
        validate_physical_staff as validate,
    )

    staff = displace(grid(count=40), 15, 2)
    result = recovered(staff)
    assert result is not None
    validate(result, WIDTH, HEIGHT)

    removed_x = staff["lines"][0][15][0]
    with_knot = copy.deepcopy(result)
    position = next(i for i, p in enumerate(with_knot["lines"][0]) if p[0] > removed_x)
    for line in range(5):
        left = with_knot["lines"][line][position - 1]
        right = with_knot["lines"][line][position]
        t = (removed_x - left[0]) / (right[0] - left[0])
        with_knot["lines"][line].insert(position, [removed_x, left[1] + (right[1] - left[1]) * t])
    with_knot["spacing"].insert(
        position,
        [
            removed_x,
            (with_knot["lines"][4][position][1] - with_knot["lines"][0][position][1]) / 4,
        ],
    )

    def evaluate(staff_dict: dict, x: float) -> tuple[float, float]:
        xs = [p[0] for p in staff_dict["lines"][0]]
        index = next(i for i in range(1, len(xs)) if xs[i] >= x)
        t = (x - xs[index - 1]) / (xs[index] - xs[index - 1])
        return tuple(
            staff_dict["lines"][line][index - 1][1]
            + (staff_dict["lines"][line][index][1] - staff_dict["lines"][line][index - 1][1]) * t
            for line in (0, 4)
        )

    retained = result["lines"][0]
    samples = [retained[0][0] + (retained[-1][0] - retained[0][0]) * i / 200 for i in range(201)]
    for x in samples:
        a, b = evaluate(result, x), evaluate(with_knot, x)
        assert a[0] == pytest.approx(b[0], abs=1e-9)
        assert a[1] == pytest.approx(b[1], abs=1e-9)


def test_the_builder_publishes_repaired_geometry_with_its_provenance() -> None:
    transform = PredictionCoordinateTransform(
        source_image_size=(WIDTH, HEIGHT),
        autocrop_box=(0, 0, WIDTH, HEIGHT),
        cropped_size=(WIDTH, HEIGHT),
        resized_size=(WIDTH, HEIGHT),
        resize_scale=(1.0, 1.0),
        prediction_size=(WIDTH, HEIGHT),
    )
    points = [
        StaffPoint(float(x), [400.0 + i * 15.0 for i in range(5)], 0) for x in range(0, 500, 11)
    ]
    points[20].y[3] += 9.0
    builder = VisualSidecarBuilder(transform)
    builder.add_staff_geometry(0, Staff(points), 0)

    sidecar = builder.to_json_dict()

    assert "annotation_geometry_rejection" not in sidecar
    assert sidecar["annotation_geometry"]["version"] == 1
    repairs = sidecar["annotation_geometry_repairs"]["staffs"]
    assert len(repairs) == 1
    assert repairs[0]["staff_id"] == "staff-0-0"
    assert repairs[0]["removed_count"] == 1
    assert repairs[0]["kind"] == "interior"
    # The published staff schema itself is untouched, so a v1 reader sees no change.
    assert set(sidecar["annotation_geometry"]["staffs"][0]) == {
        "staff_id",
        "staff_group_index",
        "staff_index",
        "system_index",
        "lines",
        "spacing",
        "extent",
    }


def test_a_staff_recovery_declines_keeps_its_original_rejection() -> None:
    transform = PredictionCoordinateTransform(
        source_image_size=(WIDTH, HEIGHT),
        autocrop_box=(0, 0, WIDTH, HEIGHT),
        cropped_size=(WIDTH, HEIGHT),
        resized_size=(WIDTH, HEIGHT),
        resize_scale=(1.0, 1.0),
        prediction_size=(WIDTH, HEIGHT),
    )
    points = [
        StaffPoint(float(x), [400.0 + i * 15.0 for i in range(5)], 0) for x in range(0, 500, 11)
    ]
    for index in range(10, 30):
        points[index].y[3] += 9.0
    builder = VisualSidecarBuilder(transform)
    builder.add_staff_geometry(0, Staff(points), 0)

    sidecar = builder.to_json_dict()

    assert "annotation_geometry" not in sidecar
    rejection = sidecar["annotation_geometry_rejection"]
    assert rejection["reason"] == "implausible-staff-spacing"
    assert rejection["stage"] == "producer-validation"
    assert "annotation_geometry_repairs" not in sidecar
    assert any(
        entry["stage"] == "recovery" for entry in rejection.get("diagnostics", [])
    ), "the declined recovery is recorded as supporting evidence, not as the reason"
