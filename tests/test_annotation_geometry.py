# Pytest assertions provide the diagnostic comparison output these tests need.
# ruff: noqa: S101
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from homr.model import Staff, StaffPoint
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from homr.visual_sidecar.annotation_capture import (
    CAPTURE_DIRECTORY_ENV,
    write_annotation_capture,
)
from homr.visual_sidecar.annotation_geometry import (
    AnnotationGeometryError,
    export_staff_geometry,
    originating_diagnostic,
    validate_annotation_geometry,
)


def geometry() -> tuple[PredictionCoordinateTransform, Staff]:
    transform = PredictionCoordinateTransform(
        source_image_size=(500, 800),
        autocrop_box=(30, 40, 400, 600),
        cropped_size=(400, 600),
        resized_size=(200, 200),
        resize_scale=(0.5, 1 / 3),
        prediction_size=(200, 200),
    )
    staff = Staff(
        [
            StaffPoint(
                x,
                [20 + i * 5 + x * 0.01 for i in range(5)]
                + [90 + i * 5 + x * 0.01 for i in range(5)],
                0,
            )
            for x in (10, 80, 180)
        ]
    )
    return transform, staff


def test_exports_cropped_nonuniform_scaled_skewed_grand_staff() -> None:
    transform, staff = geometry()
    builder = VisualSidecarBuilder(transform)
    builder.add_staff_geometry(4, staff, 2)
    sidecar = builder.to_json_dict()
    validate_annotation_geometry(sidecar)
    upper, lower = sidecar["annotation_geometry"]["staffs"]
    assert upper["lines"][0][0] == pytest.approx([50, 100.3])
    assert upper["spacing"][0] == pytest.approx([50, 15])
    assert lower["lines"][0][0] == pytest.approx([50, 310.3])
    assert upper["system_index"] == lower["system_index"] == 2
    assert (upper["staff_group_index"], lower["staff_index"]) == (4, 1)
    assert upper["extent"][2] == 390
    assert "annotation_geometry_error" not in sidecar
    assert "annotation_geometry_rejection" not in sidecar


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("nan", "coordinates-out-of-bounds"),
        ("reversed", "implausible-staff-spacing"),
        ("spacing", "spacing-disagreement"),
        ("identity", "duplicate-staff-identity"),
        ("reference", "missing-staff-reference"),
        ("extent", "extent-disagreement"),
        ("duplicate_x", "non-increasing-x"),
        ("version", "unsupported-version"),
        ("ragged", "inconsistent-sample-count"),
    ],
)
def test_rejects_malformed_geometry_with_a_stable_reason_code(fault: str, reason: str) -> None:
    transform, staff = geometry()
    data: dict[str, Any] = {
        "source_image_size": [500, 800],
        "annotation_geometry": {
            "version": 1,
            "staffs": export_staff_geometry(staff, 0, 0, transform).staffs,
        },
        "visual_groups": [],
    }
    first = data["annotation_geometry"]["staffs"][0]
    if fault == "nan":
        first["lines"][0][0][0] = float("nan")
    elif fault == "reversed":
        first["lines"].reverse()
    elif fault == "spacing":
        first["spacing"][0][1] = 1
    elif fault == "identity":
        data["annotation_geometry"]["staffs"].append(copy.deepcopy(first))
    elif fault == "reference":
        data["visual_groups"] = [{"staff_group_index": 100, "staff_index": 0}]
    elif fault == "duplicate_x":
        for line in first["lines"]:
            line[1][0] = line[0][0]
        first["spacing"][1][0] = first["spacing"][0][0]
    elif fault == "version":
        data["annotation_geometry"]["version"] = 2
    elif fault == "ragged":
        for line in first["lines"][1:]:
            del line[-1]
    else:
        first["extent"][0] += 2
    with pytest.raises(AnnotationGeometryError) as raised:
        validate_annotation_geometry(data)
    assert raised.value.diagnostic.reason == reason
    assert raised.value.diagnostic.stage == "producer-validation"
    # The concise summary existing consumers already read stays a plain string.
    assert str(raised.value) == raised.value.diagnostic.message


def test_spacing_rejection_reports_the_sample_and_the_limits_it_violated() -> None:
    transform, _ = geometry()
    staff = Staff([StaffPoint(x, [10, 20, 30, 40, 90], 0) for x in (10, 100)])
    data: dict[str, Any] = {
        "source_image_size": [500, 800],
        "annotation_geometry": {
            "version": 1,
            "staffs": export_staff_geometry(staff, 3, 1, transform).staffs,
        },
        "visual_groups": [],
    }
    with pytest.raises(AnnotationGeometryError) as raised:
        validate_annotation_geometry(data)
    diagnostic = raised.value.diagnostic
    assert diagnostic.reason == "implausible-staff-spacing"
    assert (diagnostic.staff_group_index, diagnostic.system_index) == (3, 1)
    assert diagnostic.staff_id == "staff-3-0"
    assert diagnostic.sample_index == 0
    assert diagnostic.gaps is not None and len(diagnostic.gaps) == 4
    assert diagnostic.limits["max_gap_ratio"] == 1.5
    assert json.dumps(diagnostic.to_dict())


def test_old_sidecars_remain_valid_and_degenerate_grids_do_not_advertise_geometry() -> None:
    validate_annotation_geometry({"version": 3})
    transform, _ = geometry()
    staff = Staff([StaffPoint(10, [10, 20, 30, 40, 50], 0)])
    export = export_staff_geometry(staff, 0, 0, transform)
    assert export.staffs == []
    assert [d.reason for d in export.diagnostics] == ["export-insufficient-samples"]


@pytest.mark.parametrize(
    ("grid", "reason"),
    [
        ([StaffPoint(10, [10, 20, 30, 40, 50], 0)], "export-insufficient-samples"),
        (
            [StaffPoint(x, [10, 20, 30, 40, 50], 0) for x in (10, 100)],
            None,
        ),
    ],
)
def test_export_reports_structural_defects_instead_of_an_empty_list(
    grid: list[StaffPoint], reason: str | None
) -> None:
    transform, _ = geometry()
    export = export_staff_geometry(Staff(grid), 0, 0, transform)
    assert [d.reason for d in export.diagnostics] == ([reason] if reason else [])


def test_export_reports_a_ragged_sample_grid() -> None:
    transform, _ = geometry()
    staff = Staff([StaffPoint(x, [10, 20, 30, 40, 50], 0) for x in (10, 100, 200)])
    # A staff whose samples disagree on their line count used to export silently.
    staff.grid[1].y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    export = export_staff_geometry(staff, 7, 2, transform)
    assert export.staffs == []
    diagnostic = export.diagnostics[0]
    assert diagnostic.reason == "export-ragged-sample-grid"
    assert (diagnostic.stage, diagnostic.sample_index, diagnostic.x) == ("export", 1, 100.0)


def test_invalid_detected_curves_disable_only_optional_annotation_capability() -> None:
    transform, _ = geometry()
    staff = Staff([StaffPoint(x, [10, 20, 30, 40, 90], 0) for x in [10, 100]])
    builder = VisualSidecarBuilder(transform)
    builder.add_staff_geometry(0, staff, 0)
    output = builder.to_json_dict()
    assert output["version"] == 3
    assert output["notes"] == []
    assert "annotation_geometry" not in output
    assert "spacing" in output["annotation_geometry_error"]
    rejection = output["annotation_geometry_rejection"]
    assert rejection["reason"] == "implausible-staff-spacing"
    assert rejection["stage"] == "producer-validation"
    assert rejection["message"] == output["annotation_geometry_error"]


def test_export_failure_survives_as_the_reason_rather_than_a_membership_symptom() -> None:
    transform, valid_staff = geometry()
    builder = VisualSidecarBuilder(transform)
    builder.add_staff_geometry(0, valid_staff, 0)
    # A staff whose grid never reached two samples exports nothing at all, so its
    # visual groups reference geometry that was never written. The membership
    # complaint is the symptom; the missing export is the cause.
    builder.add_staff_geometry(1, Staff([StaffPoint(10, [10, 20, 30, 40, 50], 0)]), 1)

    sidecar = builder.to_json_dict()
    sidecar["visual_groups"] = [{"staff_group_index": 1, "staff_index": 0}]
    with pytest.raises(AnnotationGeometryError) as raised:
        validate_annotation_geometry(sidecar)
    assert raised.value.diagnostic.reason == "missing-staff-reference"

    resolved = originating_diagnostic(
        raised.value.diagnostic, list(builder.state.annotation_diagnostics)
    )
    assert resolved.reason == "export-insufficient-samples"
    assert resolved.staff_group_index == 1


def test_duplicate_staff_group_registration_is_rejected_not_overwritten() -> None:
    transform, staff = geometry()
    builder = VisualSidecarBuilder(transform)
    builder.add_staff_geometry(4, staff, 2)
    exported = copy.deepcopy(builder.state.annotation_staffs[4])
    builder.add_staff_geometry(4, Staff([StaffPoint(10, [10, 20, 30, 40, 50], 0)]), 3)
    assert builder.state.annotation_staffs[4] == exported
    assert [d.reason for d in builder.state.annotation_diagnostics] == [
        "export-duplicate-staff-group"
    ]


def test_nothing_exported_at_all_still_reports_why() -> None:
    transform, _ = geometry()
    builder = VisualSidecarBuilder(transform)
    builder.add_staff_geometry(0, Staff([StaffPoint(10, [10, 20, 30, 40, 50], 0)]), 0)
    output = builder.to_json_dict()
    assert "annotation_geometry" not in output
    assert output["annotation_geometry_rejection"]["reason"] == "export-insufficient-samples"
    assert output["annotation_geometry_rejection"]["stage"] == "export"


def test_pre_validation_capture_is_opt_in_and_keeps_full_grids_out_of_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transform, staff = geometry()
    monkeypatch.delenv(CAPTURE_DIRECTORY_ENV, raising=False)
    builder = VisualSidecarBuilder(transform)
    builder.add_staff_geometry(0, staff, 0)
    output = builder.to_json_dict()
    assert not list(tmp_path.iterdir())
    assert "lines" not in json.dumps(output.get("annotation_geometry_rejection", {}))

    monkeypatch.setenv(CAPTURE_DIRECTORY_ENV, str(tmp_path))
    export = export_staff_geometry(staff, 0, 0, transform)
    path = write_annotation_capture(
        staffs=export.staffs,
        diagnostics=export.diagnostics,
        transform=transform,
        source_image_size=[500, 800],
        label="rejected",
    )
    assert path is not None
    captured = json.loads(path.read_text(encoding="utf-8"))
    assert captured["pre_validation_staffs"][0]["lines"][0][0] == pytest.approx([50, 100.3])
    assert captured["coordinate_transform"]["resized_size"] == [200, 200]
    assert captured["producer"]["import_location"].endswith("homr")
    assert "available_onnxruntime_providers" in captured["inference"]
