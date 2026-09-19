# Pytest assertions provide the diagnostic comparison output these tests need.
# ruff: noqa: S101
import copy

import pytest

from homr.model import Staff, StaffPoint
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from homr.visual_sidecar.annotation_geometry import (
    export_staff_geometry,
    validate_annotation_geometry,
)


def geometry():
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


def test_exports_cropped_nonuniform_scaled_skewed_grand_staff():
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


@pytest.mark.parametrize(
    "fault", ["nan", "reversed", "spacing", "identity", "reference", "extent"]
)
def test_rejects_malformed_geometry(fault):
    transform, staff = geometry()
    data = {
        "source_image_size": [500, 800],
        "annotation_geometry": {
            "version": 1,
            "staffs": export_staff_geometry(staff, 0, 0, transform),
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
    else:
        first["extent"][0] += 2
    with pytest.raises(ValueError):
        validate_annotation_geometry(data)


def test_old_sidecars_remain_valid_and_degenerate_grids_do_not_advertise_geometry():
    validate_annotation_geometry({"version": 3})
    transform, _ = geometry()
    staff = Staff([StaffPoint(10, [10, 20, 30, 40, 50], 0)])
    assert export_staff_geometry(staff, 0, 0, transform) == []
