"""Optional v3 annotation geometry, always in source-raster coordinates."""

import math
from typing import Any

from homr.model import Staff
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform

ANNOTATION_GEOMETRY_VERSION = 1


def export_staff_geometry(
    staff: Staff,
    group_index: int,
    system_index: int,
    transform: PredictionCoordinateTransform,
) -> list[dict[str, Any]]:
    grid = sorted(staff.grid, key=lambda point: point.x)
    if len(grid) < 2:
        return []
    count = len(grid[0].y)
    if count % 5 or any(len(point.y) != count for point in grid):
        return []
    result = []
    for index in range(count // 5):
        lines = [
            [
                list(transform.prediction_point_to_source((p.x, p.y[index * 5 + line])))
                for p in grid
            ]
            for line in range(5)
        ]
        spacing = [
            [lines[0][i][0], (lines[4][i][1] - lines[0][i][1]) / 4] for i in range(len(grid))
        ]
        result.append(
            {
                "staff_id": f"staff-{group_index}-{index}",
                "staff_group_index": group_index,
                "staff_index": index,
                "system_index": system_index,
                "lines": lines,
                "spacing": spacing,
                "extent": [
                    lines[0][0][0],
                    min(p[1] for p in lines[0]),
                    lines[0][-1][0],
                    max(p[1] for p in lines[4]),
                ],
            }
        )
    return result


def validate_annotation_geometry(sidecar: dict[str, Any]) -> None:
    """Absent geometry is supported; malformed advertised geometry is an error."""
    geometry = sidecar.get("annotation_geometry")
    if geometry is None:
        return
    if not isinstance(geometry, dict) or geometry.get("version") != 1:
        raise ValueError("Unsupported annotation geometry version")
    staffs = geometry.get("staffs")
    if not isinstance(staffs, list) or not staffs:
        raise ValueError("Annotation geometry requires physical staffs")
    width, height = sidecar["source_image_size"]
    identities = set()
    ids = set()

    def finite(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    def pair(value: Any) -> bool:
        return isinstance(value, list) and len(value) == 2 and all(finite(v) for v in value)

    for staff in staffs:
        if not isinstance(staff, dict):
            raise ValueError("Invalid physical staff")
        indices = [staff.get(key) for key in ("staff_group_index", "staff_index", "system_index")]
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices):
            raise ValueError("Invalid physical staff indices")
        identity = tuple(indices[:2])
        staff_id = staff.get("staff_id")
        if (
            not isinstance(staff_id, str)
            or not staff_id
            or staff_id in ids
            or identity in identities
        ):
            raise ValueError("Duplicate or invalid physical staff identity")
        identities.add(identity)
        ids.add(staff_id)
        lines, spacing, extent = (staff.get(key) for key in ("lines", "spacing", "extent"))
        if (
            not isinstance(lines, list)
            or len(lines) != 5
            or not all(isinstance(line, list) for line in lines)
        ):
            raise ValueError("Physical staff requires five sampled lines")
        size = len(lines[0])
        if size < 2 or any(len(line) != size for line in lines):
            raise ValueError("Staff lines require a common sample grid")
        if any(
            not pair(p) or not (0 <= p[0] <= width and 0 <= p[1] <= height)
            for line in lines
            for p in line
        ):
            raise ValueError("Staff coordinates must be finite and inside the source image")
        if any(lines[0][i][0] >= lines[0][i + 1][0] for i in range(size - 1)):
            raise ValueError("Staff samples must have increasing x")
        if (
            not isinstance(spacing, list)
            or len(spacing) != size
            or not all(pair(p) for p in spacing)
        ):
            raise ValueError("Invalid local spacing samples")
        for i in range(size):
            x = lines[0][i][0]
            gaps = [lines[j + 1][i][1] - lines[j][i][1] for j in range(4)]
            unit = sum(gaps) / 4
            if any(line[i][0] != x for line in lines) or spacing[i][0] != x:
                raise ValueError("Staff samples must share x coordinates")
            if not 1 <= unit <= height / 10 or any(
                not 0.5 * unit <= gap <= 1.5 * unit for gap in gaps
            ):
                raise ValueError("Unordered lines or implausible staff spacing")
            if not math.isclose(spacing[i][1], unit, abs_tol=0.01):
                raise ValueError("Local spacing disagrees with staff lines")
        expected = [
            lines[0][0][0],
            min(p[1] for p in lines[0]),
            lines[0][-1][0],
            max(p[1] for p in lines[4]),
        ]
        if (
            not isinstance(extent, list)
            or len(extent) != 4
            or any(not finite(v) for v in extent)
            or any(abs(a - b) > 0.01 for a, b in zip(extent, expected, strict=True))
        ):
            raise ValueError("Staff extent disagrees with sampled lines")
    for group in sidecar.get("visual_groups", []):
        if (group["staff_group_index"], group["staff_index"]) not in identities:
            raise ValueError("Visual group references missing physical staff geometry")
