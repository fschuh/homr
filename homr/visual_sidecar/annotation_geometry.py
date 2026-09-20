"""Optional v3 annotation geometry, always in source-raster coordinates."""

import math
from dataclasses import dataclass, field
from typing import Any

from homr.model import Staff
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform

ANNOTATION_GEOMETRY_VERSION = 1

# Diagnostics are versioned independently of the geometry contract: a reader that
# understands v1 geometry must still tolerate a producer that reports a reason code
# it has never seen.
ANNOTATION_DIAGNOSTICS_VERSION = 1

# Stages, from earliest to latest. A later stage never relabels an earlier failure.
STAGE_EXPORT = "export"
STAGE_PRODUCER_VALIDATION = "producer-validation"

# Spacing limits the validator enforces. They are named so that a diagnostic can
# report the limit that was violated rather than only the value that violated it.
MIN_STAFF_SPACE_PIXELS = 1.0
MAX_STAFF_SPACE_IMAGE_FRACTION = 10.0
MIN_GAP_RATIO = 0.5
MAX_GAP_RATIO = 1.5
SPACING_AGREEMENT_TOLERANCE = 0.01
EXTENT_AGREEMENT_TOLERANCE = 0.01


@dataclass(frozen=True)
class GeometryDiagnostic:
    """One machine-readable reason why annotation geometry is unavailable.

    ``reason`` is a stable code; ``message`` is the concise summary that older
    consumers already read out of ``annotation_geometry_error``.
    """

    reason: str
    stage: str
    message: str
    staff_id: str | None = None
    staff_group_index: int | None = None
    staff_index: int | None = None
    system_index: int | None = None
    sample_index: int | None = None
    x: float | None = None
    gaps: list[float] | None = None
    limits: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "reason": self.reason,
            "stage": self.stage,
            "message": self.message,
        }
        for key in (
            "staff_id",
            "staff_group_index",
            "staff_index",
            "system_index",
            "sample_index",
            "x",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.gaps is not None:
            result["gaps"] = [round(float(gap), 4) for gap in self.gaps]
        if self.limits:
            result["limits"] = self.limits
        return result


class AnnotationGeometryError(ValueError):
    """Validation failure that carries the structured reason with it.

    It remains a ``ValueError`` so the existing evaluation and worker callers,
    which only read ``str(error)``, keep working unchanged.
    """

    def __init__(self, diagnostic: GeometryDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class StaffGeometryExport:
    """Exported physical staffs plus the reasons any of them were not exported.

    The exporter used to return an empty list for several distinct structural
    defects, which made them indistinguishable from "this staff simply has no
    geometry". Each of those defects now produces a diagnostic instead.
    """

    staffs: list[dict[str, Any]]
    diagnostics: list[GeometryDiagnostic] = field(default_factory=list)


def _export_failure(
    reason: str,
    message: str,
    group_index: int,
    system_index: int,
    **extra: Any,
) -> StaffGeometryExport:
    return StaffGeometryExport(
        staffs=[],
        diagnostics=[
            GeometryDiagnostic(
                reason=reason,
                stage=STAGE_EXPORT,
                message=message,
                staff_group_index=group_index,
                system_index=system_index,
                **extra,
            )
        ],
    )


def export_staff_geometry(
    staff: Staff,
    group_index: int,
    system_index: int,
    transform: PredictionCoordinateTransform,
) -> StaffGeometryExport:
    grid = sorted(staff.grid, key=lambda point: point.x)
    if len(grid) < 2:
        return _export_failure(
            "export-insufficient-samples",
            "Detected staff has fewer than two grid samples",
            group_index,
            system_index,
            limits={"minimum_samples": 2, "observed_samples": len(grid)},
        )
    count = len(grid[0].y)
    if count % 5:
        return _export_failure(
            "export-line-count-not-multiple-of-five",
            "Detected staff line count is not a multiple of five",
            group_index,
            system_index,
            limits={"observed_lines": count},
        )
    ragged = next(
        (index for index, point in enumerate(grid) if len(point.y) != count),
        None,
    )
    if ragged is not None:
        return _export_failure(
            "export-ragged-sample-grid",
            "Detected staff samples do not all carry the same lines",
            group_index,
            system_index,
            sample_index=ragged,
            x=float(grid[ragged].x),
            limits={"expected_lines": count, "observed_lines": len(grid[ragged].y)},
        )
    result = []
    for index in range(count // 5):
        lines = [
            [list(transform.prediction_point_to_source((p.x, p.y[index * 5 + line]))) for p in grid]
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
    return StaffGeometryExport(staffs=result)


def _reject(reason: str, message: str, **details: Any) -> AnnotationGeometryError:
    return AnnotationGeometryError(
        GeometryDiagnostic(
            reason=reason, stage=STAGE_PRODUCER_VALIDATION, message=message, **details
        )
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_finite(v) for v in value)


def _staff_identity(staff: dict[str, Any]) -> dict[str, Any]:
    """Whatever identity fields are already trustworthy, for the diagnostic."""
    details: dict[str, Any] = {}
    staff_id = staff.get("staff_id")
    if isinstance(staff_id, str) and staff_id:
        details["staff_id"] = staff_id
    for key in ("staff_group_index", "staff_index", "system_index"):
        value = staff.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            details[key] = value
    return details


def _validate_samples(staff: dict[str, Any], height: float, identity: dict[str, Any]) -> None:
    lines = staff["lines"]
    spacing = staff["spacing"]
    size = len(lines[0])
    max_unit = height / MAX_STAFF_SPACE_IMAGE_FRACTION
    for i in range(size):
        x = lines[0][i][0]
        gaps = [lines[j + 1][i][1] - lines[j][i][1] for j in range(4)]
        unit = sum(gaps) / 4
        if any(line[i][0] != x for line in lines) or spacing[i][0] != x:
            raise _reject(
                "misaligned-sample-x",
                "Staff samples must share x coordinates",
                sample_index=i,
                x=x,
                **identity,
            )
        if not MIN_STAFF_SPACE_PIXELS <= unit <= max_unit or any(
            not MIN_GAP_RATIO * unit <= gap <= MAX_GAP_RATIO * unit for gap in gaps
        ):
            raise _reject(
                "implausible-staff-spacing",
                "Unordered lines or implausible staff spacing",
                sample_index=i,
                x=x,
                gaps=gaps,
                limits={
                    "unit": round(unit, 4),
                    "min_unit": MIN_STAFF_SPACE_PIXELS,
                    "max_unit": round(max_unit, 4),
                    "min_gap_ratio": MIN_GAP_RATIO,
                    "max_gap_ratio": MAX_GAP_RATIO,
                },
                **identity,
            )
        if not math.isclose(spacing[i][1], unit, abs_tol=SPACING_AGREEMENT_TOLERANCE):
            raise _reject(
                "spacing-disagreement",
                "Local spacing disagrees with staff lines",
                sample_index=i,
                x=x,
                limits={
                    "unit": round(unit, 4),
                    "declared": spacing[i][1],
                    "tolerance": SPACING_AGREEMENT_TOLERANCE,
                },
                **identity,
            )


def _validate_staff(
    staff: Any,
    width: float,
    height: float,
    identities: set[tuple[int, int]],
    ids: set[str],
) -> None:
    if not isinstance(staff, dict):
        raise _reject("invalid-staff-entry", "Invalid physical staff")
    identity_details = _staff_identity(staff)
    indices: list[Any] = [
        staff.get(key) for key in ("staff_group_index", "staff_index", "system_index")
    ]
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices):
        raise _reject("invalid-staff-indices", "Invalid physical staff indices", **identity_details)
    identity: tuple[int, int] = (indices[0], indices[1])
    staff_id = staff.get("staff_id")
    if not isinstance(staff_id, str) or not staff_id or staff_id in ids or identity in identities:
        raise _reject(
            "duplicate-staff-identity",
            "Duplicate or invalid physical staff identity",
            **identity_details,
        )
    identities.add(identity)
    ids.add(staff_id)
    lines, spacing, extent = (staff.get(key) for key in ("lines", "spacing", "extent"))
    if (
        not isinstance(lines, list)
        or len(lines) != 5
        or not all(isinstance(line, list) for line in lines)
    ):
        raise _reject(
            "invalid-line-count",
            "Physical staff requires five sampled lines",
            **identity_details,
        )
    size = len(lines[0])
    if size < 2 or any(len(line) != size for line in lines):
        raise _reject(
            "inconsistent-sample-count",
            "Staff lines require a common sample grid",
            **identity_details,
        )
    for line_index, line in enumerate(lines):
        for sample_index, point in enumerate(line):
            if not _pair(point) or not (0 <= point[0] <= width and 0 <= point[1] <= height):
                raise _reject(
                    "coordinates-out-of-bounds",
                    "Staff coordinates must be finite and inside the source image",
                    sample_index=sample_index,
                    limits={
                        "line": line_index,
                        "source_image_size": [width, height],
                        "point": point if _pair(point) else None,
                    },
                    **identity_details,
                )
    for i in range(size - 1):
        if lines[0][i][0] >= lines[0][i + 1][0]:
            raise _reject(
                "non-increasing-x",
                "Staff samples must have increasing x",
                sample_index=i,
                x=lines[0][i][0],
                limits={"next_x": lines[0][i + 1][0]},
                **identity_details,
            )
    if not isinstance(spacing, list) or len(spacing) != size or not all(_pair(p) for p in spacing):
        raise _reject(
            "invalid-spacing-samples", "Invalid local spacing samples", **identity_details
        )
    _validate_samples(staff, height, identity_details)
    expected = [
        lines[0][0][0],
        min(p[1] for p in lines[0]),
        lines[0][-1][0],
        max(p[1] for p in lines[4]),
    ]
    if (
        not isinstance(extent, list)
        or len(extent) != 4
        or any(not _finite(v) for v in extent)
        or any(
            abs(a - b) > EXTENT_AGREEMENT_TOLERANCE for a, b in zip(extent, expected, strict=True)
        )
    ):
        raise _reject(
            "extent-disagreement",
            "Staff extent disagrees with sampled lines",
            limits={
                "expected": [round(v, 4) for v in expected],
                "declared": extent,
                "tolerance": EXTENT_AGREEMENT_TOLERANCE,
            },
            **identity_details,
        )


def validate_physical_staff(staff: dict[str, Any], width: float, height: float) -> None:
    """Validate one physical staff on its own, with the same rules as a sidecar.

    Recovery needs to ask whether a single staff is in contract without inventing
    a surrounding document, and it must get that answer from the same code that
    gates publication rather than from a second, drifting copy of the rules.
    """
    _validate_staff(staff, width, height, set(), set())


def validate_annotation_geometry(sidecar: dict[str, Any]) -> None:
    """Absent geometry is supported; malformed advertised geometry is an error."""
    geometry = sidecar.get("annotation_geometry")
    if geometry is None:
        return
    if not isinstance(geometry, dict) or geometry.get("version") != ANNOTATION_GEOMETRY_VERSION:
        raise _reject("unsupported-version", "Unsupported annotation geometry version")
    staffs = geometry.get("staffs")
    if not isinstance(staffs, list) or not staffs:
        raise _reject("missing-staffs", "Annotation geometry requires physical staffs")
    width, height = sidecar["source_image_size"]
    identities: set[tuple[int, int]] = set()
    ids: set[str] = set()
    for staff in staffs:
        _validate_staff(staff, width, height, identities, ids)
    for group in sidecar.get("visual_groups", []):
        key = (group["staff_group_index"], group["staff_index"])
        if key not in identities:
            raise _reject(
                "missing-staff-reference",
                "Visual group references missing physical staff geometry",
                staff_group_index=key[0],
                staff_index=key[1],
            )


def geometry_diagnostic(error: Exception) -> GeometryDiagnostic:
    """The structured reason behind a validation failure, for any exception type."""
    if isinstance(error, AnnotationGeometryError):
        return error.diagnostic
    return GeometryDiagnostic(
        reason="validation-failed",
        stage=STAGE_PRODUCER_VALIDATION,
        message=str(error),
    )


def rejection_payload(
    primary: GeometryDiagnostic, diagnostics: list[GeometryDiagnostic] | None = None
) -> dict[str, Any]:
    """The structured ``annotation_geometry_rejection`` block consumers read.

    ``primary`` is the originating reason. Supporting diagnostics are listed
    alongside it so that a later stage never has to overwrite an earlier one.
    """
    payload = {"version": ANNOTATION_DIAGNOSTICS_VERSION, **primary.to_dict()}
    supporting = [d for d in diagnostics or [] if d is not primary]
    if supporting:
        payload["diagnostics"] = [d.to_dict() for d in supporting]
    return payload


def originating_diagnostic(
    failure: GeometryDiagnostic, diagnostics: list[GeometryDiagnostic]
) -> GeometryDiagnostic:
    """Prefer the export defect that caused a missing-membership symptom.

    A staff whose export failed leaves its visual groups pointing at geometry that
    was never written. Reporting that symptom would hide the structural defect
    that actually stopped the export, so the earlier stage wins.
    """
    if failure.reason != "missing-staff-reference":
        return failure
    for diagnostic in diagnostics:
        if (
            diagnostic.stage == STAGE_EXPORT
            and diagnostic.staff_group_index == failure.staff_group_index
        ):
            return diagnostic
    return failure
