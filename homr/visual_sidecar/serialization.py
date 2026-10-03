import json
from pathlib import Path
from typing import Any

import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.segmentation.config import model_name as segmentation_model_name
from homr.transformer.configs import model_name as transformer_model_name
from homr.visual_sidecar.annotation_capture import write_annotation_capture
from homr.visual_sidecar.annotation_geometry import (
    ANNOTATION_DIAGNOSTICS_VERSION,
    GeometryDiagnostic,
    geometry_diagnostic,
    originating_diagnostic,
    rejection_payload,
    validate_annotation_geometry,
)
from homr.visual_sidecar.chords import ChordResolver
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform
from homr.visual_sidecar.models import (
    PRODUCER_NAME,
    VISUAL_SIDECAR_VERSION,
    MusicXmlRestRecord,
    SidecarState,
    VisualGroup,
    homr_version,
)
from homr.visual_sidecar.rests import REST_VERIFICATION_VERSION, UNVERIFIED

HORIZONTAL_HOLLOW_NOTEHEAD_ASPECT_RATIO = 1.8


class VisualSidecarSerializer:
    def __init__(
        self,
        state: SidecarState,
        coordinate_transform: PredictionCoordinateTransform,
        stem_fragments: list[RotatedBoundingBox],
        chords: ChordResolver,
    ) -> None:
        self.state = state
        self.coordinate_transform = coordinate_transform
        self.stem_fragments = stem_fragments
        self.chords = chords
        self.visual_groups = state.visual_groups
        self.musicxml_notes = state.musicxml_notes

    def to_dict(self) -> dict[str, Any]:
        return self.to_json_dict()

    def _raw_stem_contours_for_output(self) -> list[dict[str, Any]]:
        return [
            {
                "debug_id": stem.debug_id,
                "contour": self.coordinate_transform.prediction_contour_to_source(stem.contours),
                "bbox": self.coordinate_transform.prediction_contour_to_source(stem.polygon),
            }
            for stem in self.stem_fragments
        ]

    def _typical_notehead_angles_by_staff_group(self) -> dict[int, float]:
        angles_by_staff_group: dict[int, list[float]] = {}
        for group in self.visual_groups.values():
            for ellipse in group.notehead_ellipses:
                if ellipse.get("_fit_source") == "fallback":
                    continue
                if abs(float(ellipse["angle"])) < 6:
                    continue
                if float(ellipse["rx"]) <= float(ellipse["ry"]):
                    continue
                angles_by_staff_group.setdefault(group.staff_group_index, []).append(
                    float(ellipse["angle"])
                )

        all_angles = [angle for angles in angles_by_staff_group.values() for angle in angles]
        global_angle = float(np.median(all_angles)) if all_angles else None
        result = {}
        for staff_group_index, angles in angles_by_staff_group.items():
            result[staff_group_index] = float(np.median(angles))
        if global_angle is not None:
            for group in self.visual_groups.values():
                result.setdefault(group.staff_group_index, global_angle)
        return result

    def _notehead_ellipses_for_output(
        self, group: VisualGroup, typical_angle: float | None
    ) -> list[dict[str, Any]]:
        ellipses = []
        for ellipse in group.notehead_ellipses:
            output = {key: value for key, value in ellipse.items() if not key.startswith("_")}
            fit_source = ellipse.get("_fit_source")
            aspect_ratio = float(output["rx"]) / max(float(output["ry"]), 1e-6)
            mask_angle_needs_fallback = (
                fit_source == "mask"
                and abs(float(output["angle"])) < 6
                and aspect_ratio < HORIZONTAL_HOLLOW_NOTEHEAD_ASPECT_RATIO
                and not bool(ellipse.get("_is_hollow", False))
            )
            if typical_angle is not None and (
                fit_source == "fallback" or mask_angle_needs_fallback
            ):
                output["angle"] = round(typical_angle, 3)
            ellipses.append(output)
        return ellipses

    def to_json_dict(self) -> dict[str, Any]:
        typical_angles_by_staff_group = self._typical_notehead_angles_by_staff_group()
        annotation_staffs = [
            staff
            for index in sorted(self.state.annotation_staffs)
            for staff in self.state.annotation_staffs[index]
        ]
        annotation_diagnostics = list(self.state.annotation_diagnostics)
        result = {
            "version": VISUAL_SIDECAR_VERSION,
            **(
                {"annotation_geometry": {"version": 1, "staffs": annotation_staffs}}
                if annotation_staffs
                else {}
            ),
            "producer": {
                "name": PRODUCER_NAME,
                "version": homr_version(),
                # The weights decide the geometry in this file, so they are the provenance
                # that matters. They also move independently of any homr release: upstream
                # shipped three transformer models within v0.7.0 alone.
                "models": {
                    "transformer": transformer_model_name,
                    "segmentation": segmentation_model_name,
                },
            },
            "source_image_size": list(self.coordinate_transform.source_image_size),
            "preprocessing": {
                "autocrop_box": list(self.coordinate_transform.autocrop_box),
                "cropped_size": list(self.coordinate_transform.cropped_size),
                "resized_size": list(self.coordinate_transform.resized_size),
                "resize_scale": [
                    round(self.coordinate_transform.resize_scale[0], 8),
                    round(self.coordinate_transform.resize_scale[1], 8),
                ],
                "prediction_size": list(self.coordinate_transform.prediction_size),
            },
            "notes": [
                {
                    "musicxml_id": record.musicxml_id,
                    "part": record.part,
                    "measure": record.measure,
                    "musicxml_staff_number": record.musicxml_staff_number,
                    "voice": record.voice,
                    "pitch": record.pitch,
                    "duration": record.duration,
                    "match_confidence": record.match_confidence,
                    "visual_group_id": record.visual_group_id,
                    "alignment_method": record.alignment_method,
                }
                for record in self.musicxml_notes
            ],
            "raw_stem_contours": self._raw_stem_contours_for_output(),
            "visual_groups": [
                {
                    "visual_group_id": group.visual_id,
                    "staff_group_index": group.staff_group_index,
                    "staff_index": group.staff_index,
                    "staff_position": group.staff_position,
                    "center": [
                        round(
                            self.coordinate_transform.prediction_point_to_source(
                                group.prediction_center
                            )[0],
                            3,
                        ),
                        round(
                            self.coordinate_transform.prediction_point_to_source(
                                group.prediction_center
                            )[1],
                            3,
                        ),
                    ],
                    "bbox": group.bbox,
                    "notehead_ellipses": self._notehead_ellipses_for_output(
                        group, typical_angles_by_staff_group.get(group.staff_group_index)
                    ),
                    "notehead_contours": group.notehead_contours,
                    "detected_notehead_contours": group.detected_notehead_contours,
                    "refined_notehead_contours": group.refined_notehead_contours,
                    "detected_stem_contours": group.detected_stem_contours,
                    "stem_contours": group.stem_contours,
                    "stem_component_ids": self.chords.stem_component_ids_for_output(group),
                    "is_hollow_notehead": group.is_hollow_notehead,
                    "musicxml_id": group.musicxml_id,
                    "visual_status": group.visual_status,
                    "provenance": group.provenance,
                    "moment_id": group.moment_id,
                    "chord_id": group.chord_id,
                    "repair_actions": group.repair_actions,
                }
                for group in sorted(self.visual_groups.values(), key=lambda g: g.visual_id)
            ],
        }
        self._apply_annotation_geometry_status(result, annotation_staffs, annotation_diagnostics)
        if self.state.musicxml_rests:
            result["rest_verification"] = {
                "version": REST_VERIFICATION_VERSION,
                "rests": [self._rest_for_output(record) for record in self.state.musicxml_rests],
            }
        return result

    def _source_point(self, point: tuple[float, float]) -> list[float]:
        x, y = self.coordinate_transform.prediction_point_to_source(point)
        return [round(x, 3), round(y, 3)]

    def _rest_for_output(self, record: MusicXmlRestRecord) -> dict[str, Any]:
        """Describe one MusicXML rest and the ink, if any, that backs it.

        Positions are in source-image pixels like every other geometry here. A rest
        without a verdict never reached the verifier and is reported as unverified.
        """
        verdict = record.verdict
        output: dict[str, Any] = {
            "rest_id": record.rest_id,
            "part": record.part,
            "measure": record.measure,
            "musicxml_staff_number": record.musicxml_staff_number,
            "voice": record.voice,
            "duration": record.duration,
            "status": verdict.status if verdict is not None else UNVERIFIED,
            "reason": verdict.reason if verdict is not None else "not_verified",
            "staff_group_index": verdict.staff_group_index if verdict is not None else None,
            "staff_index": verdict.staff_index if verdict is not None else None,
            "center": None,
            "staff_lines": [],
            "unit_size": None,
            "position_estimated": verdict.position_estimated if verdict is not None else False,
        }
        if verdict is None or verdict.center is None or verdict.unit_size is None:
            return output
        center_x = verdict.center[0]
        output["center"] = self._source_point(verdict.center)
        output["staff_lines"] = [self._source_point((center_x, y))[1] for y in verdict.staff_lines]
        top = self._source_point((center_x, verdict.center[1]))[1]
        bottom = self._source_point((center_x, verdict.center[1] + verdict.unit_size))[1]
        output["unit_size"] = round(bottom - top, 3)
        return output

    def _apply_annotation_geometry_status(
        self,
        result: dict[str, Any],
        annotation_staffs: list[dict[str, Any]],
        diagnostics: list[GeometryDiagnostic],
    ) -> None:
        """Validate the advertised geometry and record why it was withdrawn.

        The capability is optional: a malformed detection grid must not invalidate
        otherwise usable authoritative note links. What changes here is that the
        reason survives, in a form a consumer can act on rather than match strings against.
        """
        primary: GeometryDiagnostic | None = None
        if annotation_staffs:
            try:
                validate_annotation_geometry(result)
            except ValueError as error:
                failure = geometry_diagnostic(error)
                diagnostics.append(failure)
                primary = originating_diagnostic(failure, diagnostics)
                result.pop("annotation_geometry", None)
        elif diagnostics:
            # Nothing was exported at all. Without this the page would look like an
            # ordinary score that simply carries no optional geometry.
            primary = diagnostics[0]
        if primary is not None:
            result["annotation_geometry_error"] = primary.message
            result["annotation_geometry_rejection"] = rejection_payload(primary, diagnostics)
        if self.state.annotation_repairs and result.get("annotation_geometry"):
            # Provenance sits beside the geometry rather than inside it, so the v1
            # staff schema a reader already parses stays exactly as it was.
            result["annotation_geometry_repairs"] = {
                "version": ANNOTATION_DIAGNOSTICS_VERSION,
                "staffs": [repair.to_dict() for repair in self.state.annotation_repairs],
            }
        write_annotation_capture(
            staffs=annotation_staffs,
            diagnostics=diagnostics,
            transform=self.coordinate_transform,
            source_image_size=result["source_image_size"],
            label="rejected" if primary is not None else "accepted",
        )


def write_visual_sidecar(path: str, document: dict[str, Any]) -> None:
    visual_sidecar_path = Path(path)
    visual_sidecar_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
