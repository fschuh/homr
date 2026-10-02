"""Opt-in capture of the pre-validation annotation grid and its runtime provenance.

A rejected sidecar keeps only a reason code: the grid that caused the rejection is
gone, so a failure seen on another machine cannot be replayed here. This module
writes that grid out verbatim, but only when the operator asks for it, because a
full grid is far too large for a UI message or a log line.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import homr
from homr.segmentation.config import model_name as segmentation_model_name
from homr.transformer.configs import model_name as transformer_model_name
from homr.visual_sidecar.annotation_geometry import (
    ANNOTATION_DIAGNOSTICS_VERSION,
    GeometryDiagnostic,
)
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform
from homr.visual_sidecar.models import PRODUCER_NAME, homr_version

CAPTURE_DIRECTORY_ENV = "HOMR_ANNOTATION_GEOMETRY_CAPTURE_DIR"
CAPTURE_VERSION = 1


def capture_directory() -> Path | None:
    value = os.environ.get(CAPTURE_DIRECTORY_ENV)
    return Path(value) if value else None


def _runtime_provenance() -> dict[str, Any]:
    """Where the producer was imported from, and what actually runs inference.

    An editable install reports whatever version its last build recorded, so the
    import location and the live provider list are the fields that identify the
    running code.
    """
    providers: list[str] = []
    try:
        import onnxruntime as ort  # noqa: PLC0415 - optional extra, absence is tolerated

        providers = list(ort.get_available_providers())
    except Exception:  # pragma: no cover - inference engine is an optional extra
        providers = []
    return {
        "producer": {
            "name": PRODUCER_NAME,
            "version": homr_version(),
            "import_location": str(Path(homr.__file__).resolve().parent),
            "models": {
                "transformer": transformer_model_name,
                "segmentation": segmentation_model_name,
            },
        },
        "inference": {"available_onnxruntime_providers": providers},
    }


def write_annotation_capture(
    *,
    staffs: list[dict[str, Any]],
    diagnostics: list[GeometryDiagnostic],
    transform: PredictionCoordinateTransform,
    source_image_size: list[int] | tuple[int, int],
    label: str | None = None,
) -> Path | None:
    """Write the pre-validation grid, or return ``None`` when capture is off."""
    directory = capture_directory()
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "capture_version": CAPTURE_VERSION,
        "diagnostics_version": ANNOTATION_DIAGNOSTICS_VERSION,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": label,
        "source_image_size": list(source_image_size),
        "coordinate_transform": {
            "source_image_size": list(transform.source_image_size),
            "autocrop_box": list(transform.autocrop_box),
            "cropped_size": list(transform.cropped_size),
            "resized_size": list(transform.resized_size),
            "resize_scale": list(transform.resize_scale),
            "prediction_size": list(transform.prediction_size),
        },
        **_runtime_provenance(),
        "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
        "pre_validation_staffs": staffs,
    }
    name = f"{label or 'annotation'}-{uuid.uuid4().hex[:12]}.annotation-capture.json"
    path = directory / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
