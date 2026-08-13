from __future__ import annotations

import numpy as np
import pytest

from homr.main import ProcessingConfig, get_predictions

# Pytest assertions are intentional in this focused configuration test module.
# ruff: noqa: S101


def make_config(**overrides: object) -> ProcessingConfig:
    values: dict[str, object] = {
        "enable_debug": False,
        "enable_cache": False,
        "write_staff_positions": False,
        "read_staff_positions": False,
        "selected_staff": -1,
        "transformer_use_gpu": False,
        "segnet_use_gpu": False,
        "coreml_encoder": False,
        "write_visual_sidecar": True,
    }
    values.update(overrides)
    return ProcessingConfig(**values)  # type: ignore[arg-type]


def test_processing_config_defaults_to_non_overlapping_segnet_tiles() -> None:
    assert make_config().segmentation_step_size == 320


@pytest.mark.parametrize("step_size", [320, 240, 160])
def test_processing_config_accepts_requested_step_sizes(step_size: int) -> None:
    assert make_config(segmentation_step_size=step_size).segmentation_step_size == step_size


@pytest.mark.parametrize("step_size", [0, -1, 321, 640])
def test_processing_config_rejects_step_sizes_outside_window(step_size: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 320"):
        make_config(segmentation_step_size=step_size)


def test_step_size_propagates_into_segnet_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_extract(
        _preprocessed: np.ndarray,
        _image_path: str,
        **kwargs: object,
    ) -> object:
        captured.update(kwargs)
        masks = np.zeros((4, 4), dtype=np.uint8)
        return type(
            "Result",
            (),
            {
                "staff": masks,
                "symbols": masks,
                "stems_rests": masks,
                "notehead": masks,
                "clefs_keys": masks,
            },
        )()

    monkeypatch.setattr("homr.main.extract", fake_extract)
    image = np.zeros((4, 4), dtype=np.uint8)

    get_predictions(
        image,
        image,
        "page.png",
        enable_cache=False,
        segnet_use_gpu=False,
        segmentation_step_size=240,
    )

    assert captured["step_size"] == 240
