from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from homr.main import (
    GpuSupport,
    ProcessingConfig,
    ProcessingResult,
    download_weights,
    process_image,
)
from homr.music_xml_generator import XmlGeneratorArguments
from homr.onnx_providers import coreml_available, cuda_available
from homr.simple_logging import eprint
from homr.visual_sidecar.evaluation import (
    EvaluationInputError,
    VisualEvalReport,
    evaluate_musicxml_sidecar,
)

InferenceRunner = Callable[[str, ProcessingConfig, XmlGeneratorArguments], ProcessingResult]
ModelPreparer = Callable[[bool, bool, bool], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="homr-visual-eval",
        description="Run HOMR on one image and validate MusicXML against visual sidecar v3.",
    )
    parser.add_argument("image", type=Path, help="Score image to process")
    parser.add_argument("--report", type=Path, help="Optional JSON evaluation report path")
    parser.add_argument("--debug", action="store_true", help="Enable HOMR debug output")
    parser.add_argument("--cache", action="store_true", help="Use HOMR inference caches")
    parser.add_argument(
        "--gpu",
        type=GpuSupport,
        choices=list(GpuSupport),
        default=GpuSupport.AUTO,
        metavar="{no,auto,force}",
        help="GPU mode: no, auto, or force",
    )
    parser.add_argument(
        "--coreml-encoder",
        action="store_true",
        help="Use the CoreML transformer encoder when available",
    )
    return parser


def _processing_config(
    args: argparse.Namespace,
) -> tuple[ProcessingConfig, tuple[bool, bool, bool]]:
    force_gpu = args.gpu == GpuSupport.FORCE
    auto_gpu = args.gpu == GpuSupport.AUTO
    transformer_use_gpu = force_gpu or (auto_gpu and cuda_available())
    segnet_use_gpu = force_gpu or (auto_gpu and (cuda_available() or coreml_available()))
    coreml_encoder = args.coreml_encoder and not transformer_use_gpu and coreml_available()
    return (
        ProcessingConfig(
            enable_debug=args.debug,
            enable_cache=args.cache,
            write_staff_positions=False,
            read_staff_positions=False,
            selected_staff=-1,
            transformer_use_gpu=transformer_use_gpu,
            segnet_use_gpu=segnet_use_gpu,
            coreml_encoder=coreml_encoder,
            write_visual_sidecar=True,
        ),
        (segnet_use_gpu, transformer_use_gpu, coreml_encoder),
    )


def _load_sidecar(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationInputError(f"Cannot read visual sidecar {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationInputError("Visual sidecar root must be a JSON object")
    return value


def _report_document(
    report: VisualEvalReport, image: Path, result: ProcessingResult
) -> dict[str, Any]:
    document = report.to_dict()
    document["input_image"] = str(image.resolve())
    document["musicxml_path"] = str(result.musicxml_path)
    document["visual_sidecar_path"] = (
        str(result.visual_sidecar_path) if result.visual_sidecar_path is not None else None
    )
    return document


def _write_human_summary(report: VisualEvalReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    counts = report.counts
    sys.stdout.write(
        f"{status}: {counts['musicxml_notes']} MusicXML notes, "
        f"{counts['linked_visual_groups']} linked visual groups, "
        f"{counts['diagnostic_visual_groups']} diagnostic candidates, "
        f"{counts['divergences']} divergences\n"
    )
    for divergence in report.divergences:
        identity = divergence.musicxml_id or divergence.visual_group_id or "document"
        sys.stdout.write(f"- {divergence.kind} [{identity}]: {divergence.message}\n")


def run(
    argv: Sequence[str] | None = None,
    *,
    inference_runner: InferenceRunner = process_image,
    model_preparer: ModelPreparer = download_weights,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    image: Path = args.image
    if not image.is_file():
        eprint(f"{image} is not a valid image file")
        return 2

    try:
        config, model_options = _processing_config(args)
        model_preparer(*model_options)
        result = inference_runner(str(image), config, XmlGeneratorArguments())
        if result.visual_sidecar_path is None:
            raise EvaluationInputError("HOMR did not produce a visual sidecar")
        musicxml = result.musicxml_path.read_text(encoding="utf-8")
        sidecar = _load_sidecar(result.visual_sidecar_path)
        report = evaluate_musicxml_sidecar(musicxml, sidecar)
        report_document = _report_document(report, image, result)
        if args.report is not None:
            args.report.write_text(
                json.dumps(report_document, indent=2) + "\n",
                encoding="utf-8",
            )
        _write_human_summary(report)
        return 0 if report.passed else 1
    except Exception as error:
        eprint(f"Visual evaluation failed: {error}")
        return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
