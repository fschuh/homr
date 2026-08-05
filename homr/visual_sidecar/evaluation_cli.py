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
        description=(
            "Run HOMR on an image, or validate an existing MusicXML and visual sidecar v3 pair."
        ),
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Score image to process; omit when evaluating existing artifacts",
    )
    parser.add_argument(
        "--musicxml",
        type=Path,
        help="Existing MusicXML artifact; infers the sibling sidecar when used alone",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        help="Existing visual sidecar v3; infers the sibling MusicXML when used alone",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON evaluation report path")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable HOMR debug output in image inference mode",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use HOMR inference caches in image inference mode",
    )
    parser.add_argument(
        "--gpu",
        type=GpuSupport,
        choices=list(GpuSupport),
        default=GpuSupport.AUTO,
        metavar="{no,auto,force}",
        help="GPU mode for image inference: no, auto, or force",
    )
    parser.add_argument(
        "--coreml-encoder",
        action="store_true",
        help="Use the CoreML transformer encoder for image inference when available",
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
    report: VisualEvalReport,
    *,
    evaluation_mode: str,
    musicxml_path: Path,
    visual_sidecar_path: Path,
    input_image: Path | None,
) -> dict[str, Any]:
    document = report.to_dict()
    document["evaluation_mode"] = evaluation_mode
    document["input_image"] = str(input_image.resolve()) if input_image is not None else None
    document["musicxml_path"] = str(musicxml_path.resolve())
    document["visual_sidecar_path"] = str(visual_sidecar_path.resolve())
    return document


def _musicxml_path_from_sidecar(sidecar_path: Path) -> Path:
    suffix = ".homr.visual.json"
    if sidecar_path.name.endswith(suffix):
        return sidecar_path.with_name(sidecar_path.name[: -len(suffix)] + ".musicxml")
    return sidecar_path.with_suffix(".musicxml")


def _artifact_paths(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[Path, Path] | None:
    image: Path | None = args.image
    musicxml_path: Path | None = args.musicxml
    sidecar_path: Path | None = args.sidecar
    if image is not None:
        if musicxml_path is not None or sidecar_path is not None:
            parser.error("the image argument cannot be combined with --musicxml or --sidecar")
        return None
    if musicxml_path is None and sidecar_path is None:
        parser.error("provide an image, --musicxml, or --sidecar")
    if musicxml_path is None:
        if sidecar_path is None:
            parser.error("cannot infer MusicXML without a visual sidecar path")
        musicxml_path = _musicxml_path_from_sidecar(sidecar_path)
        sys.stdout.write(
            f"Inferring MusicXML path from visual sidecar: {musicxml_path.resolve()}\n"
        )
    if sidecar_path is None:
        sidecar_path = musicxml_path.with_suffix(".homr.visual.json")
        sys.stdout.write(f"Inferring visual sidecar path from MusicXML: {sidecar_path.resolve()}\n")
    return musicxml_path, sidecar_path


def _evaluate_artifacts(
    musicxml_path: Path, sidecar_path: Path
) -> tuple[VisualEvalReport, dict[str, Any]]:
    try:
        musicxml = musicxml_path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvaluationInputError(f"Cannot read MusicXML {musicxml_path}: {error}") from error
    report = evaluate_musicxml_sidecar(musicxml, _load_sidecar(sidecar_path))
    return report, _report_document(
        report,
        evaluation_mode="artifacts",
        musicxml_path=musicxml_path,
        visual_sidecar_path=sidecar_path,
        input_image=None,
    )


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
    artifact_paths = _artifact_paths(args, parser)

    try:
        if artifact_paths is not None:
            report, report_document = _evaluate_artifacts(*artifact_paths)
        else:
            image: Path = args.image
            if not image.is_file():
                eprint(f"{image} is not a valid image file")
                return 2
            config, model_options = _processing_config(args)
            model_preparer(*model_options)
            result = inference_runner(str(image), config, XmlGeneratorArguments())
            if result.visual_sidecar_path is None:
                raise EvaluationInputError("HOMR did not produce a visual sidecar")
            musicxml = result.musicxml_path.read_text(encoding="utf-8")
            sidecar = _load_sidecar(result.visual_sidecar_path)
            report = evaluate_musicxml_sidecar(musicxml, sidecar)
            report_document = _report_document(
                report,
                evaluation_mode="inference",
                musicxml_path=result.musicxml_path,
                visual_sidecar_path=result.visual_sidecar_path,
                input_image=image,
            )
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
