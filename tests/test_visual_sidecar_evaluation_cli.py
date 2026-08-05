import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from homr.main import ProcessingConfig, ProcessingResult
from homr.music_xml_generator import XmlGeneratorArguments
from homr.visual_sidecar.evaluation_cli import run
from tests.test_visual_sidecar_evaluation import (
    musicxml_document,
    valid_sidecar,
    xml_note,
)


class TestVisualSidecarEvaluationCli(unittest.TestCase):
    def test_cli_runs_inference_writes_report_and_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "score.png"
            image.write_bytes(b"test image boundary")
            report_path = root / "report.json"
            prepared_models: list[tuple[bool, bool, bool]] = []

            def prepare_models(
                segnet_use_gpu: bool,
                transformer_use_gpu: bool,
                coreml_encoder: bool,
            ) -> None:
                prepared_models.append((segnet_use_gpu, transformer_use_gpu, coreml_encoder))

            def infer(
                image_path: str,
                config: ProcessingConfig,
                _xml_args: XmlGeneratorArguments,
            ) -> ProcessingResult:
                self.assertEqual(Path(image_path), image)
                self.assertTrue(config.write_visual_sidecar)
                musicxml_path = root / "score.musicxml"
                sidecar_path = root / "score.homr.visual.json"
                musicxml_path.write_text(
                    musicxml_document(xml_note("homr-note-1", "C", 4)),
                    encoding="utf-8",
                )
                sidecar_path.write_text(json.dumps(valid_sidecar()), encoding="utf-8")
                return ProcessingResult(musicxml_path, sidecar_path)

            exit_code = run(
                [str(image), "--gpu", "no", "--report", str(report_path)],
                inference_runner=infer,
                model_preparer=prepare_models,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(prepared_models, [(False, False, False)])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["evaluation_mode"], "inference")
            self.assertEqual(report["input_image"], str(image.resolve()))

    def test_cli_evaluates_explicit_existing_artifacts_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            musicxml_path = root / "score.musicxml"
            sidecar_path = root / "custom-sidecar.json"
            report_path = root / "report.json"
            musicxml_path.write_text(
                musicxml_document(xml_note("homr-note-1", "C", 4)),
                encoding="utf-8",
            )
            sidecar_path.write_text(json.dumps(valid_sidecar()), encoding="utf-8")

            output = StringIO()
            with redirect_stdout(output):
                exit_code = run(
                    [
                        "--musicxml",
                        str(musicxml_path),
                        "--sidecar",
                        str(sidecar_path),
                        "--report",
                        str(report_path),
                    ],
                    inference_runner=lambda *_args: self.fail("inference must not run"),
                    model_preparer=lambda *_args: self.fail("models must not be prepared"),
                )

            self.assertEqual(exit_code, 0)
            self.assertNotIn("Inferring", output.getvalue())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["evaluation_mode"], "artifacts")
            self.assertIsNone(report["input_image"])
            self.assertEqual(report["musicxml_path"], str(musicxml_path.resolve()))
            self.assertEqual(report["visual_sidecar_path"], str(sidecar_path.resolve()))

    def test_cli_infers_and_announces_sidecar_from_musicxml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            musicxml_path = root / "score.musicxml"
            sidecar_path = root / "score.homr.visual.json"
            musicxml_path.write_text(
                musicxml_document(xml_note("homr-note-1", "C", 4)),
                encoding="utf-8",
            )
            sidecar_path.write_text(json.dumps(valid_sidecar()), encoding="utf-8")
            output = StringIO()

            with redirect_stdout(output):
                exit_code = run(
                    ["--musicxml", str(musicxml_path)],
                    inference_runner=lambda *_args: self.fail("inference must not run"),
                    model_preparer=lambda *_args: self.fail("models must not be prepared"),
                )

            self.assertEqual(exit_code, 0)
            self.assertIn(
                f"Inferring visual sidecar path from MusicXML: {sidecar_path.resolve()}",
                output.getvalue(),
            )

    def test_cli_infers_and_announces_musicxml_from_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            musicxml_path = root / "score.musicxml"
            sidecar_path = root / "score.homr.visual.json"
            musicxml_path.write_text(
                musicxml_document(xml_note("homr-note-1", "C", 4)),
                encoding="utf-8",
            )
            sidecar_path.write_text(json.dumps(valid_sidecar()), encoding="utf-8")
            output = StringIO()

            with redirect_stdout(output):
                exit_code = run(
                    ["--sidecar", str(sidecar_path)],
                    inference_runner=lambda *_args: self.fail("inference must not run"),
                    model_preparer=lambda *_args: self.fail("models must not be prepared"),
                )

            self.assertEqual(exit_code, 0)
            self.assertIn(
                f"Inferring MusicXML path from visual sidecar: {musicxml_path.resolve()}",
                output.getvalue(),
            )

    def test_cli_rejects_mixing_image_and_artifact_modes(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            run(["score.png", "--musicxml", "score.musicxml"])

        self.assertEqual(raised.exception.code, 2)

    def test_cli_reports_an_inferred_counterpart_that_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            musicxml_path = Path(directory) / "missing.musicxml"
            sidecar_path = Path(directory) / "missing.homr.visual.json"
            output = StringIO()

            with redirect_stdout(output):
                exit_code = run(["--musicxml", str(musicxml_path)])

            self.assertEqual(exit_code, 2)
            self.assertIn(str(sidecar_path.resolve()), output.getvalue())

    def test_cli_returns_one_for_artifact_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "score.png"
            image.write_bytes(b"test image boundary")

            def infer(
                _image_path: str,
                _config: ProcessingConfig,
                _xml_args: XmlGeneratorArguments,
            ) -> ProcessingResult:
                musicxml_path = root / "score.musicxml"
                sidecar_path = root / "score.homr.visual.json"
                musicxml_path.write_text(
                    musicxml_document(xml_note("homr-note-1", "C", 4)),
                    encoding="utf-8",
                )
                sidecar = valid_sidecar()
                notes = sidecar["notes"]
                notes[0]["pitch"] = "D4"
                sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
                return ProcessingResult(musicxml_path, sidecar_path)

            exit_code = run(
                [str(image), "--gpu", "no"],
                inference_runner=infer,
                model_preparer=lambda *_args: None,
            )

            self.assertEqual(exit_code, 1)

    def test_cli_returns_two_for_non_v3_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "score.png"
            image.write_bytes(b"test image boundary")

            def infer(
                _image_path: str,
                _config: ProcessingConfig,
                _xml_args: XmlGeneratorArguments,
            ) -> ProcessingResult:
                musicxml_path = root / "score.musicxml"
                sidecar_path = root / "score.homr.visual.json"
                musicxml_path.write_text(
                    musicxml_document(xml_note("homr-note-1", "C", 4)),
                    encoding="utf-8",
                )
                sidecar = valid_sidecar()
                sidecar["version"] = 2
                sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
                return ProcessingResult(musicxml_path, sidecar_path)

            exit_code = run(
                [str(image), "--gpu", "no"],
                inference_runner=infer,
                model_preparer=lambda *_args: None,
            )

            self.assertEqual(exit_code, 2)

    def test_cli_returns_two_when_inference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "score.png"
            image.write_bytes(b"test image boundary")

            def infer(
                _image_path: str,
                _config: ProcessingConfig,
                _xml_args: XmlGeneratorArguments,
            ) -> ProcessingResult:
                raise RuntimeError("inference failed")

            exit_code = run(
                [str(image), "--gpu", "no"],
                inference_runner=infer,
                model_preparer=lambda *_args: None,
            )

            self.assertEqual(exit_code, 2)
