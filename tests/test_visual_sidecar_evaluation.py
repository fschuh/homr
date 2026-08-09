import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from homr.bounding_boxes import BoundingEllipse
from homr.model import Note
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from homr.visual_sidecar.evaluation import (
    UnsupportedSidecarVersionError,
    VisualEvalReport,
    evaluate_musicxml_sidecar,
)


def musicxml_document(notes: str, clef: str | None = None) -> str:
    clef_xml = clef or "<clef number='1'><sign>G</sign><line>2</line></clef>"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions>{clef_xml}</attributes>
      {notes}
    </measure>
  </part>
</score-partwise>
"""


def xml_note(
    musicxml_id: str,
    step: str,
    octave: int,
    *,
    alter: int | None = None,
    chord: bool = False,
    voice: int = 1,
    staff: int = 1,
) -> str:
    alter_xml = f"<alter>{alter}</alter>" if alter is not None else ""
    chord_xml = "<chord/>" if chord else ""
    return f"""
<note id="{musicxml_id}">
  {chord_xml}<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>
  <duration>1</duration><voice>{voice}</voice><staff>{staff}</staff>
</note>
"""


def sidecar_note(
    musicxml_id: str,
    pitch: str,
    visual_group_id: str | None,
    *,
    voice: int = 1,
    staff: int = 1,
) -> dict[str, Any]:
    return {
        "musicxml_id": musicxml_id,
        "part": 1,
        "measure": 1,
        "musicxml_staff_number": staff,
        "voice": voice,
        "pitch": pitch,
        "duration": "note_4",
        "match_confidence": 1.0,
        "visual_group_id": visual_group_id,
        "alignment_method": "structural",
    }


def visual_group(
    visual_group_id: str,
    musicxml_id: str | None,
    staff_position: int,
    *,
    moment_id: str | None = "moment-1-1",
    chord_id: str | None = None,
    status: str = "canonical",
    staff: int = 0,
) -> dict[str, Any]:
    return {
        "visual_group_id": visual_group_id,
        "staff_group_index": 0,
        "staff_index": staff,
        "staff_position": staff_position,
        "center": [10.0, 10.0],
        "bbox": [5.0, 5.0, 15.0, 15.0],
        "notehead_ellipses": [{"cx": 10.0, "cy": 10.0, "rx": 5.0, "ry": 4.0}],
        "notehead_contours": [],
        "detected_notehead_contours": [],
        "refined_notehead_contours": [],
        "detected_stem_contours": [],
        "stem_contours": [],
        "stem_component_ids": [],
        "is_hollow_notehead": False,
        "musicxml_id": musicxml_id,
        "visual_status": status,
        "provenance": "segmentation",
        "moment_id": moment_id,
        "chord_id": chord_id,
        "repair_actions": [],
    }


def valid_sidecar(
    *,
    pitch: str = "C4",
    staff_position: int = -1,
) -> dict[str, Any]:
    return {
        "version": 3,
        "source_image_size": [100, 100],
        "preprocessing": {},
        "notes": [sidecar_note("homr-note-1", pitch, "vnote-1")],
        "raw_stem_contours": [],
        "visual_groups": [visual_group("vnote-1", "homr-note-1", staff_position)],
    }


def divergence_kinds(report: VisualEvalReport) -> set[str]:
    return {divergence.kind for divergence in report.divergences}


class TestVisualSidecarEvaluation(unittest.TestCase):
    def test_generated_homr_musicxml_and_v3_sidecar_pass_together(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        note = Note(
            BoundingEllipse(((30, 50), (10, 8), 0), np.array([[25, 46], [35, 54]]), 1),
            position=1,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(0, [note], [note.copy()])
        symbols = [
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol("note_4", "E4", position="upper", coordinates=(30, 50)),
        ]
        builder.add_staff_matches(symbols, 0)
        generated = generate_xml(XmlGeneratorArguments(), [[*symbols]], "", visual_sidecar=builder)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score.musicxml"
            generated.write(str(path))
            musicxml = path.read_text(encoding="utf-8")

        report = evaluate_musicxml_sidecar(musicxml, builder.to_json_dict())

        self.assertTrue(report.passed, report.to_dict())

    def test_valid_v3_sidecar_passes(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "C", 4))

        report = evaluate_musicxml_sidecar(xml, valid_sidecar())

        self.assertTrue(report.passed)
        self.assertEqual(report.counts["musicxml_notes"], 1)
        self.assertEqual(report.to_dict()["report_version"], 1)

    def test_v2_is_rejected_without_compatibility(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "C", 4))
        sidecar = valid_sidecar()
        sidecar["version"] = 2

        with self.assertRaises(UnsupportedSidecarVersionError):
            evaluate_musicxml_sidecar(xml, sidecar)

    def test_diagnostic_visual_candidate_is_reported_without_failing(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "C", 4))
        sidecar = valid_sidecar()
        groups = sidecar["visual_groups"]
        groups.append(
            visual_group(
                "diagnostic-1",
                None,
                4,
                moment_id=None,
                status="diagnostic",
            )
        )

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertTrue(report.passed)
        self.assertEqual(report.diagnostic_visual_group_ids, ("diagnostic-1",))

    def test_missing_and_extra_sidecar_notes_are_distinct(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "C", 4))
        sidecar = valid_sidecar()
        notes = sidecar["notes"]
        notes.clear()
        notes.append(sidecar_note("homr-note-extra", "D4", None))
        groups = sidecar["visual_groups"]
        groups.clear()

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertIn("missing_sidecar_note", divergence_kinds(report))
        self.assertIn("extra_sidecar_note", divergence_kinds(report))
        self.assertNotIn("added_visual_note", divergence_kinds(report))

    def test_existing_sidecar_note_without_visual_group_fails(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "C", 4))
        sidecar = valid_sidecar()
        notes = sidecar["notes"]
        groups = sidecar["visual_groups"]
        notes[0]["visual_group_id"] = None
        groups.clear()

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertIn("missing_visual_note", divergence_kinds(report))

    def test_artifact_pitch_and_visual_position_are_independent(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "C", 4))
        sidecar = valid_sidecar(pitch="D4", staff_position=1)

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertIn("pitch_divergence", divergence_kinds(report))
        self.assertIn("visual_pitch_divergence", divergence_kinds(report))

    def test_accidental_is_checked_between_artifacts_but_not_by_position(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "F", 4, alter=1))
        sidecar = valid_sidecar(pitch="F#4", staff_position=2)

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertTrue(report.passed)
        self.assertEqual(report.capabilities["visual_accidental_check"], "not_evaluated")

    def test_marked_cross_staff_repair_uses_the_physical_staff_clef(self) -> None:
        clefs = (
            "<clef number='1'><sign>G</sign><line>2</line></clef>"
            "<clef number='2'><sign>F</sign><line>4</line></clef>"
        )
        xml = musicxml_document(xml_note("homr-note-1", "F", 3, alter=1, staff=2), clefs)
        note = sidecar_note("homr-note-1", "F#3", "vnote-1", staff=2)
        note["alignment_method"] = "cross_staff_repair"
        group = visual_group("vnote-1", "homr-note-1", -5, status="fallback", staff=0)
        group["repair_actions"].append("cross_staff_link_repaired")
        sidecar = {"version": 3, "notes": [note], "visual_groups": [group]}

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(report.capabilities["visual_accidental_check"], "not_evaluated")

    def test_cross_staff_mismatch_without_complete_repair_metadata_fails(self) -> None:
        clefs = (
            "<clef number='1'><sign>G</sign><line>2</line></clef>"
            "<clef number='2'><sign>F</sign><line>4</line></clef>"
        )
        xml = musicxml_document(xml_note("homr-note-1", "F", 3, staff=2), clefs)
        note = sidecar_note("homr-note-1", "F3", "vnote-1", staff=2)
        note["alignment_method"] = "cross_staff_repair"
        group = visual_group("vnote-1", "homr-note-1", -5, status="fallback", staff=0)
        sidecar = {"version": 3, "notes": [note], "visual_groups": [group]}

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertIn("contract_error", divergence_kinds(report))

    def test_cross_staff_repair_still_checks_diatonic_visual_position(self) -> None:
        clefs = (
            "<clef number='1'><sign>G</sign><line>2</line></clef>"
            "<clef number='2'><sign>F</sign><line>4</line></clef>"
        )
        xml = musicxml_document(xml_note("homr-note-1", "F", 3, staff=2), clefs)
        note = sidecar_note("homr-note-1", "F3", "vnote-1", staff=2)
        note["alignment_method"] = "cross_staff_repair"
        group = visual_group("vnote-1", "homr-note-1", -4, status="fallback", staff=0)
        group["repair_actions"].append("cross_staff_link_repaired")
        sidecar = {"version": 3, "notes": [note], "visual_groups": [group]}

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertEqual(divergence_kinds(report), {"visual_pitch_divergence"})

    def test_supported_clefs_and_octave_changes_derive_local_positions(self) -> None:
        cases = [
            ("<clef number='1'><sign>G</sign><line>2</line></clef>", "C", 4, -1),
            ("<clef number='1'><sign>F</sign><line>4</line></clef>", "C", 4, 11),
            ("<clef number='1'><sign>C</sign><line>3</line></clef>", "C", 4, 5),
            (
                "<clef number='1'><sign>G</sign><line>2</line>"
                "<clef-octave-change>1</clef-octave-change></clef>",
                "G",
                5,
                3,
            ),
        ]
        for clef, step, octave, staff_position in cases:
            with self.subTest(clef=clef):
                xml = musicxml_document(xml_note("homr-note-1", step, octave), clef)
                sidecar = valid_sidecar(pitch=f"{step}{octave}", staff_position=staff_position)

                report = evaluate_musicxml_sidecar(xml, sidecar)

                self.assertTrue(report.passed, report.to_dict())

    def test_clef_changes_are_applied_in_document_order(self) -> None:
        xml = """<score-partwise version="4.0">
<part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
<part id="P1"><measure number="1">
  <attributes><clef number="1"><sign>G</sign><line>2</line></clef></attributes>
  <note id="homr-note-1"><pitch><step>E</step><octave>4</octave></pitch>
    <duration>1</duration><voice>1</voice><staff>1</staff></note>
  <attributes><clef number="1"><sign>F</sign><line>4</line></clef></attributes>
  <note id="homr-note-2"><pitch><step>F</step><octave>3</octave></pitch>
    <duration>1</duration><voice>1</voice><staff>1</staff></note>
</measure></part></score-partwise>"""
        sidecar = {
            "version": 3,
            "notes": [
                sidecar_note("homr-note-1", "E4", "vnote-1"),
                sidecar_note("homr-note-2", "F3", "vnote-2"),
            ],
            "visual_groups": [
                visual_group("vnote-1", "homr-note-1", 1),
                visual_group("vnote-2", "homr-note-2", 7, moment_id="moment-1-2"),
            ],
        }

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertTrue(report.passed, report.to_dict())

    def test_musicxml_staff_number_selects_the_physical_staff_and_clef(self) -> None:
        clefs = (
            "<clef number='1'><sign>G</sign><line>2</line></clef>"
            "<clef number='2'><sign>F</sign><line>4</line></clef>"
        )
        xml = musicxml_document(xml_note("homr-note-1", "C", 4, voice=5, staff=2), clefs)
        sidecar = {
            "version": 3,
            "notes": [sidecar_note("homr-note-1", "C4", "vnote-1", voice=5, staff=2)],
            "visual_groups": [visual_group("vnote-1", "homr-note-1", 11, staff=1)],
        }

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertTrue(report.passed, report.to_dict())

    def test_unsupported_clef_fails_required_visual_pitch_check(self) -> None:
        xml = musicxml_document(
            xml_note("homr-note-1", "C", 4),
            "<clef number='1'><sign>percussion</sign><line>3</line></clef>",
        )

        report = evaluate_musicxml_sidecar(xml, valid_sidecar())

        self.assertIn("unevaluable_visual_pitch", divergence_kinds(report))

    def test_inverse_links_are_strictly_one_to_one(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "C", 4))
        sidecar = valid_sidecar()
        groups = sidecar["visual_groups"]
        duplicate = copy.deepcopy(groups[0])
        duplicate["visual_group_id"] = "vnote-duplicate"
        groups.append(duplicate)

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertIn("contract_error", divergence_kinds(report))

    def test_many_musicxml_notes_cannot_share_one_visual_group(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "E", 4) + xml_note("homr-note-2", "F", 4))
        sidecar = {
            "version": 3,
            "notes": [
                sidecar_note("homr-note-1", "E4", "vnote-1"),
                sidecar_note("homr-note-2", "F4", "vnote-1"),
            ],
            "visual_groups": [visual_group("vnote-1", "homr-note-1", 1)],
        }

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertIn("contract_error", divergence_kinds(report))
        self.assertFalse(report.passed)

    def test_linked_group_requires_moment_and_notehead_geometry(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "C", 4))
        sidecar = valid_sidecar()
        group = sidecar["visual_groups"][0]
        group["moment_id"] = None
        group["bbox"] = []
        group["notehead_ellipses"] = []

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertIn("contract_error", divergence_kinds(report))
        self.assertIn("missing_visual_note", divergence_kinds(report))

    def test_diagnostic_group_cannot_link_to_musicxml(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "C", 4))
        sidecar = valid_sidecar()
        groups = sidecar["visual_groups"]
        groups.append(
            visual_group(
                "diagnostic-1",
                "homr-note-1",
                4,
                moment_id=None,
                status="diagnostic",
            )
        )

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertIn("contract_error", divergence_kinds(report))

    def test_same_staff_musicxml_chord_can_share_a_physical_chord_id(self) -> None:
        xml = musicxml_document(
            xml_note("homr-note-1", "E", 4) + xml_note("homr-note-2", "G", 4, chord=True)
        )
        sidecar = {
            "version": 3,
            "notes": [
                sidecar_note("homr-note-1", "E4", "vnote-1"),
                sidecar_note("homr-note-2", "G4", "vnote-2"),
            ],
            "visual_groups": [
                visual_group("vnote-1", "homr-note-1", 1, chord_id="chord-1"),
                visual_group("vnote-2", "homr-note-2", 3, chord_id="chord-1"),
            ],
        }

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertTrue(report.passed, report.to_dict())

    def test_opposed_stem_musicxml_chord_needs_only_a_shared_moment(self) -> None:
        xml = musicxml_document(
            xml_note("homr-note-21", "E", 4) + xml_note("homr-note-22", "G", 4, chord=True)
        )
        sidecar = {
            "version": 3,
            "notes": [
                sidecar_note("homr-note-21", "E4", "vnote-21"),
                sidecar_note("homr-note-22", "G4", "vnote-22"),
            ],
            "visual_groups": [
                visual_group("vnote-21", "homr-note-21", 1),
                visual_group("vnote-22", "homr-note-22", 3),
            ],
        }
        for group in sidecar["visual_groups"]:
            group["repair_actions"] = ["opposed_stems_separated"]

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertTrue(report.passed, report.to_dict())

    def test_physical_chord_id_can_span_same_moment_musicxml_voices(self) -> None:
        xml = musicxml_document(
            xml_note("homr-note-1", "F", 4, voice=1)
            + "<backup><duration>1</duration></backup>"
            + xml_note("homr-note-2", "C", 4, voice=2)
            + xml_note("homr-note-3", "A", 3, chord=True, voice=2)
        )
        sidecar = {
            "version": 3,
            "notes": [
                sidecar_note("homr-note-1", "F4", "vnote-1", voice=1),
                sidecar_note("homr-note-2", "C4", "vnote-2", voice=2),
                sidecar_note("homr-note-3", "A3", "vnote-3", voice=2),
            ],
            "visual_groups": [
                visual_group("vnote-1", "homr-note-1", 2, chord_id="chord-1"),
                visual_group("vnote-2", "homr-note-2", -1, chord_id="chord-1"),
                visual_group("vnote-3", "homr-note-3", -3, chord_id="chord-1"),
            ],
        }
        sidecar["notes"][0]["duration"] = "note_8"
        sidecar["notes"][1]["duration"] = "note_1"
        sidecar["notes"][2]["duration"] = "note_1"

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertTrue(report.passed, report.to_dict())

    def test_broken_same_staff_chord_identifies_moment_assignments(self) -> None:
        xml = musicxml_document(
            xml_note("homr-note-31", "E", 4) + xml_note("homr-note-32", "G", 4, chord=True)
        )
        sidecar = {
            "version": 3,
            "notes": [
                sidecar_note("homr-note-31", "E4", "vnote-31"),
                sidecar_note("homr-note-32", "G4", "vnote-32"),
            ],
            "visual_groups": [
                visual_group(
                    "vnote-31",
                    "homr-note-31",
                    1,
                    moment_id="moment-left",
                    chord_id="chord-31",
                ),
                visual_group(
                    "vnote-32",
                    "homr-note-32",
                    3,
                    moment_id="moment-right",
                    chord_id="chord-31",
                ),
            ],
        }

        report = evaluate_musicxml_sidecar(xml, sidecar)
        error = next(
            divergence
            for divergence in report.divergences
            if "inconsistent moment_id assignments" in divergence.message
        )

        self.assertEqual(error.musicxml_id, "homr-note-31")
        self.assertIn("homr-note-31/vnote-31=moment-left", error.message)
        self.assertIn("homr-note-32/vnote-32=moment-right", error.message)

    def test_consecutive_beamed_notes_cannot_share_a_visual_moment(self) -> None:
        xml = musicxml_document(xml_note("homr-note-1", "E", 4) + xml_note("homr-note-2", "F", 4))
        sidecar = {
            "version": 3,
            "notes": [
                sidecar_note("homr-note-1", "E4", "vnote-1"),
                sidecar_note("homr-note-2", "F4", "vnote-2"),
            ],
            "visual_groups": [
                visual_group("vnote-1", "homr-note-1", 1),
                visual_group("vnote-2", "homr-note-2", 2),
            ],
        }

        report = evaluate_musicxml_sidecar(xml, sidecar)

        self.assertIn("contract_error", divergence_kinds(report))
        self.assertTrue(any("Sequential non-chord" in item.message for item in report.divergences))

        groups = sidecar["visual_groups"]
        groups[1]["moment_id"] = "moment-1-2"
        repaired_report = evaluate_musicxml_sidecar(xml, sidecar)
        self.assertTrue(repaired_report.passed, repaired_report.to_dict())
