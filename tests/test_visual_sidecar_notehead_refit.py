import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note, Staff, StaffPoint
from homr.note_detection import NoteheadWithStem, split_clumps_of_noteheads
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from homr.visual_sidecar.models import VisualMatch


class TestVisualSidecarNoteheadRefit(unittest.TestCase):
    @staticmethod
    def _transform(width: int = 140, height: int = 120) -> PredictionCoordinateTransform:
        return PredictionCoordinateTransform(
            source_image_size=(width, height),
            autocrop_box=(0, 0, width, height),
            cropped_size=(width, height),
            resized_size=(width, height),
            resize_scale=(1.0, 1.0),
            prediction_size=(width, height),
        )

    @staticmethod
    def _staff(width: int = 140) -> Staff:
        return Staff(
            [
                StaffPoint(0, [40, 50, 60, 70, 80], 0),
                StaffPoint(width, [40, 50, 60, 70, 80], 0),
            ]
        )

    @staticmethod
    def _note(
        x: int,
        y: int,
        position: int,
        visual_id: str,
        *,
        split_clump_id: str | None = None,
        split_clump_bounds: tuple[int, int, int, int] | None = None,
    ) -> Note:
        contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
        return Note(
            BoundingEllipse(((x, y), (14, 10), -20), contour),
            position,
            None,
            None,
            visual_id,
            split_clump_id=split_clump_id,
            split_clump_bounds=split_clump_bounds,
        )

    @staticmethod
    def _filled_head(image: np.ndarray, center: tuple[int, int]) -> None:
        cv2.ellipse(image, center, (7, 5), -20, 0, 360, 0, -1)

    def test_split_clump_members_are_jointly_refitted_to_distinct_ink(self) -> None:
        image = np.full((120, 140), 255, dtype=np.uint8)
        mask = np.zeros_like(image)
        for center in ((60, 60), (60, 70)):
            self._filled_head(image, center)
            cv2.ellipse(mask, center, (7, 5), -20, 0, 360, 255, -1)
        clump_id = "split-clump-test"
        clump_bounds = (52, 54, 68, 76)
        upper = self._note(
            60,
            64,
            4,
            "upper",
            split_clump_id=clump_id,
            split_clump_bounds=clump_bounds,
        )
        lower = self._note(
            60,
            74,
            2,
            "lower",
            split_clump_id=clump_id,
            split_clump_bounds=clump_bounds,
        )
        builder = VisualSidecarBuilder(
            self._transform(),
            notehead_mask=mask,
            source_image=image,
        )
        builder.add_staff_visual_notes(0, [upper, lower], [upper.copy(), lower.copy()])
        builder.add_staff_matches(
            [
                EncodedSymbol("clef_G2", position="upper"),
                EncodedSymbol("note_4", "B4", position="upper", coordinates=(60, 64)),
                EncodedSymbol("chord"),
                EncodedSymbol("note_4", "G4", position="upper", coordinates=(60, 74)),
            ],
            0,
            source_staff=self._staff(),
        )

        repaired_upper = builder.visual_groups["upper"]
        repaired_lower = builder.visual_groups["lower"]
        self.assertAlmostEqual(repaired_upper.prediction_center[1], 60, delta=2)
        self.assertAlmostEqual(repaired_lower.prediction_center[1], 70, delta=2)
        self.assertEqual((repaired_upper.staff_position, repaired_lower.staff_position), (5, 3))
        self.assertTrue(repaired_upper.notehead_core_pixels)
        self.assertTrue(repaired_lower.notehead_core_pixels)
        self.assertTrue(
            repaired_upper.notehead_core_pixels.isdisjoint(repaired_lower.notehead_core_pixels)
        )
        self.assertIn("joint_notehead_refit", repaired_upper.repair_actions)
        self.assertIn("joint_notehead_refit", repaired_lower.repair_actions)

    def test_refitted_staggered_members_retain_physical_chord_identity(self) -> None:
        image = np.full((120, 140), 255, dtype=np.uint8)
        mask = np.zeros_like(image)
        for center in ((69, 65), (50, 70), (50, 80)):
            self._filled_head(image, center)
            cv2.ellipse(mask, center, (7, 5), -20, 0, 360, 255, -1)

        stem_contour = np.asarray(
            [[[58, 58]], [[61, 58]], [[61, 92]], [[58, 92]]], dtype=np.float32
        )
        shared_stem = RotatedBoundingBox(
            cv2.minAreaRect(stem_contour),
            stem_contour,
        )
        clump_id = "staggered-chord-clump"
        clump_bounds = (42, 63, 62, 88)

        def note(
            x: int,
            y: int,
            position: int,
            visual_id: str,
            *,
            split: bool = False,
        ) -> Note:
            result = self._note(
                x,
                y,
                position,
                visual_id,
                split_clump_id=clump_id if split else None,
                split_clump_bounds=clump_bounds if split else None,
            )
            result.stem = shared_stem
            return result

        notes = [
            note(69, 65, 4, "top"),
            note(55, 75, 2, "middle", split=True),
            note(55, 85, 0, "bottom", split=True),
        ]
        builder = VisualSidecarBuilder(
            self._transform(),
            notehead_mask=mask,
            stem_fragments=[shared_stem],
            source_image=image,
        )
        builder.add_staff_visual_notes(0, notes, [candidate.copy() for candidate in notes])
        builder.add_staff_matches(
            [
                EncodedSymbol("clef_G2", position="upper"),
                EncodedSymbol("note_16", "A4", position="upper", coordinates=(69, 65)),
                EncodedSymbol("chord"),
                EncodedSymbol("note_16", "G4", position="upper", coordinates=(55, 75)),
                EncodedSymbol("chord"),
                EncodedSymbol("note_16", "E4", position="upper", coordinates=(55, 85)),
            ],
            0,
            source_staff=self._staff(),
        )

        groups = [builder.visual_groups[visual_id] for visual_id in ("top", "middle", "bottom")]
        self.assertIn("joint_notehead_refit", groups[1].repair_actions)
        self.assertIn("joint_notehead_refit", groups[2].repair_actions)
        self.assertTrue(
            set(groups[0].owned_stem_component_ids).intersection(
                groups[1].owned_stem_component_ids,
                groups[2].owned_stem_component_ids,
            )
        )
        self.assertEqual(len({group.moment_id for group in groups}), 1)
        self.assertEqual(len({group.chord_id for group in groups}), 1)
        self.assertIsNotNone(groups[0].chord_id)

    def test_two_members_cannot_claim_one_core_ink_region(self) -> None:
        image = np.full((120, 140), 255, dtype=np.uint8)
        self._filled_head(image, (60, 70))
        clump_id = "one-core"
        clump_bounds = (52, 64, 68, 78)
        first = self._note(
            57,
            75,
            2,
            "first",
            split_clump_id=clump_id,
            split_clump_bounds=clump_bounds,
        )
        second = self._note(
            63,
            75,
            2,
            "second",
            split_clump_id=clump_id,
            split_clump_bounds=clump_bounds,
        )
        builder = VisualSidecarBuilder(self._transform(), source_image=image)
        builder.add_staff_visual_notes(0, [first, second], [first.copy(), second.copy()])
        first_symbol = EncodedSymbol("note_4", "G4", position="upper", coordinates=(57, 75))
        second_symbol = EncodedSymbol("note_4", "G4", position="upper", coordinates=(63, 75))
        symbols = [first_symbol, EncodedSymbol("chord"), second_symbol]
        builder.matches_by_symbol_id[first_symbol.visual_match_id] = VisualMatch(
            first_symbol, "first", 1.0, "structural"
        )
        builder.matches_by_symbol_id[second_symbol.visual_match_id] = VisualMatch(
            second_symbol, "second", 1.0, "structural"
        )
        builder.notehead_refitter.refit(
            symbols,
            0,
            source_staff=self._staff(),
            expected_staff_positions={
                first_symbol.visual_match_id: 3,
                second_symbol.visual_match_id: 3,
            },
            physical_staff_lines=builder.recovery.physical_staff_lines_at_x,
        )

        self.assertEqual(builder.visual_groups["first"].prediction_center, (57, 75))
        self.assertEqual(builder.visual_groups["second"].prediction_center, (63, 75))
        self.assertNotIn("joint_notehead_refit", builder.visual_groups["first"].repair_actions)
        self.assertNotIn("joint_notehead_refit", builder.visual_groups["second"].repair_actions)

    def test_neighboring_moment_ink_is_not_available_to_a_repair(self) -> None:
        image = np.full((120, 140), 255, dtype=np.uint8)
        self._filled_head(image, (60, 70))
        owner = self._note(60, 70, 3, "owner")
        target = self._note(64, 75, 2, "target", split_clump_id="target-clump")
        builder = VisualSidecarBuilder(self._transform(), source_image=image)
        builder.add_staff_visual_notes(0, [owner, target], [owner.copy(), target.copy()])
        builder.add_staff_matches(
            [
                EncodedSymbol("clef_G2", position="upper"),
                EncodedSymbol("note_4", "G4", position="upper", coordinates=(60, 70)),
                EncodedSymbol("note_4", "G4", position="upper", coordinates=(64, 75)),
            ],
            0,
            source_staff=self._staff(),
        )

        unchanged = builder.visual_groups["target"]
        self.assertEqual(unchanged.prediction_center, (64, 75))
        self.assertNotIn("joint_notehead_refit", unchanged.repair_actions)

    def test_staff_stem_and_beam_ink_cannot_prove_a_notehead(self) -> None:
        image = np.full((120, 140), 255, dtype=np.uint8)
        cv2.line(image, (20, 70), (120, 70), 0, 2)
        cv2.line(image, (60, 45), (60, 92), 0, 2)
        cv2.rectangle(image, (45, 67), (90, 72), 0, -1)
        target = self._note(60, 75, 2, "target", split_clump_id="straight-ink")
        builder = VisualSidecarBuilder(self._transform(), source_image=image)
        builder.add_staff_visual_notes(0, [target], [target.copy()])
        builder.add_staff_matches(
            [
                EncodedSymbol("clef_G2", position="upper"),
                EncodedSymbol("note_4", "G4", position="upper", coordinates=(60, 75)),
            ],
            0,
            source_staff=self._staff(),
        )

        unchanged = builder.visual_groups["target"]
        self.assertEqual(unchanged.staff_position, 2)
        self.assertNotIn("joint_notehead_refit", unchanged.repair_actions)

    def test_equal_quality_pixel_assignments_are_rejected(self) -> None:
        image = np.full((120, 140), 255, dtype=np.uint8)
        self._filled_head(image, (55, 70))
        self._filled_head(image, (65, 70))
        target = self._note(60, 75, 2, "target", split_clump_id="ambiguous")
        target.split_clump_bounds = (47, 64, 73, 77)
        builder = VisualSidecarBuilder(self._transform(), source_image=image)
        builder.add_staff_visual_notes(0, [target], [target.copy()])
        builder.add_staff_matches(
            [
                EncodedSymbol("clef_G2", position="upper"),
                EncodedSymbol("note_4", "G4", position="upper", coordinates=(60, 75)),
            ],
            0,
            source_staff=self._staff(),
        )

        unchanged = builder.visual_groups["target"]
        self.assertEqual(unchanged.prediction_center, (60, 75))
        self.assertNotIn("joint_notehead_refit", unchanged.repair_actions)

    def test_strong_nonclump_geometry_and_clean_geometry_are_unchanged(self) -> None:
        image = np.full((120, 140), 255, dtype=np.uint8)
        self._filled_head(image, (45, 75))
        self._filled_head(image, (95, 70))
        mismatched = self._note(45, 75, 2, "mismatched")
        clean = self._note(95, 70, 3, "clean")
        builder = VisualSidecarBuilder(self._transform(), source_image=image)
        builder.add_staff_visual_notes(
            0,
            [mismatched, clean],
            [mismatched.copy(), clean.copy()],
        )
        builder.add_staff_matches(
            [
                EncodedSymbol("clef_G2", position="upper"),
                EncodedSymbol("note_4", "G4", position="upper", coordinates=(45, 75)),
                EncodedSymbol("note_4", "G4", position="upper", coordinates=(95, 70)),
            ],
            0,
            source_staff=self._staff(),
        )

        self.assertEqual(builder.visual_groups["mismatched"].prediction_center, (45, 75))
        self.assertEqual(builder.visual_groups["clean"].prediction_center, (95, 70))
        self.assertNotIn("joint_notehead_refit", builder.visual_groups["mismatched"].repair_actions)
        self.assertNotIn("joint_notehead_refit", builder.visual_groups["clean"].repair_actions)

    def test_pitch_guided_location_without_ink_remains_unmatched(self) -> None:
        image = np.full((120, 140), 255, dtype=np.uint8)
        builder = VisualSidecarBuilder(self._transform(), source_image=image)
        missing = EncodedSymbol("note_4", "E5", position="upper", coordinates=(60, 35))

        builder.add_staff_matches(
            [EncodedSymbol("clef_G2", position="upper"), missing],
            0,
            source_staff=self._staff(),
        )

        self.assertIsNone(builder.matches_by_symbol_id[missing.visual_match_id].visual_id)
        self.assertEqual(builder.visual_groups, {})

    def test_split_detection_preserves_shared_clump_provenance(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.rectangle(mask, (30, 20), (49, 49), 255, -1)
        contour = np.array(
            [[[30, 20]], [[49, 20]], [[49, 49]], [[30, 49]], [[30, 20]]],
            dtype=np.float32,
        )
        candidate = NoteheadWithStem(
            BoundingEllipse(((39.5, 34.5), (19, 29), 0), contour, debug_id=7),
            None,
        )

        members = split_clumps_of_noteheads(candidate, mask, self._staff(80))

        self.assertGreater(len(members), 1)
        self.assertEqual(len({member.split_clump_id for member in members}), 1)
        self.assertIsNotNone(members[0].split_clump_id)
        self.assertTrue(all(member.split_clump_bounds == (30, 20, 49, 49) for member in members))


if __name__ == "__main__":
    unittest.main()
