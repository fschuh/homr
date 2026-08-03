import cv2
import numpy as np

from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform
from homr.visual_sidecar.matching_utils import source_notehead_bounds
from homr.visual_sidecar.models import SidecarState, VisualGroup

DUPLICATE_NOTEHEAD_AREA_RATIO = 0.6
DUPLICATE_NOTEHEAD_MAX_HORIZONTAL_DISTANCE = 1.5
DUPLICATE_NOTEHEAD_MAX_VERTICAL_DISTANCE = 0.3
MAX_VISUAL_GROUP_DISTANCE_FROM_CLEF = 16.0


class CandidateCleaner:
    def __init__(
        self, state: SidecarState, coordinate_transform: PredictionCoordinateTransform
    ) -> None:
        self.state = state
        self.coordinate_transform = coordinate_transform
        self.visual_groups = state.visual_groups
        self.unmatched_visual_notes = state.unmatched_visual_notes
        self._duplicate_staff_positions_by_visual_id = state.duplicate_staff_positions_by_visual_id

    def repair(self, symbols: list[EncodedSymbol], staff_index: int) -> None:
        self._repair_candidate_geometry(symbols, staff_index)

    def merge_split_whole_note_fragments(self, staff_index: int) -> None:
        self._merge_split_whole_note_fragments(staff_index)

    def _repair_candidate_geometry(self, symbols: list[EncodedSymbol], staff_index: int) -> None:
        """Stage 2: quarantine artifacts and repair mergeable segmentation."""
        self._discard_visual_groups_near_clefs(symbols, staff_index)
        self._consolidate_exact_duplicate_noteheads(staff_index)
        self._discard_duplicate_notehead_fragments(staff_index)
        self._consolidate_split_hollow_noteheads(staff_index)

    def _consolidate_split_hollow_noteheads(self, staff_index: int) -> None:
        """Merge touching halves before they make a complete chord look surplus.

        A staff line can split a hollow head into two adjacent segmentation
        candidates. Waiting until after matching to rejoin them only works when
        attention happens to select one half. Consolidate unambiguous pairs first
        so structural matching sees one physical head at each staff position.
        """
        groups = [
            group
            for group in self.visual_groups.values()
            if (
                group.staff_index == staff_index
                and group.visual_status != "diagnostic"
                and group.is_hollow_notehead
            )
        ]
        possible_pairs = [
            (first, second)
            for first_index, first in enumerate(groups)
            for second in groups[first_index + 1 :]
            if (
                first.stave_index == second.stave_index
                and first.staff_position == second.staff_position
                and self._looks_like_horizontal_notehead_fragment(first, second)
            )
        ]
        possible_pairs.sort(
            key=lambda pair: (
                abs(pair[0].prediction_center[0] - pair[1].prediction_center[0]),
                pair[0].visual_id,
                pair[1].visual_id,
            )
        )
        merged_visual_ids: set[str] = set()
        for first, second in possible_pairs:
            if first.visual_id in merged_visual_ids or second.visual_id in merged_visual_ids:
                continue
            primary, fragment = sorted((first, second), key=lambda group: group.visual_id)
            self._merge_notehead_fragment(primary, fragment)
            primary.provenance = "merged_fragments"
            if "merged_split_notehead_before_matching" not in primary.repair_actions:
                primary.repair_actions.append("merged_split_notehead_before_matching")
            fragment.visual_status = "diagnostic"
            fragment.repair_actions.append(f"merged_into:{primary.visual_id}")
            merged_visual_ids.update((primary.visual_id, fragment.visual_id))

    def _consolidate_exact_duplicate_noteheads(self, staff_index: int) -> None:
        """Keep one physical head when overlapping stave zones emitted it twice.

        ``add_notes_to_staffs`` deliberately permits ledger notes in each nearby
        stave zone. In the overlap between a grand staff's staves, that can create
        two Note objects backed by the exact same segmentation contour but with
        different staff positions. Retain one physical candidate, preserve the
        rejected candidate for diagnostics, and remember both stave positions so a
        structurally complete transformer moment can resolve ownership later.
        """
        staff_groups = [
            group
            for group in self.visual_groups.values()
            if group.staff_index == staff_index and group.visual_status != "diagnostic"
        ]
        visited: set[str] = set()
        for group_index, group in enumerate(staff_groups):
            if group.visual_id in visited:
                continue
            duplicates = [
                candidate
                for candidate in staff_groups[group_index + 1 :]
                if candidate.visual_id not in visited
                and self._same_exact_notehead_candidate(group, candidate)
            ]
            if not duplicates:
                continue
            cluster = [group, *duplicates]
            visited.update(candidate.visual_id for candidate in cluster)
            primary = min(cluster, key=lambda candidate: candidate.visual_id)
            staff_positions = {
                candidate.stave_index: candidate.staff_position for candidate in cluster
            }
            self._duplicate_staff_positions_by_visual_id[primary.visual_id] = staff_positions
            if "duplicate_candidates_consolidated" not in primary.repair_actions:
                primary.repair_actions.append("duplicate_candidates_consolidated")
            if len(staff_positions) > 1 and (
                "cross_stave_duplicate_consolidated" not in primary.repair_actions
            ):
                primary.repair_actions.append("cross_stave_duplicate_consolidated")
            for duplicate in cluster:
                if duplicate is primary:
                    continue
                duplicate.visual_status = "diagnostic"
                if "suspected_duplicate" not in duplicate.repair_actions:
                    duplicate.repair_actions.append("suspected_duplicate")
                duplicate_of = f"duplicate_of:{primary.visual_id}"
                if duplicate_of not in duplicate.repair_actions:
                    duplicate.repair_actions.append(duplicate_of)

    @staticmethod
    def _same_exact_notehead_candidate(first: VisualGroup, second: VisualGroup) -> bool:
        """Return true only for candidates backed by the same segmented pixels."""
        return (
            np.allclose(first.prediction_center, second.prediction_center, atol=1e-6)
            and np.allclose(
                first.prediction_notehead_size,
                second.prediction_notehead_size,
                atol=1e-6,
            )
            and first.notehead_contours == second.notehead_contours
            and first.detected_notehead_contours == second.detected_notehead_contours
            and first.refined_notehead_contours == second.refined_notehead_contours
        )

    def _discard_visual_groups_near_clefs(
        self, symbols: list[EncodedSymbol], staff_index: int
    ) -> None:
        """Quarantine notehead candidates sitting on recognized clef glyphs."""
        clef_centers = [
            symbol.coordinates
            for symbol in symbols
            if symbol.rhythm.startswith("clef") and symbol.coordinates is not None
        ]
        if not clef_centers:
            return
        clef_artifact_ids = {
            group.visual_id
            for group in self.visual_groups.values()
            if group.staff_index == staff_index
            and group.transformer_center is not None
            and any(
                np.linalg.norm(np.subtract(group.transformer_center, clef_center))
                <= MAX_VISUAL_GROUP_DISTANCE_FROM_CLEF
                for clef_center in clef_centers
            )
        }
        for visual_id in clef_artifact_ids:
            group = self.visual_groups[visual_id]
            group.visual_status = "diagnostic"
            if "clef_artifact" not in group.repair_actions:
                group.repair_actions.append("clef_artifact")

    def _discard_duplicate_notehead_fragments(self, staff_index: int) -> None:
        """Quarantine weak horizontal fragments duplicated from a nearby notehead.

        Segmentation can emit a small, hollow-looking fragment beside a full
        notehead while attaching both candidates to the exact same detected stem.
        Keeping both candidates shifts matching through dense note sequences. A
        genuine chord may also share a stem, so require the fragment to have much
        less detected ink and nearly the same vertical center as the full head.
        """
        staff_groups = [
            group for group in self.visual_groups.values() if group.staff_index == staff_index
        ]
        duplicate_ids: set[str] = set()
        for fragment in staff_groups:
            if not fragment.is_hollow_notehead or not fragment.detected_stem_contours:
                continue
            fragment_area = self._detected_notehead_area(fragment)
            fragment_bounds = source_notehead_bounds(fragment)
            if fragment_area <= 0 or fragment_bounds is None:
                continue
            for notehead in staff_groups:
                if (
                    notehead.visual_id == fragment.visual_id
                    or notehead.stave_index != fragment.stave_index
                    or notehead.detected_stem_contours != fragment.detected_stem_contours
                ):
                    continue
                notehead_area = self._detected_notehead_area(notehead)
                notehead_bounds = source_notehead_bounds(notehead)
                if notehead_area <= 0 or notehead_bounds is None:
                    continue
                if fragment_area > notehead_area * DUPLICATE_NOTEHEAD_AREA_RATIO:
                    continue
                fragment_width = fragment_bounds[2] - fragment_bounds[0]
                fragment_height = fragment_bounds[3] - fragment_bounds[1]
                notehead_width = notehead_bounds[2] - notehead_bounds[0]
                notehead_height = notehead_bounds[3] - notehead_bounds[1]
                max_width = max(fragment_width, notehead_width, 1.0)
                max_height = max(fragment_height, notehead_height, 1.0)
                center_dx = abs(fragment.prediction_center[0] - notehead.prediction_center[0])
                center_dy = abs(fragment.prediction_center[1] - notehead.prediction_center[1])
                if (
                    center_dx <= max_width * DUPLICATE_NOTEHEAD_MAX_HORIZONTAL_DISTANCE
                    and center_dy <= max_height * DUPLICATE_NOTEHEAD_MAX_VERTICAL_DISTANCE
                ):
                    duplicate_ids.add(fragment.visual_id)
                    break

        for visual_id in duplicate_ids:
            group = self.visual_groups[visual_id]
            group.visual_status = "diagnostic"
            if "suspected_duplicate" not in group.repair_actions:
                group.repair_actions.append("suspected_duplicate")

    @staticmethod
    def _detected_notehead_area(group: VisualGroup) -> float:
        return sum(
            abs(cv2.contourArea(np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)))
            for contour in group.detected_notehead_contours
            if len(contour) >= 3
        )

    def _merge_split_whole_note_fragments(self, staff_index: int) -> None:
        """Rejoin whole-note heads split into touching horizontal fragments.

        Staff lines can divide the outline of a vertically stacked whole-note chord
        into four segmentation components: a left and right half for each actual
        head. Recognition still emits the correct two-note chord, leaving one half
        of each head unmatched. Rejoin only those tightly touching, stemless hollow
        fragments that share an exact staff position with a recognized whole note.
        """
        matched_groups = [
            group
            for group in self.visual_groups.values()
            if (
                group.staff_index == staff_index
                and group.duration is not None
                and group.duration.rstrip(".") == "note_1"
                and group.is_hollow_notehead
                and not group.stem_contours
                and "merged_split_notehead_before_matching" not in group.repair_actions
            )
        ]
        for group in matched_groups:
            candidates = [
                candidate
                for candidate in self.visual_groups.values()
                if (
                    candidate.visual_id in self.unmatched_visual_notes
                    and candidate.staff_index == group.staff_index
                    and candidate.stave_index == group.stave_index
                    and candidate.staff_position == group.staff_position
                    and candidate.is_hollow_notehead
                    and not candidate.stem_contours
                    and self._looks_like_horizontal_notehead_fragment(group, candidate)
                )
            ]
            if not candidates:
                continue
            fragment = min(
                candidates,
                key=lambda candidate: abs(
                    candidate.prediction_center[0] - group.prediction_center[0]
                ),
            )
            self._merge_notehead_fragment(group, fragment)
            group.provenance = "merged_fragments"
            if "merged_split_notehead" not in group.repair_actions:
                group.repair_actions.append("merged_split_notehead")
            fragment.visual_status = "diagnostic"
            fragment.repair_actions.append(f"merged_into:{group.visual_id}")

    @staticmethod
    def _looks_like_horizontal_notehead_fragment(
        group: VisualGroup, candidate: VisualGroup
    ) -> bool:
        group_bounds = source_notehead_bounds(group)
        candidate_bounds = source_notehead_bounds(candidate)
        if group_bounds is None or candidate_bounds is None:
            return False
        group_left, group_top, group_right, group_bottom = group_bounds
        candidate_left, candidate_top, candidate_right, candidate_bottom = candidate_bounds
        group_width = max(group_right - group_left, 1.0)
        candidate_width = max(candidate_right - candidate_left, 1.0)
        group_height = max(group_bottom - group_top, 1.0)
        candidate_height = max(candidate_bottom - candidate_top, 1.0)
        vertical_overlap = min(group_bottom, candidate_bottom) - max(group_top, candidate_top)
        if vertical_overlap < min(group_height, candidate_height) * 0.75:
            return False
        horizontal_gap = max(
            0.0,
            max(group_left, candidate_left) - min(group_right, candidate_right),
        )
        if horizontal_gap > min(group_width, candidate_width) * 0.2:
            return False
        return (
            abs(group.prediction_center[1] - candidate.prediction_center[1])
            <= min(group_height, candidate_height) * 0.2
        )

    def _merge_notehead_fragment(self, group: VisualGroup, fragment: VisualGroup) -> None:
        group.notehead_contours.extend(fragment.notehead_contours)
        group.detected_notehead_contours.extend(fragment.detected_notehead_contours)
        group.refined_notehead_contours.extend(fragment.refined_notehead_contours)
        points = [point for contour in group.notehead_contours for point in contour]
        if len(points) >= 5:
            ellipse = cv2.fitEllipse(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
            fitted = self.coordinate_transform._ellipse_to_json(ellipse)
            fitted["_fit_source"] = "merged-fragments"
            fitted["_is_hollow"] = True
            group.notehead_ellipses = [fitted]
        group.prediction_center = (
            (group.prediction_center[0] + fragment.prediction_center[0]) / 2,
            (group.prediction_center[1] + fragment.prediction_center[1]) / 2,
        )
