from dataclasses import dataclass, field
from typing import Any

from homr.model import Note
from homr.transformer.vocabulary import EncodedSymbol

VISUAL_SIDECAR_VERSION = 3
CROSS_STAFF_ALIGNMENT_METHOD = "cross_staff_repair"
CROSS_STAFF_REPAIR_ACTION = "cross_staff_link_repaired"


def sounding_pitch(symbol: EncodedSymbol) -> str | None:
    if symbol.pitch in ("_", "."):
        return None
    if symbol.lift in ("#", "##", "b", "bb"):
        return f"{symbol.pitch[0]}{symbol.lift}{symbol.pitch[1:]}"
    return symbol.pitch


@dataclass
class VisualGroup:
    visual_id: str
    staff_group_index: int
    staff_index: int
    staff_position: int
    prediction_center: tuple[float, float]
    prediction_notehead_size: tuple[float, float]
    transformer_center: tuple[float, float] | None
    transformer_notehead_size: tuple[float, float] | None
    notehead_ellipses: list[dict[str, Any]]
    notehead_contours: list[list[list[float]]]
    detected_notehead_contours: list[list[list[float]]]
    refined_notehead_contours: list[list[list[float]]]
    detected_stem_contours: list[list[list[float]]]
    stem_contours: list[list[list[float]]]
    owned_stem_component_ids: list[str]
    is_hollow_notehead: bool
    visual_status: str
    provenance: str
    moment_id: str | None = None
    chord_id: str | None = None
    repair_actions: list[str] = field(default_factory=list)
    duration: str | None = None
    musicxml_id: str | None = None

    @property
    def bbox(self) -> list[float]:
        points = [
            point for contour in self.notehead_contours + self.stem_contours for point in contour
        ]
        if not points:
            return []
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]


@dataclass
class MusicXmlNoteRecord:
    musicxml_id: str
    part: int
    measure: int
    musicxml_staff_number: int
    voice: int
    pitch: str | None
    duration: str
    match_confidence: float
    visual_group_id: str | None
    alignment_method: str


@dataclass
class VisualMatch:
    symbol: EncodedSymbol
    visual_id: str | None
    confidence: float
    alignment_method: str


@dataclass
class StructuralMatchPlan:
    assignments: list[tuple[int, int]]
    reserved_group_indices: set[int]
    fallback_assignments: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class StructuralMomentCompatibility:
    staff_by_group_index: dict[int, int]
    fallback_subset: bool = False
    symbol_by_group_index: dict[int, EncodedSymbol] = field(default_factory=dict)


@dataclass
class StemOwnershipCache:
    component_by_fragment_id: dict[int, int]
    owner_note_ids_by_component: dict[int, set[int]]


@dataclass
class SidecarState:
    recovery_notes_by_staff_id: dict[int, list[Note]] = field(default_factory=dict)
    staff_index_by_visual_id: dict[str, int] = field(default_factory=dict)
    staff_position_by_visual_id: dict[str, int] = field(default_factory=dict)
    duplicate_staff_positions_by_visual_id: dict[str, dict[int, int]] = field(default_factory=dict)
    stem_ownership_cache: StemOwnershipCache | None = None
    visual_groups: dict[str, VisualGroup] = field(default_factory=dict)
    matches_by_symbol_id: dict[int, VisualMatch] = field(default_factory=dict)
    musicxml_notes: list[MusicXmlNoteRecord] = field(default_factory=list)
    unmatched_visual_group_ids: set[str] = field(default_factory=set)
    moment_id_by_symbol_id: dict[int, str] = field(default_factory=dict)
    next_musicxml_note_id: int = 1
    next_recovered_visual_id: int = 1
    next_transformer_recovered_visual_id: int = 1
