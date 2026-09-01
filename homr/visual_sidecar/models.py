import importlib.metadata
from dataclasses import dataclass, field
from typing import Any

from homr.model import Note
from homr.transformer.vocabulary import EncodedSymbol

VISUAL_SIDECAR_VERSION = 3
CROSS_STAFF_ALIGNMENT_METHOD = "cross_staff_repair"
CROSS_STAFF_REPAIR_ACTION = "cross_staff_link_repaired"

# Sidecars name their producer explicitly. The version alone cannot: it is derived from
# the fork's visual/v* tags with the namespace stripped, so "0.1.0" is indistinguishable
# from upstream liebharc/homr's own v0.1.0 release, and an untagged build reports 0.0.0,
# which reads as stock homr of unknown vintage. Only this name separates the two.
PRODUCER_NAME = "homr-visual"

# Distribution names to try, in order, when reporting the installed homr version.
_DISTRIBUTION_CANDIDATES = ("homr-visual", "homr")


def homr_version() -> str:
    """Return the installed homr version, or a marker if it is not installed.

    poetry-dynamic-versioning derives this from the git tags at build time. A build made
    at a visual/v* tag reports that release ("0.1.0"); one made past it appends the commit
    distance and hash ("0.1.0-post.4+13c68f3"); one made before any release tag exists
    still carries the hash, with 0.0.0 as the base ("0.0.0-post.473+e2eb29e"). The hash
    identifies the source exactly in every case.
    """
    for distribution in _DISTRIBUTION_CANDIDATES:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "0.0.0+unknown"


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
    # Internal-only provenance and ownership.  The serializer deliberately does
    # not expose these implementation details or change the v3 schema.
    split_clump_id: str | None = None
    split_clump_bounds: tuple[int, int, int, int] | None = None
    notehead_core_pixels: frozenset[int] = field(default_factory=frozenset, repr=False)

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
