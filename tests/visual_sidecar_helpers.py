from typing import Any

import cv2
import numpy as np

from homr.type_definitions import NDArray
from homr.visual_sidecar.models import VisualMatch


def ellipse_contour(
    center: tuple[int, int],
    axes: tuple[int, int],
    angle: int,
    delta: int,
) -> NDArray:
    """Notehead contour points, shaped like a cv2.findContours result.

    cv2.ellipse2Poly is stubbed as returning Sequence[Sequence[int]] even though it
    returns an ndarray, so the reshape every caller needs is not expressible against
    the stub. Recover the array type once here rather than at each fixture.
    """
    return np.asarray(cv2.ellipse2Poly(center, axes, angle, 0, 360, delta)).reshape(-1, 1, 2)


def linked_visual_id(match: VisualMatch) -> str:
    """The visual id of a match the test expects to be linked.

    VisualMatch.visual_id is None for a symbol that was never linked to a visual
    group. Indexing the group storage with it raises an opaque KeyError, so name
    the unlinked symbol here instead.
    """
    if match.visual_id is None:
        raise AssertionError(f"symbol {match.symbol} has no linked visual group")
    return match.visual_id


def musicxml_note_ids(xml: object) -> list[str]:
    ids: list[str] = []

    def walk(node: object) -> None:
        if node.__class__.__name__ == "XMLNote":
            attrs = getattr(node, "_attributes", {})
            if "id" in attrs:
                ids.append(str(attrs["id"]))
        children = []
        if hasattr(node, "get_children"):
            children = node.get_children()
        elif hasattr(node, "children"):
            children = node.children
        for child in children:
            walk(child)

    walk(xml)
    return ids


def unmatched_musicxml_note_ids(sidecar: dict[str, Any]) -> list[str]:
    return sorted(
        note["musicxml_id"] for note in sidecar["notes"] if note["visual_group_id"] is None
    )


def diagnostic_visual_group_ids(sidecar: dict[str, Any]) -> list[str]:
    return sorted(
        group["visual_group_id"]
        for group in sidecar["visual_groups"]
        if group["visual_status"] == "diagnostic"
    )
