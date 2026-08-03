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
