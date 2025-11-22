import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import itertools

from neo4j import GraphDatabase

BOLT_URI = os.environ.get("BOLT_URI", "bolt://localhost:7690")


class TraversalTool:
    """Traverse the AI Act knowledge graph to assemble full legal text contexts."""

    def __init__(self, uri: str = BOLT_URI, auth=None) -> None:
        self.driver = GraphDatabase.driver(uri, auth=auth) if auth else GraphDatabase.driver(uri)
        self._node_cache: Dict[str, Dict] = {}

    def close(self) -> None:
        self.driver.close()

    # --------------------------------------------------------------------- #
    # Neo4j helpers
    # --------------------------------------------------------------------- #
    def _node_to_dict(self, node) -> Dict:
        data = dict(node)
        data["id"] = node.get("id")
        data["labels"] = list(node.labels)
        return data

    def _get_node(self, node_id: str) -> Optional[Dict]:
        if node_id in self._node_cache:
            return self._node_cache[node_id]
        with self.driver.session() as session:
            record = session.run("MATCH (n {id: $id}) RETURN n", {"id": node_id}).single()
        if not record:
            return None
        node_dict = self._node_to_dict(record["n"])
        self._node_cache[node_id] = node_dict
        return node_dict

    def _get_paragraph_parent(self, node_id: str) -> Optional[str]:
        query = """
        MATCH (parent:Paragraph)-[:CONTAINS]->(child:Paragraph {id: $id})
        RETURN parent.id AS parent_id
        LIMIT 1
        """
        with self.driver.session() as session:
            record = session.run(query, {"id": node_id}).single()
        return record["parent_id"] if record else None

    def _get_top_level_paragraph(self, node_id: str) -> str:
        current_id = node_id
        while True:
            parent_id = self._get_paragraph_parent(current_id)
            if not parent_id:
                return current_id
            current_id = parent_id

    def _get_contains_children(self, node_id: str) -> List[Tuple[str, Optional[int]]]:
        query = """
        MATCH (n {id: $id})-[rel:CONTAINS]->(child)
        RETURN child.id AS child_id, rel.order AS rel_order
        """
        with self.driver.session() as session:
            records = session.run(query, {"id": node_id})
            children = [(record["child_id"], record["rel_order"]) for record in records]
        children.sort(key=lambda item: (item[1] if item[1] is not None else float("inf"), item[0]))
        return children

    def _collect_contains_paths(self, node_id: str, visited: Optional[Set[str]] = None) -> List[List[str]]:
        if visited is None:
            visited = set()
        if node_id in visited:
            return [[node_id]]
        visited.add(node_id)
        children = self._get_contains_children(node_id)
        chains: List[List[str]] = [[node_id]]
        for child_id, _ in children:
            for path in self._collect_contains_paths(child_id, visited.copy()):
                chains.append([node_id] + path)
        return chains

    # --------------------------------------------------------------------- #
    # Text assembly
    # --------------------------------------------------------------------- #
    def _format_node_text(self, node: Dict) -> str:
        parts: List[str] = []
        seen: Set[str] = set()

        header_keys = ("title", "name", "heading", "description")
        for key in header_keys:
            value = node.get(key)
            if isinstance(value, str):
                text = value.strip()
                if text and text not in seen:
                    parts.append(text)
                    seen.add(text)

        main_text = node.get("text")
        if isinstance(main_text, str):
            text = main_text.strip()
            if text and text not in seen:
                parts.append(text)
                seen.add(text)

        if not parts:
            number = node.get("number")
            if number:
                number_text = str(number).strip()
                if number_text:
                    parts.append(number_text)

        return "\n".join(parts).strip()

    def _build_text_from_contains(self, node_id: str, visited: Optional[Set[str]] = None) -> str:
        if visited is None:
            visited = set()
        if node_id in visited:
            return ""
        visited.add(node_id)
        node = self._get_node(node_id)
        if not node:
            return ""
        parts: List[str] = []
        node_text = self._format_node_text(node)
        if node_text:
            parts.append(node_text)

        for child_id, _ in self._get_contains_children(node_id):
            child_text = self._build_text_from_contains(child_id, visited.copy())
            if child_text:
                parts.append(child_text)

        return "\n".join(part for part in parts if part).strip()

    # --------------------------------------------------------------------- #
    # Reference handling
    # --------------------------------------------------------------------- #
    def _collect_reference_nodes(self, node_id: str) -> Dict[str, Set[str]]:
        directions: Dict[str, Set[str]] = defaultdict(set)
        with self.driver.session() as session:
            outgoing = session.run(
                "MATCH (start {id: $id})-[:REFERENCES]->(target) RETURN target.id AS ref_id",
                {"id": node_id},
            )
            for record in outgoing:
                directions[record["ref_id"]].add("outgoing")

            incoming = session.run(
                "MATCH (source)-[:REFERENCES]->(start {id: $id}) RETURN source.id AS ref_id",
                {"id": node_id},
            )
            for record in incoming:
                directions[record["ref_id"]].add("incoming")

        return directions

    def _expand_node(self, node_id: str) -> Optional[Dict]:
        node = self._get_node(node_id)
        if not node:
            return None

        top_level_id = node_id
        if "Paragraph" in node.get("labels", []):
            top_level_id = self._get_top_level_paragraph(node_id)

        full_text = self._build_text_from_contains(top_level_id)
        hierarchy_paths = self._collect_contains_paths(top_level_id)

        return {
            "id": node_id,
            "labels": node.get("labels", []),
            "top_level_id": top_level_id,
            "text": full_text,
            "contains_paths": hierarchy_paths,
        }

    # --------------------------------------------------------------------- #
    # Public traversal API
    # --------------------------------------------------------------------- #
    def traverse(self, start_node_id: str) -> Dict:
        start_expansion = self._expand_node(start_node_id)
        if not start_expansion:
            return {"error": "Node not found", "start_node": start_node_id}

        reference_map = self._collect_reference_nodes(start_node_id)

        reference_payload: List[Dict] = []
        reference_hierarchies: List[Dict] = []
        for ref_id, direction_set in reference_map.items():
            ref_expansion = self._expand_node(ref_id)
            if not ref_expansion:
                continue
            directions = sorted(direction_set)
            reference_payload.append(
                {
                    "id": ref_id,
                    "directions": directions,
                    "text": ref_expansion["text"],
                    #"top_level_id": ref_expansion["top_level_id"],
                }
            )
            # reference_hierarchies.append(
            #     {
            #         "id": ref_id,
            #         "directions": directions,
            #         "top_level_id": ref_expansion["top_level_id"],
            #         "contains_paths": ref_expansion["contains_paths"],
            #     }
            # )

        result = {
            "start_node": start_node_id,
            "text": start_expansion["text"],
            "references": reference_payload,
            "children": list(set(list(itertools.chain(*start_expansion["contains_paths"])))),
        }
        return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Traverse AI Act Knowledge Graph (enhanced)")
    parser.add_argument("--node_id", required=True, help="Node ID to start traversal")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    tool = TraversalTool()
    result = tool.traverse(args.node_id)
    tool.close()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
