from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable

import mgclient


@dataclass
class SchemaFetchConfig:
    host: str = "127.0.0.1"
    port: int = 7690
    username: Optional[str] = None
    password: Optional[str] = None
    query: str = "SHOW SCHEMA INFO;"


def _connect(config: SchemaFetchConfig) -> mgclient.Connection:
    """Create an auto-commit Memgraph connection based on the provided config."""
    kwargs: Dict[str, Any] = {"host": config.host, "port": config.port}
    if config.username is not None:
        kwargs["username"] = config.username
    if config.password is not None:
        kwargs["password"] = config.password
    connection = mgclient.connect(**kwargs)
    connection.autocommit = True
    return connection


def _fetch_schema_payload(connection: mgclient.Connection, query: str) -> Dict[str, Any]:
    """Execute SHOW SCHEMA INFO and load the resulting JSON payload."""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    finally:
        cursor.close()

    if not rows:
        raise RuntimeError("SHOW SCHEMA INFO returned no rows")

    # Memgraph returns a single JSON string containing the schema description.
    raw_payload = rows[0][0]
    if isinstance(raw_payload, (bytes, bytearray)):
        raw_payload = raw_payload.decode("utf-8")
    if isinstance(raw_payload, str):
        return json.loads(raw_payload)
    if isinstance(raw_payload, dict):
        return raw_payload
    raise TypeError(f"Unexpected schema payload type: {type(raw_payload)!r}")

TYPE_MAP = {
    "String": "str",
    "Integer": "int",
    "Float": "float",
    "Boolean": "bool",
    "List": "tuple",
    "Map": "dict",
}

def _fetch_index_payload(connection: mgclient.Connection) -> List[dict]:
    cursor = connection.cursor()
    try:
        cursor.execute("CALL vector_search.show_index_info() YIELD * RETURN *;")
        columns = [desc.name for desc in cursor.description]
        vi_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()
    
    if isinstance(vi_rows, list):
        return vi_rows
    raise TypeError(f"Unexpected schema payload type: {type(vi_rows)!r}")

def _normalize_type(type_entry: str) -> str:
    return TYPE_MAP.get(type_entry, type_entry.lower())


def _collect_node_properties(schema_payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, int]]:
    label_properties: Dict[str, Dict[str, Set[str]]] = {}
    label_order: Dict[str, int] = {}
    for node in schema_payload.get("nodes", []) or []:
        labels = node.get("labels", []) or []
        properties = node.get("properties", []) or []
        for label in labels:
            if label not in label_order:
                label_order[label] = len(label_order)
            label_props = label_properties.setdefault(label, {})
            for prop in properties:
                name = prop.get("key", "<unnamed>")
                types = {
                    _normalize_type(t.get("type", "unknown"))
                    for t in prop.get("types", [])
                }
                if not types:
                    types = {"unknown"}
                label_props.setdefault(name, set()).update(types)
    return label_properties, label_order


def _collect_relationship_data(schema_payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Set[str]]], Set[Tuple[str, str, str]]]:
    relationship_properties: Dict[str, Dict[str, Set[str]]] = {}
    patterns: Set[Tuple[str, str, str]] = set()

    for edge in schema_payload.get("edges", []) or []:
        rel_type = edge.get("type", "<unknown>")
        start_labels = edge.get("start_node_labels", []) or []
        end_labels = edge.get("end_node_labels", []) or []
        for prop in edge.get("properties", []) or []:
            name = prop.get("key", "<unnamed>")
            types = {
                _normalize_type(t.get("type", "unknown"))
                for t in prop.get("types", [])
            }
            if not types:
                types = {"unknown"}
            relationship_properties.setdefault(rel_type, {}).setdefault(name, set()).update(types)
        for start_label in start_labels:
            for end_label in end_labels:
                patterns.add((start_label, rel_type, end_label))

    return relationship_properties, patterns

def _label_combo(labels: Iterable[str]) -> str:
    ordered = sorted(labels)
    return " + ".join(ordered) if ordered else "<no label>"

def _collect_index_data(schema: Dict[str, Any]):
    lines: List[str] = []
    node_indexes = schema.get("node_indexes") or []
    edge_indexes = schema.get("edge_indexes") or []
    if node_indexes:
        lines.append(f"Node indexes ({len(node_indexes)}):")
        for idx in node_indexes:
            labels = _label_combo(idx.get("labels", []))
            properties = ", ".join(idx.get("properties", [])) or "<none>"
            idx_type = idx.get("type", "unspecified")
            count = idx.get("count", "unknown")
            lines.append(f"- {idx_type} on {labels}({properties}), {count} entries")
    if edge_indexes:
        lines.append(f"Edge indexes ({len(edge_indexes)}):")
        for idx in edge_indexes:
            type_ = idx.get("type", "unspecified")
            properties = ", ".join(idx.get("properties", [])) or "<none>"
            count = idx.get("count", "unknown")
            lines.append(f"- {type_} using {properties}, {count} entries")
    return lines


def build_schema_description(schema_payload: Dict[str, Any]) -> str:
    """Render a schema summary similar to the legacy Memgraph helper."""
    node_properties, label_order = _collect_node_properties(schema_payload)
    relationship_properties, patterns = _collect_relationship_data(schema_payload)

    lines: List[str] = []
    lines.append("Node properties are the following:")

    preferred_node_order = [
        "Act",
        "EmbeddingNode",
        "Chapter",
        "Article",
        "Paragraph",
        "Chapter_Section",
        "Annex",
        "Annex_Section",
        "Recital",
    ]

    def preferred_node_index(label: str) -> int:
        try:
            return preferred_node_order.index(label)
        except ValueError:
            return len(preferred_node_order) + label_order.get(label, len(label_order))

    def label_sort_key(label: str) -> Tuple[int, int, str]:
        return (preferred_node_index(label), label_order.get(label, len(label_order)), label)

    for label in sorted(node_properties, key=label_sort_key):
        props = [
            {"property": name, "type": sorted(types)[0]}
            for name, types in sorted(node_properties[label].items())
        ]
        lines.append(f"Node name: '{label}', Node properties: {props}")

    lines.append("")
    lines.append("Relationship properties are the following:")
    if relationship_properties:
        for rel_type in sorted(relationship_properties):
            props = [
                {"property": name, "type": sorted(types)[0]}
                for name, types in sorted(relationship_properties[rel_type].items())
            ]
            lines.append(f"Relationship name: '{rel_type}', Relationship properties: {props}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("The relationships are the following:")
    relationship_type_order = {"CONTAINS": 0, "NEXT": 1, "REFERENCES": 2}

    preferred_relationship_start_order = {
        "CONTAINS": [
            "Act",
            "EmbeddingNode",
            "Chapter",
            "Article",
            "Paragraph",
            "Chapter_Section",
            "Annex",
            "Annex_Section",
            "Recital",
        ],
        "NEXT": [
            "Chapter",
            "EmbeddingNode",
            "Article",
            "Paragraph",
            "Chapter_Section",
            "Annex",
            "Annex_Section",
            "Recital",
        ],
        "REFERENCES": [
            "EmbeddingNode",
            "Paragraph",
            "Chapter",
            "Article",
            "Annex",
            "Recital",
            "Chapter_Section",
            "Annex_Section",
            "Act",
        ],
    }

    preferred_relationship_end_order = {
        "CONTAINS": [
            "Chapter",
            "EmbeddingNode",
            "Annex",
            "Recital",
            "Article",
            "Paragraph",
            "Chapter_Section",
            "Annex_Section",
            "Act",
        ],
        "NEXT": [
            "Chapter",
            "EmbeddingNode",
            "Article",
            "Paragraph",
            "Chapter_Section",
            "Annex",
            "Annex_Section",
            "Recital",
        ],
        "REFERENCES": [
            "EmbeddingNode",
            "Recital",
            "Annex",
            "Paragraph",
            "Article",
            "Chapter",
            "Chapter_Section",
            "Annex_Section",
            "Act",
        ],
    }

    def start_index(rel_type: str, label: str) -> int:
        order = preferred_relationship_start_order.get(rel_type, preferred_node_order)
        try:
            return order.index(label)
        except ValueError:
            return len(order) + label_order.get(label, len(label_order))

    def end_index(rel_type: str, label: str) -> int:
        order = preferred_relationship_end_order.get(rel_type, preferred_relationship_end_order.get("CONTAINS", []))
        try:
            return order.index(label)
        except ValueError:
            return len(order) + label_order.get(label, len(label_order))

    def pattern_sort_key(pattern: Tuple[str, str, str]) -> Tuple[int, int, int, str, str, str]:
        start_label, rel_type, end_label = pattern
        return (
            relationship_type_order.get(rel_type, len(relationship_type_order)),
            start_index(rel_type, start_label),
            end_index(rel_type, end_label),
            start_label,
            rel_type,
            end_label,
        )

    for start_label, rel_type, end_label in sorted(patterns, key=pattern_sort_key):
        lines.append(f"(:{start_label})-[:{rel_type}]->(:{end_label})")
    
    lines.extend(_collect_index_data(schema=schema_payload))

    return "\n".join(lines).strip()


def get_llm_ready_schema_description(config: SchemaFetchConfig | None = None) -> str:
    """
    Fetch Memgraph schema information and return a prompt-ready textual summary.

    Example:
        >>> print(get_llm_ready_schema_description())
    """
    config = config or SchemaFetchConfig()
    connection = _connect(config)
    try:
        payload = _fetch_schema_payload(connection, config.query)
    finally:
        connection.close()
    return build_schema_description(payload)


if __name__ == "__main__":
    print(get_llm_ready_schema_description())
