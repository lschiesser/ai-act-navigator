from neo4j import GraphDatabase
from .graph_traversal import GraphTraversal
import os


BOLT_URI = os.environ.get("BOLT_URI", "bolt://localhost:7690")

class Subgraph(GraphTraversal):

    def create_node_from_memgraph_node(self, node):
        node_id = node["id"]
        labels = list(node.labels) if node.labels else ["Node"]
        label = labels[0]

        return {
            "id": node_id,
            "label": label,
            "labels": labels,
            "properties": {k: v for k, v in node.items() if k != "embed"}
        }
    
    def get_subgraph(self, start_node_id):
        start_node = self.get_node(start_node_id)
        start_node = self.create_node_from_memgraph_node(start_node)
        edges = []
        nodes = []
        if not start_node:
            return {"error": "Node not found", "start_node": start_node_id}
        
        subnodes = self.get_contains_subnodes(start_node_id)

        referenced_nodes = self.get_references(start_node_id)

        referenced_by_nodes = self.get_referenced_by(start_node_id)
        # TODO figure out what get_parent does
        if start_node.get("text") and "Paragraph" in start_node.get("labels", []):
            parents = self.get_parent_paragraph(start_node_id)
        nodes.append(start_node)
        edges = []
        for subnode in subnodes:
            subnode = self.create_node_from_memgraph_node(subnode)
            edges.append({
                "source": start_node_id,
                "target": str(subnode["id"]),
                "type": "CONTAINS"
            })
            nodes.append(subnode)
        
        for ref_node in referenced_nodes:
            ref_node = self.create_node_from_memgraph_node(ref_node)
            edges.append({
                'start_node': start_node_id,
                "target": str(ref_node["id"]),
                "type": "REFERENCES"
            })
            nodes.append(ref_node)
        
        for ref_by_node in referenced_by_nodes:
            ref_by_node = self.create_node_from_memgraph_node(ref_by_node)

            edges.append({
                "start_node": str(ref_by_node["id"]),
                "target": start_node_id,
                "type": "REFERENCES"
            })
            nodes.append(ref_by_node)

        return {
            'nodes': nodes,
            'edges': edges
        }