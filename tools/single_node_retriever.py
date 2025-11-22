from neo4j import GraphDatabase
import os

BOLT_URI = os.environ.get("BOLT_URI", "bolt://localhost:7690")

class SingleNodeRetriever:
    def __init__(self, uri=BOLT_URI):
        self.driver = GraphDatabase.driver(uri)
    
    def close(self):
        self.driver.close()
    
    def get_node(self, node_id):
        with self.driver.session() as session:
            result = session.run("MATCH (n {id: $id}) RETURN n", {"id": node_id})
            record = result.single()
            node = None
            if record:
                node = self.create_node_from_memgraph_node(record["n"])
            return node
    
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