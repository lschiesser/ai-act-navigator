"""
Node Retrieval Tool for AI Act Knowledge Graph

This tool takes a user query, embeds it, performs vector similarity search, and returns the k most relevant nodes and their metadata (excluding embeddings).
"""
import ollama

from neo4j import GraphDatabase
import os
import argparse
import json

BOLT_URI = os.environ.get("BOLT_URI", "bolt://localhost:7690")
AUTH = ("", "")

class NodeRetrievalTool:
    """Tool for embedding user queries and retrieving relevant nodes from the knowledge graph."""
    def __init__(self, host='http://crai-ollama:11434', k=12):
        self.client = ollama.Client(host=host)
        self.driver = GraphDatabase.driver(BOLT_URI, auth=AUTH)
        self.k = k

    def close(self):
        self.driver.close()

    def embed_query(self, query_text):
        """Embed the user query using Ollama."""
        res = self.client.embed(
            model='embeddinggemma:latest',
            input=query_text,
        )
        return res['embeddings'][0]

    def vector_search(self, query_vec):
        """Perform vector similarity search in the knowledge graph."""
        with self.driver.session() as sess:
            res = sess.run(f"""
                CALL vector_search.search('embed_index', {self.k}, $qvec) YIELD node, similarity
                RETURN node.id AS id,
                       CASE WHEN node.text IS NULL THEN node.name ELSE node.text END AS text,
                       node.number AS number,
                       node.sub_level AS sub_level,
                       node.__labels__ AS labels,
                       similarity
                ORDER BY similarity DESC
            """, qvec=query_vec)
            return res.data()

    def retrieve(self, query_text):
        """Embed the query, perform search, and return top-k nodes and metadata."""
        query_vec = self.embed_query(query_text)
        hits = self.vector_search(query_vec)
        results = []
        for h in hits:
            node_info = {
                "id": h["id"],
                "text": h["text"],
                "number": h.get("number"),
                "sub_level": h.get("sub_level"),
                "labels": h.get("labels"),
                "similarity": round(h["similarity"], 3)
            }
            results.append(node_info)
        # maybe add Reranker here?
        return {
            "query": query_text,
            "results": results
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve relevant nodes from AI Act Knowledge Graph using vector search.")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to return")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    tool = NodeRetrievalTool(k=args.top_k)
    result = tool.retrieve(args.query)
    tool.close()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
