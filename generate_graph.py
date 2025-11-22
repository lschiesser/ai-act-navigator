"""
Ingest-Loader für AI-Act JSON-Dateien (Article, Annex, Recital) in Memgraph
unter Verwendung von neo4j.
Die Struktur des EU-AI Acts wird als Graph mit Knoten und Kanten abgebildet.
Textinhalte werden (optional) mit Gemma-Embeddings codiert um semantische Suche zu ermöglichen.

Erzeugt:
- Knoten: Chapter, Chapter_Section, Article, Annex, Annex_Section, Recital, Paragraph
- Kanten:
  - :CONTAINS {order:int}  (+ inverse :PART_OF)
  - :NEXT {seq:int} (+ inverse :PREVIOUS)
  - :REFERENCES {ref_kind}

Abgedeckte Struktur laut Vorgabe:
(:Chapter)-[:NEXT]->(:Chapter)
(:Chapter)-[:CONTAINS]->(:Article)
(:Chapter)-[:CONTAINS]->(:Chapter_Section)
(:Chapter_Section)-[:CONTAINS]->(:Article)
(:Chapter_Section)-[:NEXT]->(:Chapter_Section)
(:Article)-[:NEXT]->(:Article)
(:Article)-[:CONTAINS]->(:Paragraph)
(:Annex)-[:NEXT]->(:Annex)
(:Annex)-[:CONTAINS]->(:Annex_Section)
(:Annex_Section)-[:NEXT]->(:Annex_Section)
(:Annex_Section)-[:CONATINS]->(:Paragraph)
(:Paragraph)-[:NEXT]->(:Paragraph)
(:Paragraph)-[:REFERENCES]->(:Chapter|:Chapter_Section|:Article|:Paragraph|:Annex|:Annex_Section|:Recital)

Voraussetzungen:
- Memgraph läuft auf bolt://localhost:7690 (docker-compose)
- ./requirements.txt should be installed in environment
Aufruf:
    python generate_graph.py --embeddings/-e (argument optional, default: no embeddings)
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from neo4j import GraphDatabase
from roman import fromRoman, toRoman
import ollama

import argparse
import id_generator as id_gen

# Konfiguration
# -----------------------------------------------------------------------------
BOLT_URI = os.environ.get("BOLT_URI", "bolt://localhost:7690")
EMBEDDING_MODEL = os.environ.get("MODEL", 'embeddinggemma:latest')
AUTH = None  # Memgraph: i. d. R. keine Auth. Sonst: ("user","pass")
ACT_ID = f"EU-AI-ACT-2024-1689"
RELATED_RECITAL_PATTERN = re.compile(
    r"Related:\s*Recitals?\s+(?:\d+|\d+(?:,\s*\d+)*(?:,\s*and\s+\d+|\s+and\s+\d+))\s*$"
)


def create_ref_id(ref: Dict[str, Any], key: str) -> str:
    if key == "annexes":
        return id_gen.id_annex(ref["number"])
    elif key == "recitals":
        return id_gen.id_recital(ref["number"])
    elif key == "chapters":
        # turn int to roman
        return id_gen.id_chapter(toRoman(int(ref["number"])))
    elif key == "articles":
        return id_gen.id_article(ref["number"])
    elif key == "chapter_sections":
        return id_gen.id_chapter_section(toRoman(ref["number"][0]), ref["number"][1])
    elif "article" in key:
        if "point" in key:
            prefix = f"({ref['points']})"
        else:
            prefix = str(ref["paragraph"])
        # Check if paragraph with that id starts with str(ref["paragraph"])
        # This requires access to the paragraphs data, assuming a global paragraphs dict
        with open(f"./data/articles/article_{ref['number']}.json", "r", encoding="utf-8") as f:
            art_data = json.load(f)
        paragraphs = art_data.get("paragraphs", [])
        for i, p in enumerate(paragraphs):
            if p["text"].startswith(prefix):
                para_num = i + 1
                return id_gen.id_article_paragraph(ref["number"], para_num)
        return None
    elif "annex" in key:
        if "point" in key:
            prefix = f"({ref['points']})"
        else:
            prefix = str(ref["paragraph"])
        # Check if paragraph with that id starts with str(ref["paragraph"])
        # This requires access to the paragraphs data, assuming a global paragraphs dict
        with open(f"./data/annexes/annex_{ref['number']}.json", "r", encoding="utf-8") as f:
            annex_data = json.load(f)
        paragraphs = annex_data.get("paragraphs", [])
        for i, p in enumerate(paragraphs):
            if p["text"].startswith(prefix):
                para_num = i + 1
                return id_gen.id_annex_paragraph(ref["number"], para_num, section_key=ref.get("section"))
        return None
    else:
        raise Exception("Unexpected reference to not article paragraph {ref}")


# -----------------------------------------------------------------------------
# Cypher Snippets
class GraphFiller:
    """
    Fills the Memgraph/Neo4j graph database with nodes and edges according to the AI Act structure.
    Nodes: Chapter, Chapter_Section, Article, Annex, Recital, Annex_Section, Paragraph
    Edges: CONTAINS, NEXT, REFERENCES
    """
    def __init__(self, uri=BOLT_URI, auth=AUTH):
        """Initialize the graph database connection."""
        self.driver = GraphDatabase.driver(uri, auth=auth) if auth else GraphDatabase.driver(uri)
        return

    def clear_graph(self):
        """Clear all nodes and relationships in the graph database."""
        self.run_query("MATCH (n) DETACH DELETE n")
        return

    def close(self):
        """Close the graph database connection."""
        self.driver.close()
        return

    def run_query(self, query, parameters=None):
        """Run a Cypher query and return the result cursor."""
        with self.driver.session() as session:
            return session.run(query, parameters or {})
    
    def run_query_print(self, query, parameters=None):
        """Run a Cypher query and return the result cursor."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            for record in result:
                print("Record:")
                print(record)
            return
    
    def run_query_single(self, query, parameters=None):
        """Run a Cypher query and return a single result record."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return result.single() if result else None
    
    def add_vector_index(self):
        """Create a vector index for embedding similarity search, using cosine similarity."""
        self.run_query("""
            CREATE VECTOR INDEX embed_index
            ON :EmbeddingNode(embed)
            WITH CONFIG {
            "dimension": 768,
            "capacity": 100000,
            "metric": "cos",
            "resize_coefficient": 2,
            "scalar_kind": "f32"
            };
        """)
        return

    def create_reference(self, ref: Dict[str, Any], key: str, from_id: str):
        """Create a REFERENCES relationship from a paragraph to a referenced node."""
        ref_kind = ""
        if ref.get("paragraph") is not None:
            ref_kind = f"{key}_paragraph"
        if isinstance(ref.get("points"), list):
            for p in ref["points"]:
                ref["points"] = p
                self.create_reference(ref, key, from_id)
                return
        elif ref.get("points") is not None:
            ref_kind = f"{ref_kind if ref_kind != '' else key}_point"
        elif key == "sections":
            if "Chapter" in ref.get("text", ""):
                ref_kind = "chapter_sections"
                ref["number"] = ref["number"].split(".")
                ref["number"] = [int(n) for n in ref["number"]]
            else:
                raise Exception("Unexpected AnnexSection reference in paragraph")
        elif ref.get("paragraph") is None:
            ref_kind = key

        ref_id = create_ref_id(ref, ref_kind)
        if ref_id is None:
            # Could not resolve reference {ref} of kind {ref_kind} from {from_id} because error on webpage
            return
        
        self.run_query(
            "MATCH (p:Paragraph {id: $pid}), (target {id: $tid}) MERGE (p)-[:REFERENCES {ref_kind: $kind}]->(target)",
            {
                "pid": from_id,
                "tid": ref_id,
                "kind": ref_kind,
            }
        )
        return

    def add_act(self, act_name):
        """Add the Act node to the graph."""
        act_id = id_gen.id_act()
        self.run_query(
            """
            MERGE (a:Act:EmbeddingNode {id: $id})
            SET a.name = $name, a.embed = $embed
            """,
            {"id": act_id, "name": act_name[0], "embed": act_name[1]}
        )
        return

    def add_chapter(self, chapter_name, chapter_number, next_chapter=None):
        """Add a Chapter node and its relationships to the graph.
        CONTAINS from Act to Chapter,
        NEXT to next Chapter if provided.
        """
        chapter_id = id_gen.id_chapter(chapter_number)
        self.run_query(
            """
            MERGE (c:Chapter:EmbeddingNode {id: $id})
            ON CREATE SET
              c.name   = $name,
              c.embed  = $embed,
              c.number = $number
            ON MATCH SET
              c.name   = $name,
              c.number = $number
            """,
            {"id": chapter_id, "name": chapter_name[0], "embed": chapter_name[1], "number": chapter_number}
        )
        self.run_query(
            "MATCH (a:Act {id: $aid}), (c:Chapter{id: $cid}) MERGE (a)-[:CONTAINS]->(c)",
            {"aid": id_gen.id_act(), "cid": chapter_id}
        )
        if next_chapter:
            next_id = id_gen.id_chapter(next_chapter)
            self.run_query(
                "MATCH (c1:Chapter {id: $id1}), (c2:Chapter {id: $id2}) MERGE (c1)-[:NEXT]->(c2)",
                {"id1": chapter_id, "id2": next_id}
            )
        return

    def add_chapter_section(self, chapter_number, section_number, section_name, last_section_number=None):
        """Add a Chapter_Section node and its relationships to the graph.
        CONTAINS from Chapter to Chapter_Section,
        NEXT to next Chapter_Section if provided.
        """
        section_id = id_gen.id_chapter_section(chapter_number, section_number)
        self.run_query(
            """
            MERGE (s:Chapter_Section:EmbeddingNode {id: $id})
            ON CREATE SET
              s.name   = $name,
              s.embed  = $embed,
              s.number = $number
            ON MATCH SET
              s.name   = $name,
              s.number = $number
            """,
            {"id": section_id, "name": section_name[0], "embed": section_name[1], "number": section_number}
        )
        chapter_id = id_gen.id_chapter(chapter_number)
        self.run_query(
            "MATCH (c:Chapter {id: $cid}), (s:Chapter_Section {id: $sid}) MERGE (c)-[:CONTAINS]->(s)",
            {"cid": chapter_id, "sid": section_id}
        )
        if last_section_number:
            last_id = id_gen.id_chapter_section(chapter_number, last_section_number)
            self.run_query(
                "MATCH (s1:Chapter_Section {id: $id1}), (s2:Chapter_Section {id: $id2}) MERGE (s1)-[:NEXT]->(s2)",
                {"id1": last_id, "id2": section_id}
            )
        return

    def add_article(self, article_number, article_name, chapter_number, next_article, chapter_section=None):
        """Add an Article node and its relationships to the graph.
        CONTAINS from Chapter or Chapter_Section to Article,
        NEXT to next Article if provided.
        """
        article_id = id_gen.id_article(article_number)
        self.run_query(
            """
            MERGE (a:Article:EmbeddingNode {id: $id})
            ON CREATE SET
              a.name   = $name,
              a.embed  = $embed,
              a.number = $number
            ON MATCH SET
              a.name   = $name,
              a.number = $number
            """,
            {"id": article_id, "name": article_name[0], "embed": article_name[1], "number": article_number}
        )
        chapter_id = id_gen.id_chapter(chapter_number)
        
        if chapter_section:
            section_id = id_gen.id_chapter_section(chapter_number, chapter_section)
            self.run_query(
                "MATCH (s:Chapter_Section {id: $sid}), (a:Article {id: $aid}) MERGE (s)-[:CONTAINS]->(a)",
                {"sid": section_id, "aid": article_id}
            )
        else:
            self.run_query(
                "MATCH (c:Chapter {id: $cid}), (a:Article {id: $aid}) MERGE (c)-[:CONTAINS]->(a)",
                {"cid": chapter_id, "aid": article_id}
            )

        if next_article:
            next_id = id_gen.id_article(next_article)
            self.run_query(
                "MATCH (a1:Article {id: $id1}), (a2:Article {id: $id2}) MERGE (a1)-[:NEXT]->(a2)",
                {"id1": article_id, "id2": next_id}
            )
        return


    def add_paragraph(self, parent_id, para_id, text, sub_level, next_p_idx=None, references=None, parent_type="Paragraph"):
        """Add a Paragraph node to a parent node, and reference relationships.
        CONTAINS from parent to Paragraph,
        NEXT to next Paragraph if provided.
        parent_type: one of {"Article", "Paragraph", "Annex_Section", "Annex", "Recital"}
        """
        self.run_query(
            """
            MERGE (p:Paragraph:EmbeddingNode {id: $id})
            ON CREATE SET
              p.text   = $text,
              p.embed  = $embed,
              p.number = $number,
              p.sub_level = $sub_level
            ON MATCH SET
              p.text   = $text,
              p.number = $number
            """,
            {"id": para_id, "text": text[0], "embed": text[1], "number": para_id.split(".")[-1], "sub_level": sub_level}
        )
        # Link paragraph to its parent using the provided parent_type label
        allowed = {"Article", "Paragraph", "Annex_Section", "Annex", "Recital"}
        if parent_type not in allowed:
            raise Exception(f"Unexpected parent_type: {parent_type}")
        query = f"MATCH (parent:{parent_type} {{id: $ptid}}), (p:Paragraph {{id: $pgid}}) MERGE (parent)-[:CONTAINS]->(p)"
        self.run_query(query, {"ptid": parent_id, "pgid": para_id})
        
        if next_p_idx is not None:
            self.run_query(
                "MATCH (p1:Paragraph {id: $id1}), (p2:Paragraph {id: $id2}) MERGE (p1)-[:NEXT]->(p2)",
                {"id1": para_id, "id2": next_p_idx}
            )
        for key, refs in references.items():
            for ref in refs or []:
                self.create_reference(ref, key, para_id)
        return


    def add_annex(self, an_num, name, next_annex=None):
        """Add an Annex node and its relationships to the graph.
        CONTAINS from Act to Annex,
        NEXT to next Annex if provided.
        """
        annex_id = id_gen.id_annex(an_num)
        self.run_query(
            """
            MERGE (a:Annex:EmbeddingNode {id: $id})
            ON CREATE SET
              a.an_num = $an_num,
              a.name   = $name,
              a.embed  = $embed
            ON MATCH SET
              a.an_num = $an_num,
              a.name   = $name
            """,
            {"id": annex_id, "an_num": an_num, "name": name[0], "embed": name[1]}
        )
        self.run_query(
            "MATCH (a:Act {id: $aid}), (ann:Annex {id: $annid}) MERGE (a)-[:CONTAINS]->(ann)",
            {"aid": id_gen.id_act(), "annid": annex_id}
        )
        if next_annex:
            next_id = id_gen.id_annex(next_annex)
            self.run_query(
                "MATCH (a1:Annex {id: $id1}), (a2:Annex {id: $id2}) MERGE (a1)-[:NEXT]->(a2)",
                {"id1": annex_id, "id2": next_id}
            )

    def add_annex_section(self, an_num, s_num, section_id, name, next_section_id=None):
        """Add an Annex_Section node and its relationships to the graph.
        CONTAINS from Annex to Annex_Section,
        NEXT to next Annex_Section if provided.
        """
        self.run_query(
            """
            MERGE (s:Annex_Section:EmbeddingNode {id: $id})
            ON CREATE SET
              s.s_num  = $s_num,
              s.annex  = $annex,
              s.name   = $name,
              s.embed  = $embed
            ON MATCH SET
              s.s_num  = $s_num,
              s.annex  = $annex,
              s.name   = $name
            """,
            {"id": section_id, "s_num": s_num, "annex": an_num, "name": name[0], "embed": name[1]}
        )
        annex_id = id_gen.id_annex(an_num)
        self.run_query(
            "MATCH (a:Annex {id: $aid}), (s:Annex_Section {id: $sid}) MERGE (a)-[:CONTAINS]->(s)",
            {"aid": annex_id, "sid": section_id}
        )
        if next_section_id:
            self.run_query(
                "MATCH (s1:Annex_Section {id: $id1}), (s2:Annex_Section {id: $id2}) MERGE (s1)-[:NEXT]->(s2)",
                {"id1": section_id, "id2": next_section_id}
            )


    def add_recital(self, rec_num, name, next_recital=None):
        """Add a Recital node and its relationships to the graph.
        CONTAINS from Act to Recital,
        NEXT to next Recital if provided.
        """
        rec_id = id_gen.id_recital(rec_num)
        self.run_query(
            """
            MERGE (r:Recital:EmbeddingNode {id: $id})
            ON CREATE SET
              r.name   = $name,
              r.embed  = $embed,
              r.number = $number
            ON MATCH SET
              r.name   = $name,
              r.number = $number
            """,
            {"id": rec_id, "name": name[0], "embed": name[1], "number": rec_num}
        )
        self.run_query(
            "MATCH (a:Act {id: $aid}), (r:Recital {id: $rid}) MERGE (a)-[:CONTAINS]->(r)",
            {"aid": id_gen.id_act(), "rid": rec_id}
        )
        if next_recital:
            next_id = id_gen.id_recital(next_recital)
            self.run_query(
                "MATCH (r1:Recital {id: $id1}), (r2:Recital {id: $id2}) MERGE (r1)-[:NEXT]->(r2)",
                {"id1": rec_id, "id2": next_id}
            )


def recursively_add_paragraphs(
    gf,
    paragraphs,
    parent_id,
    create_embeddings,
    parent_type="Paragraph",
    include_parent_context: bool = False,
    parent_text: Optional[str] = None,
):
    """Recursively add paragraphs and their subparagraphs to the graph."""
    for idx, par in enumerate(paragraphs):
        next_p_idx = paragraphs[idx + 1]["id"] if idx + 1 < len(paragraphs) else None
        clean_text = RELATED_RECITAL_PATTERN.sub("", par["text"]).rstrip()
        embedding_context = None
        if include_parent_context and parent_text:
            embedding_context = f"{parent_text}\n\n{clean_text}"
        gf.add_paragraph(
            parent_id,
            par["id"],
            generate_embedding_of_string(clean_text, create_embeddings, embedding_context),
            par.get("level"),
            next_p_idx=next_p_idx,
            references=par.get("references") or {},
            parent_type=parent_type,
        )
        if par.get("totalSubParagraphs") not in (None, []):
            recursively_add_paragraphs(
                gf,
                par["totalSubParagraphs"],
                par["id"],
                create_embeddings,
                parent_type="Paragraph",
                include_parent_context=include_parent_context,
                parent_text=clean_text,
            )
    return


def create_all_articles(gf, create_embeddings, include_parent_context: bool = False):
    """Load all articles from JSON and add them to the graph."""
    # for all articles 1..113 load json and add to graph
    for art_num in tqdm(range(1, 114), desc="Processing Articles"):      
        with open(f"./data/articles/article_{art_num}.json", "r", encoding="utf-8") as f:
            art_data = json.load(f)
        
        # Add Chapter if not exists
        chapter_roman = art_data["metadata"]["chapter"][0]
        chapter_num = fromRoman(chapter_roman)
        gf.add_chapter(generate_embedding_of_string(art_data["article"]["chapter"]["name"], create_embeddings), chapter_roman, next_chapter=toRoman(chapter_num+1) if chapter_num+1 < 14 else None)
        next_article = art_num + 1 if art_num + 1 < 114 else None
        # Add Chapter_Section if exists
        if "section" in art_data["metadata"]:
            gf.add_chapter_section(chapter_roman, art_data["metadata"]["section"][0], generate_embedding_of_string(art_data["article"]["section"]["name"], create_embeddings), last_section_number=int(art_data["metadata"]["section"][0])-1 if int(art_data["metadata"]["section"][0]) > 1 else None)
            # Add Article
            gf.add_article(art_num, generate_embedding_of_string(art_data["article"]["name"], create_embeddings), chapter_roman, next_article, chapter_section=art_data["metadata"]["section"][0])
        else:
            # Add Article
            gf.add_article(art_num, generate_embedding_of_string(art_data["article"]["name"], create_embeddings), chapter_roman, next_article)

        # Add Paragraphs (and subparagraphs)
        paragraphs = art_data.get("paragraphs", [])
        recursively_add_paragraphs(
            gf,
            paragraphs,
            art_data["article"]["id"],
            create_embeddings,
            parent_type="Article",
            include_parent_context=include_parent_context,
        )
    return

        
def create_all_annexes(gf, create_embeddings, include_parent_context: bool = False):
    """Load all annexes from JSON and add them to the graph."""
    # for all annexes 1..13 load json and add to graph
    for anx_num in tqdm(range(1, 14), desc="Processing Annexes"):
        with open(f"./data/annexes/annex_{anx_num}.json", "r", encoding="utf-8") as f:
            anx_data = json.load(f)
        # Add Annex
        next_annex = anx_num + 1 if anx_num + 1 <= 13 else None
        gf.add_annex(anx_num, generate_embedding_of_string(anx_data["annex"]["name"], create_embeddings), next_annex=next_annex)
        # Add Annex_Sections and Paragraphs
        content = anx_data["content"]
        if content[0]["type"] == "Paragraph":
            recursively_add_paragraphs(
                gf,
                content,
                anx_data["annex"]["id"],
                create_embeddings,
                parent_type="Annex",
                include_parent_context=include_parent_context,
            )
        elif content[0]["type"] == "Section":
            for e_idx, element in enumerate(content):
                next_sec_id = content[e_idx + 1]["id"] if e_idx + 1 < len(content) else None
                gf.add_annex_section(anx_num, e_idx + 1, element["id"], generate_embedding_of_string(element["name"], create_embeddings), next_section_id=next_sec_id)
                paragraphs = element.get("paragraphs", [])
                recursively_add_paragraphs(
                    gf,
                    paragraphs,
                    element["id"],
                    create_embeddings,
                    parent_type="Annex_Section",
                    include_parent_context=include_parent_context,
                )
    return

def create_all_recitals(gf, create_embeddings):
    """Load all recitals from JSON and add them to the graph."""
    # for all recitals 1..180 load json and add to graph
    for rec_num in tqdm(range(1, 181), desc="Processing Recitals"):
        with open(f"./data/recitals/recital_{rec_num}.json", "r", encoding="utf-8") as f:
            rec_data = json.load(f)
        # Add Recital
        next_recital = rec_data["recital"]["number"] + 1 if rec_data["recital"]["number"] + 1 <= 180 else None
        gf.add_recital(rec_num, generate_embedding_of_string(rec_data["recital"]["name"], create_embeddings), next_recital=next_recital)
        
        # Add Paragraphs
        paragraphs = rec_data.get("paragraphs", [])
        recursively_add_paragraphs(gf, paragraphs, rec_data["recital"]["id"], create_embeddings, parent_type="Recital")
    return

def generate_embedding_of_string(
    text: str,
    embed: Tuple[bool, Optional[ollama.Client]],
    embedding_text_override: Optional[str] = None,
) -> Tuple[str, Optional[List[float]]]:
    """Generate an embedding for the given text using the Ollama client.
    Access to the Ollama client necessary.
    Returns text and its embedding or None if not created."""
    create_embed, client = embed
    if not create_embed or client is None:
        return text, None
    embedding = client.embed(
        model=EMBEDDING_MODEL,
        input=embedding_text_override or text,
    )
    return text, embedding["embeddings"][0]
    

# -----------------------------------------------------------------------------
# Example usage: Fill the graph with sample data
if __name__ == "__main__":
    gf = GraphFiller()
    
    gf.clear_graph()
    parser = argparse.ArgumentParser(description="Ingest AI Act JSON files into Memgraph/Neo4j")
    parser.add_argument("-e", "--embeddings", action="store_true", help="Create embeddings for Paragraphs")
    parser.add_argument(
        "--embed-parent-context",
        action="store_true",
        help="Include parent paragraph text when embedding subparagraphs",
    )
    args = parser.parse_args()
    create_embeddings = args.embeddings
    include_parent_context = args.embed_parent_context
    if create_embeddings:
        client = ollama.Client(host='http://crai-ollama:11434')
        gf.add_vector_index()
    else:
        client = None
    
    act_name = generate_embedding_of_string("European Union Artificial Intelligence Act", (create_embeddings, client))
    gf.add_act(act_name)


    # Zwei Durchläufe um Verbindungen zu im ersten Durchlauf noch nicht existierenden Knoten zu ermöglichen:
    # 1. mit Embeddings, 2. damit es schneller geht ohne Embeddings (bleiben bestehen)
    for run in range(2):
        create_all_articles(gf, (create_embeddings, client), include_parent_context)
        create_all_annexes(gf, (create_embeddings, client), include_parent_context)
        create_all_recitals(gf, (create_embeddings, client))
        create_embeddings = False

    


    gf.close()
