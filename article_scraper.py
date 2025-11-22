#!/usr/bin/env python3
"""
AI Act Legal Text Scraper - Python Version

This script extracts legal paragraphs and references from AI Act HTML content
and outputs structured JSON data suitable for knowledge graph construction.

Requirements:
    pip install beautifulsoup4 lxml requests

Usage:
    python ai_act_scraper.py --file path/to/file.html
    python ai_act_scraper.py --url https://artificialintelligenceact.eu/article/2/
    python ai_act_scraper.py --text "HTML content here"
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin, urlparse
from roman import fromRoman, toRoman

import id_generator as ident

try:
    from bs4 import BeautifulSoup, Tag, NavigableString
    import requests
except ImportError:
    print("Required packages not found. Install with:")
    print("pip install beautifulsoup4 lxml requests")
    sys.exit(1)

# (N) or (start) to (end), optionally with ", point" prefix (case-insensitive)
PARA_RE = re.compile(r'(?:,?\s*point\s*)?\(\s*(\d+)\s*\)(?:\s*to\s*\(\s*(\d+)\s*\))?', re.I)

# "points 2 to 8", "points 2, 3 and 4", "point 2"
POINTS_NUMERIC_RE = re.compile(
    r'\bpoints?\s+('
    r'(?:\d+\s*(?:to|-|–)\s*\d+)'          # range: 2 to 8 / 2-8 / 2–8
    r'|(?:\d+(?:\s*,\s*\d+)*'              # list: 2, 3, 4
    r'(?:\s*(?:and|or)\s*\d+)?)'           # optional tail: and 5 / or 5
    r')\b',
    re.I
)

# "(a), (b) and (c)" (without the "points" keyword)
LETTERS_LIST_RE = re.compile(
    r'\(\s*[a-z]\s*\)(?:\s*,\s*\(\s*[a-z]\s*\))*'
    r'(?:\s*,?\s*(?:and|or)\s*\(\s*[a-z]\s*\))?', re.I
)

# "points (a), (b) and (c)"
POINTS_LETTERS_RE = re.compile(
    r'\bpoints?\s+(' + LETTERS_LIST_RE.pattern + r')', re.I
)

SEP_RE = re.compile(r'\s*(?:,|;|\band\b|\bor\b)\s*', re.I)

HREF_ARTICLE_RE = re.compile(r'(?i)\barticle/(\d+)\b')


class AIActScraper:
    """Scraper for extracting structured data from AI Act HTML content."""
    
    def __init__(self, base_url: str = "https://artificialintelligenceact.eu"):
        self.base_url = base_url
        
    def scrape_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Scrape HTML content from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return self.scrape_html(html_content)
    
    def scrape_from_url(self, url: str) -> Dict[str, Any]:
        """Scrape HTML content from a URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return self.scrape_html(response.text)
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch URL {url}: {e}")
    
    def scrape_from_text(self, html_content: str) -> Dict[str, Any]:
        """Scrape HTML content from text string."""
        return self.scrape_html(html_content)
    
    def scrape_html(self, html_content: str) -> Dict[str, Any]:
        """Main scraping function that processes HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        soup = soup.find(id="aia-explorer-content")
        for data in soup(["script", "style"]):
            data.decompose()
        
        # Determine content type (Article or Annex)
        title_element = soup.find('h1', class_='entry-title')
        if title_element and title_element.get_text().lower().startswith("article"):
            content_type = "Article"
        elif title_element and title_element.get_text().lower().startswith("annex"):
            content_type = "Annex"
        elif title_element and title_element.get_text().lower().startswith("recital"):
            content_type = "Recital"
        
        result = {
            "article": None,
            "annex": None,
            "recital": None,
            "sections": [],
            "paragraphs": [],
            "references": {
                "articles": [],
                "chapters": [],
                "annexes": [],
                "sections": [],
                "recitals": []
            },
            "metadata": {
                "extractedAt": datetime.now().isoformat(),
                "sourceType": f"AI Act {content_type}",
                "contentType": content_type
            }
        }
        
        if content_type == "Article":
            # Extract article information
            article_info = self._extract_article_info(soup)
            result["article"] = article_info
            # Extract paragraphs from main content
            result["paragraphs"] = self._extract_article_paragraphs(soup, article_info["number"])
            meta = soup.find("div", class_="aia-post-meta-wrapper")
            a = meta.find_all("p")[-1].find_all("a")
            chapter_text = a[0].get_text()
            c = re.search(r"Chapter\s+([IVXL]+):\s*(.+)", chapter_text, re.IGNORECASE)
            if c:
                result["metadata"]["chapter"] = [c.group(1), c.group(2)]
            section_text = a[-1].get_text()
            s = re.search(r"Section\s+(\d+):\s*(.+)", section_text, re.IGNORECASE)
            if s:
                result["metadata"]["section"] = [(s.group(1)), s.group(2)]

        elif content_type == "Annex":
            # Extract annex information
            annex_info = self._extract_annex_info(soup)
            result["annex"] = annex_info
            # Extract sections and paragraphs from annex content
            result["content"] = self._extract_annex_content(soup, annex_info["number"])
        else:
            recital_info = self._extract_recital_info(soup)
            result["recital"] = recital_info
            result["paragraphs"] = self._extract_recital_content(soup, recital_info["number"])
        
        # Extract all references, excluding those already found in paragraphs
        result["references"] = self._extract_document_level_references(soup, result["paragraphs"], content_type)
        
        return result
    
    def _extract_article_info(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Extract article title, number, summary, and metadata."""
        article_info = {}
        
        # Extract article title
        title_element = soup.find('h1', class_='entry-title')
        if title_element:
            title_text = title_element.get_text()
            article_info["name"] = title_text
            article_info["number"] = self._extract_article_number(title_text)
            article_info["id"] = ident.id_article(article_info["number"])
        
        # Extract summary from CLaiRK section
        summary_element = soup.find('div', class_='aia-clairk-summary-content')
        if summary_element:
            summary_p = summary_element.find('p')
            if summary_p:
                article_info["summary"] = summary_p.get_text()
        
        # Extract chapter information
        chapter_link = soup.find('a', href=re.compile(r'/chapter/'))
        if chapter_link:
            article_info["chapter"] = {
                "name": chapter_link.get_text(),
                "url": chapter_link.get('href', '')
            }
    
        # Extract section information
        section_link = soup.find('a', href=re.compile(r'/section/'))
        if section_link:
            article_info["section"] = {
                "name": section_link.get_text(),
                "url": section_link.get('href', '')
            }

        # Extract implementation date
        eif_value = soup.find('p', class_='aia-eif-value')
        if eif_value:
            article_info["entryIntoForce"] = eif_value.get_text(strip=True)
        return article_info
        
    def _extract_annex_info(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Extract annex title, number, summary, and metadata."""
        annex_info = {}
        
        # Extract annex title
        title_element = soup.find('h1', class_='entry-title')
        if title_element:
            title_text = title_element.get_text()
            annex_info["name"] = title_text
            annex_info["identifier"] = self._extract_annex_identifier(title_text)
            annex_info["number"] = fromRoman(annex_info["identifier"])
            annex_info["id"] = ident.id_annex(fromRoman(annex_info["identifier"]))
        
        # Extract summary from CLaiRK section
        summary_element = soup.find('div', class_='aia-clairk-summary-content')
        if summary_element:
            summary_p = summary_element.find('p')
            if summary_p:
                annex_info["summary"] = summary_p.get_text()
        
        return annex_info if annex_info else None
    
    def _extract_recital_info(self, soup: BeautifulSoup) -> Optional[str]:
        recital_info = {}

        title_element = soup.find('h1', class_="entry-title")
        if title_element:
            title_text = title_element.get_text()
            recital_info["name"] = title_text
            recital_info["number"] = self._extract_recital_number(title_text)
            recital_info["id"] = ident.id_recital(recital_info["number"])

        return recital_info if recital_info else None
    
    
    def process_elements_recursively(self, element_number, elements, parent_id, level):
        paragraphs = []
        n = len(elements)
        i = 0
        prev_para_id = ""

        while i < n:
            elem = elements[i]
            if elem.name != 'p':
                i += 1
                continue
            text = elem.get_text()
            # Skip "[*]" lines
            if text.startswith('[*]') or text.startswith('[58]'):
                i += 1
                continue
            is_sub, sub_level = self._is_sub_paragraph(elem)
            if not text:
                i += 1
                continue
            elif text.startswith('Related:'):
                for p in reversed(paragraphs):
                    if p["level"] == sub_level:
                        p["references"]["recitals"].extend(self._extract_references_from_element(elem)["recitals"])
                        break
                i += 1
                continue

            number_match = re.match(r'^(\d+)\.', text[:5])
            if not number_match:
                number_match = re.match(r'^\((\d+)\)', text[:5])
            letter_match = re.match(r'^\(([a-z])\)', text[:5])
            roman_match = re.match(r'^\(([ivxlcdm]+)\)', text[:5])

            if number_match:
                mini_id = number_match.group(1)
            elif letter_match:
                mini_id = letter_match.group(1)
            elif roman_match:
                mini_id = roman_match.group(1)
            elif i == 0:
                mini_id = str(i)
            # Fallback if sub-paragraph does not start with identifier, merging with previous id
            elif prev_para_id:
                # prev_para_id of the form "string.number.letter/number.roman" where level 0 starts with "letter/number"
                mini_id = prev_para_id.split(".")[level+2]
            # Fallback if non-sub paragraph does not start with identifier, merging with previous id
            elif para_id:
                mini_id = para_id.split(".")[-1]
            else:
                raise ValueError(f"Could not determine paragraph identifier for article {element_number}, element {i} text: {text}")

            para_id = f"{parent_id}.{mini_id}"
            paragraph = {
                "type": "Paragraph",
                "id": para_id,
                "text": text,
                "references": self._extract_references_from_element(elem),
                "totalSubParagraphs": [],
                "level": sub_level if is_sub else level
            }

            # 
            j = i + 1
            child_block = []
            while j < n:
                elem2 = elements[j]
                if elem2.name != 'p':
                    j += 1
                    continue
                _, next_sub_level = self._is_sub_paragraph(elem2)
                if next_sub_level is None:
                    break
                elif next_sub_level > paragraph["level"]:
                    child_block.append(elem2)
                    j += 1
                else:
                    break
            # Recursively process sub-paragraphs
            if child_block:
                children, prev_para_id = self.process_elements_recursively(element_number, child_block, para_id, paragraph["level"] + 1)
                # Merge texts, refs and subParagraphs of children with the same id
                paragraph["totalSubParagraphs"] = children

            paragraphs.append(paragraph)

            # Jump over child block
            i = j if child_block else i + 1

        # merge paragraphs of same id
        results = []
        prev = paragraphs[0]
        for para in paragraphs[1:]:
            if para["id"] == prev["id"]:
                prev["text"] += " " + para["text"]
                prev["references"]["articles"].extend(x for x in para["references"]["articles"] if x not in prev["references"]["articles"])
                prev["references"]["chapters"].extend(x for x in para["references"]["chapters"] if x not in prev["references"]["chapters"])
                prev["references"]["annexes"].extend(x for x in para["references"]["annexes"] if x not in prev["references"]["annexes"])
                prev["references"]["sections"].extend(x for x in para["references"]["sections"] if x not in prev["references"]["sections"])
                prev["references"]["recitals"].extend(x for x in para["references"]["recitals"] if x not in prev["references"]["recitals"])
                prev["totalSubParagraphs"].extend(para["totalSubParagraphs"])
            else:
                results.append(prev)
                prev = para
        results.append(prev)
        return results, para_id
    

    def _extract_recital_content(self, soup: BeautifulSoup, number) -> List[str]:
        paragraphs = []
        main_content = soup.find("div", class_="et_pb_post_content")
        if not main_content:
            return paragraphs
        i = 1
        for element in main_content:
            if element.name == "p":
                text = element.get_text()
                if not text or re.match(r'\[\d+\]', text):
                    continue
                paragraphs.append({
                    "text": text,
                    "id": f"recital{number}_p{i}",
                    "references": self._extract_references_from_element(element)
                })
                i += 1
        return paragraphs
    
    def _extract_annex_content(self, soup: BeautifulSoup, number) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract sections and paragraphs from annex content."""
        content = []
        sections = []
        paragraphs = []
        
        # Find main content area
        main_content = soup.find('div', class_='et_pb_post_content')
        if not main_content:
            return sections, paragraphs
        
        current_section = None
        
        # Process all elements in order
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p']):
            
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # add previous section's paragraphs as children to current section
                if current_section:
                    current_section["paragraphs"], _ = self.process_elements_recursively(number, paragraphs, current_section["id"], 0)
                    sections.append(current_section)
                    paragraphs = []
                section_text = element.get_text()
                # create new current section
                if section_text:
                    current_section = {
                        "type": "Section",
                        "id": ident.id_annex_section(number, section_text.split(' ')[1]),
                        "name": section_text,
                        "paragraphs": []
                    }
            elif element.name == 'p':
                paragraphs.append(element)

        # add last section's paragraphs as children to current section
        if current_section:
            current_section["paragraphs"], _ = self.process_elements_recursively(number, paragraphs, current_section["id"], 0)
            sections.append(current_section)
            content = sections
        # No sections found, process all paragraphs as direct children of annex
        else:
            content, _ = self.process_elements_recursively(number, paragraphs, ident.id_annex(number), 0)
        
        return content


    def _extract_article_paragraphs(self, soup: BeautifulSoup, article_number) -> List[Dict[str, Any]]:
        """Recursively extract paragraphs and sub-paragraphs from the main content area."""
        paragraphs = []

        # Find main content area
        main_content = soup.find('div', class_='et_pb_post_content')
        if not main_content:
            return paragraphs

        id_article = ident.id_article(article_number)

        # Start recursion from top-level <p> tags
        paragraphs, _ = self.process_elements_recursively(article_number, main_content.find_all('p', recursive=False), id_article, 0)

        return paragraphs
    
    def _is_sub_paragraph(self, element: Tag) -> bool:
        """Check if paragraph is indented (sub-paragraph)."""
        style = element.get('style', '')
        isSub = 'padding-left' in style
        level = None
        if isSub:
            m = re.match(r"padding-left: (\d+)", style)
            level = int(int(m.group(1)) / 40)
        return isSub, level
    
    def _collect_window(self, start_node, max_chars=300):
        """Sammle Text ab start_node bis Satzende (. oder ;) oder Blockgrenze."""
        pieces, chars = [], 0
        # after the link
        for el in start_node.next_elements:
            if isinstance(el, NavigableString):
                t = str(el)
                pieces.append(t)
                chars += len(t)
                if any(ch in t for ch in ('.',';')):
                    break
                if chars >= max_chars:
                    break
            else:
                # Bei Block- oder Zeilenumbruch-Tags stoppen, wenn bereits Text gesammelt.
                if el.name in {'p','div','li','br','h1','h2','h3'} and pieces:
                    break
        after = " ".join("".join(pieces).split()).strip()

        max_chars = max_chars/2
        pieces, chars = [], 0
        # before the link
        for el in start_node.previous_elements:
            if isinstance(el, NavigableString):
                t = str(el)
                pieces.append(t)
                chars += len(t)
                if any(ch in t for ch in ('.',';')):
                    break
                if chars >= max_chars:
                    break
            else:
                # Bei Block- oder Zeilenumbruch-Tags stoppen, wenn bereits Text gesammelt.
                if el.name in {'p','div','li','br','h1','h2','h3'} and pieces:
                    break
        # Whitespace normalisieren
        before = " ".join("".join(pieces).split()).strip()
        return before, after

    def _numbers_from_numeric_points(s: str) -> List[int]:
        """Extract [2,3,...] from 'points 2 to 8' / 'points 2, 3 and 4'."""
        if 'to' in s or '-' in s or '–' in s:
            # range
            m = re.search(r'(\d+)\s*(?:to|-|–)\s*(\d+)', s, re.I)
            if not m:
                return []
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            return list(range(a, b + 1))
        # list
        return [int(x) for x in re.findall(r'\d+', s)]

    def _letters_from_body(body: str) -> List[str]:
        return re.findall(r'\(([a-z])\)', body, re.I)

    def parse_context_structured(context: List[str], check_before: bool = False) -> List[Dict[str, object]]:
        """
        Returns a list of {"paragraph": int, "points": Optional[List[str|int]]}
        Works both for '(5)(a) and (6)' and for '... referred to in points 2 to 8 ...'.
        Note: 'context' should be an indexable sequence (list/tuple), not a set.
        """
        out: List[Dict[str, object]] = []
        s = context[0] if check_before else context[2]
        i = 0
        n = len(s)

        while i < n:
            # 1) Parenthesized paragraph (possibly a range)
            m_para = PARA_RE.search(s, i)
            m_num = POINTS_NUMERIC_RE.search(s, i)
            m_let = POINTS_LETTERS_RE.search(s, i)

            # pick the earliest match among the three
            candidates = [(m_para, 'para'), (m_num, 'num'), (m_let, 'let')]
            candidates = [(m, t) for (m, t) in candidates if m]
            if not candidates:
                break
            m, typ = min(candidates, key=lambda x: x[0].start())

            # advance cursor to the match
            i = m.end()

            if typ == 'para':
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else None

                # optional letters right after
                points: Optional[List[str]] = None
                # try "points (a), (b)" first
                m_after = POINTS_LETTERS_RE.match(s, i)
                if m_after:
                    points = AIActScraper._letters_from_body(m_after.group(1)) or None
                    i = m_after.end()
                else:
                    # or bare "(a), (b)" sequence
                    m_after2 = LETTERS_LIST_RE.match(s, i)
                    if m_after2:
                        points = AIActScraper._letters_from_body(m_after2.group(0)) or None
                        i = m_after2.end()

                if end is not None:
                    if end < start:
                        start, end = end, start
                    for pnum in range(start, end + 1):
                        out.append({"paragraph": pnum, **({"points": points} if points else {})})
                else:
                    out.append({"paragraph": start, **({"points": points} if points else {})})

            elif typ == 'num':
                # 'points 2 to 8' or 'points 2, 3 and 4'
                nums = AIActScraper._numbers_from_numeric_points(m.group(1))
                # We don't have an explicit "(N)" paragraph here; treat as points-only references
                # If you want them attached to a current paragraph context, adapt here.
                out.append({"paragraph": None, "points": nums})

            else:  # 'let' (letters with "points")
                letters = AIActScraper._letters_from_body(m.group(1)) or None
                out.append({"paragraph": None, **({"points": letters} if letters else {})})

            # optional separators
            m_sep = SEP_RE.match(s, i)
            if m_sep:
                i = m_sep.end()
                continue

        return out
    

    def _extract_article_refs(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract all article references from the given BeautifulSoup object."""
        results = []
        seen = set()  # zur Deduplizierung (article, context[:80])

        # Anker: /article/N in href – auch wenn Ankertext nur „N“ ist.
        for a in soup.find_all('a', href=True):
            m = HREF_ARTICLE_RE.search(a['href'])
            if not m:
                continue
            art = int(m.group(1))
            anchor_text = a.get_text()  # z.B. "Article 5" oder "50"
            before, after = self._collect_window(a)

            context = (before, anchor_text, after[len(anchor_text):])  # Text direkt NACH dem Link
            # Progressive processing: parse trailing structured references
            context_refs = AIActScraper.parse_context_structured(context, check_before=False)
            if context_refs is None:
                # to edit: change this to work with gemma3
                context_refs = AIActScraper.parse_context_structured(context, check_before=True)
                if context_refs:
                    subparagraph_info = "before"
                else:
                    subparagraph_info = "none"
            else:
                subparagraph_info = "after"
            if context_refs:
                for tref in context_refs:
                    key = (art, tref.get("paragraph"), str(tref.get("points")))
                    if key not in seen:
                        results.append({
                            "article": art,
                            "paragraph": tref.get("paragraph"),
                            "points": tref.get("points"),
                            "source": "href",
                            "context": before + anchor_text if subparagraph_info == "before" else after,
                            "subparagraph_info": subparagraph_info
                        })
                        seen.add(key)
            else:
                key = (art, None, None)
                if key not in seen:
                    results.append({
                        "article": art,
                        "paragraph": None,
                        "points": None,
                        "source": "href",
                        "context": before + anchor_text if subparagraph_info == "before" else after,
                        "subparagraph_info": subparagraph_info
                    })
                    seen.add(key)
        # print("Article refs extracted:", [(r["article"], r["paragraph"], r["points"]) for r in results])
        return results
    
    def extract_chapter_refs(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract if a Section is whats actually referenced here."""
        results = []
        seen = set()  # zur Deduplizierung (chapter, context[:80])

        # Anker: /chapter/N in href – auch wenn Ankertext nur „N“ ist.
        for a in soup.find_all('a', href=True):
            m = re.search(r'(?i)\bchapter/(\d+)\b', a['href'])
            if not m:
                continue
            chapter = m.group(1)
            anchor_text = a.get_text()  # z.B. "Chapter IV" oder "IV"
            before, after = self._collect_window(a)

            trail = after[len(anchor_text):]  # Text direkt NACH dem Link
            is_section_ref = False
            section_match = re.search(r'Sections?\s+((?:\d+\s*(?:,|and)?\s*)+)', trail, re.IGNORECASE)
            numbers = [None]
            if section_match:
                is_section_ref = True
                numbers = re.findall(r'\d+', section_match.group(1))
            else:
                section_match = re.search(r'Sections?\s+((?:\d+\s*(?:,|and)?\s*)+)', before, re.IGNORECASE)
                if section_match:
                    is_section_ref = True
                    numbers = re.findall(r'\d+', section_match.group(1))

            for n in numbers:
                key = (chapter, n)
                context = anchor_text + trail
                if key not in seen:
                    results.append({
                        "chapter": chapter,
                        "section": n if is_section_ref else None,
                        "is_section_ref": is_section_ref,
                        "source": "href",
                        "context": context
                    })
                    seen.add(key)
        # print("Chapter refs extracted:", [r["chapter"] for r in results])
        return results

    def _extract_annex_refs(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract all annex references from the given BeautifulSoup object. similar to _extract_article_refs"""
        results = []
        seen = set()  # zur Deduplizierung (article, context[:80])

        # Anker: /annex/N in href – auch wenn Ankertext nur „N“ ist.
        for a in soup.find_all('a', href=True):
            m = re.search(r'(?i)\bannex/(\d+)\b', a['href'])
            if not m:
                continue
            annex_num = int(m.group(1))
            anchor_text = a.get_text()  # z.B. "Annex I" oder "I"
            before, after = self._collect_window(a)
            # to avoid url confusion
            if not annex_num == fromRoman(anchor_text.split(" ")[-1]):
                continue

            context = (before, anchor_text, after[len(anchor_text):])  # Text direkt NACH dem Link
            # Progressive processing: parse trailing structured references
            context_refs = AIActScraper.parse_context_structured(context, check_before=False)
            if context_refs is None:
                # to edit: parse leading structured references
                context_refs = AIActScraper.parse_context_structured(context, check_before=True)
                if context_refs:
                    subparagraph_info = "before"
                    # print("Warning: Found structured refs BEFORE anchor:", context_refs)
                else:
                    subparagraph_info = "none"
            else:
                subparagraph_info = "after"
            if context_refs:
                for tref in context_refs:
                    key = (annex_num, tref.get("paragraph"), str(tref.get("points")))
                    if key not in seen:
                        results.append({
                            "annex": annex_num,
                            "paragraph": tref.get("paragraph"),
                            "points": tref.get("points"),
                            "href": a['href'],
                            "context": before + anchor_text if subparagraph_info == "before" else after,
                            "subparagraph_info": subparagraph_info
                        })
                        seen.add(key)
            else:
                key = (annex_num, None, None)
                if key not in seen:
                    results.append({
                        "annex": annex_num,
                        "paragraph": None,
                        "points": None,
                        "href": a['href'],
                        "context": before + anchor_text if subparagraph_info == "before" else after,
                        "subparagraph_info": subparagraph_info
                    })
                    seen.add(key)
        # print("Annex refs extracted:", [(r["annex"], r["paragraph"], r["points"]) for r in results])
        return results


    def _extract_references_from_element(self, element: Tag) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all references from a specific element."""
        references = {
            "articles": [],
            "chapters": [],
            "annexes": [],
            "sections": [],
            "recitals": []
        }
        
        # Extract linked references
        links = element.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            text = link.get_text(strip=True, separator=" ")
            
            if '/article/' in href:
                parsed_refs = self._extract_article_refs(element)
                for ref in parsed_refs:
                    # if ref doesnt start with the string of variable text, continue
                    if not ref["context"].startswith(text) and not text.endswith(ref["context"]):
                        continue
                    if ref["points"] and len(ref["points"]) > 1:
                        for pt in ref["points"]:
                            references["articles"].append({
                                "number": ref["article"],
                                "paragraph": ref["paragraph"] if ref else None,
                                "points": pt,
                                "text": text,
                                "url": href
                            })
                    else:
                        references["articles"].append({
                            "number": ref["article"],
                            "paragraph": ref["paragraph"] if ref else None,
                            "points": ref["points"] if ref else None,
                            "text": text,
                            "url": href
                        })
            elif '/chapter/' in href:
                parsed_refs = self.extract_chapter_refs(element)
                for ref in parsed_refs:
                    # if ref doesnt start with the string of variable text, continue

                    # change to
                    if not ref["context"].startswith(text):
                        # print("Skipping chapter ref not matching text:", ref, text)
                        if not ref["context"].endswith(text):
                            # print("... and text doesn't end with it either, skipping.")
                            continue
                    elif ref["is_section_ref"]:
                        chapter_num = self._extract_number_from_url(href, 'chapter')
                        if chapter_num:
                            references["sections"].append({
                                "number": f"{chapter_num}.{ref["section"]}",
                                "text": text + " Section " + str(ref["section"]),
                                "url": href
                            })
                        continue
                    references["chapters"].append({
                        "number": ref["chapter"],
                        "text": text,
                        "url": href
                    })
            elif '/section' in href:
                section_num = self._extract_number_from_url(href, 'section')
                if section_num:
                    references["sections"].append({
                        "number": f"{section_num}.{text.split(" ")[-1]}",
                        "text": text,
                        "url": href
                    })
            elif '/annex/' in href:
                parsed_refs = self._extract_annex_refs(element)
                for ref in parsed_refs:
                    # if ref doesnt start or end with the string of variable text, continue
                    if not ref["context"].startswith(text) and not ref["context"].endswith(text):
                        # print("Skipping annex ref not matching text:", ref, text)
                        continue

                    if ref["points"] and len(ref["points"]) > 1:
                        for pt in ref["points"]:
                            references["annexes"].append({
                                "number": ref["annex"],
                                "paragraph": ref["paragraph"] if ref else None,
                                "points": pt,
                                "text": text,
                                "url": href
                            })
                    else:
                        references["annexes"].append({
                            "number": ref["annex"],
                            "paragraph": ref["paragraph"] if ref else None,
                            "points": ref["points"] if ref else None,
                            "text": text,
                            "url": href
                        })

            elif '/recital/' in href:
                recital_num = self._extract_number_from_url(href, 'recital')
                if recital_num:
                    references["recitals"].append({
                        "number": recital_num,
                        "text": text,
                        "url": href
                    })
        
        
        # Remove duplicates
        for ref_type in references:
            references[ref_type] = self._remove_duplicate_references(references[ref_type], ref_type)
        
        return references
    
    def _extract_document_level_references(self, soup: BeautifulSoup, paragraphs: List[Dict[str, Any]], content_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract references from the entire document, excluding those already found in paragraphs."""
        document_references = {
            "articles": [],
            "chapters": [],
            "annexes": [],
            "sections": [],
            "recitals": []
        }
        
        # Collect all paragraph-level references for comparison
        paragraph_refs = self._collect_paragraph_references(paragraphs)
        
        # Extract recitals from suitable recitals section
        recital_links = soup.find_all('a', href=re.compile(r'/recital/'))
        for link in recital_links:
            recital_num = self._extract_number_from_url(link.get('href'), 'recital')
            if recital_num and content_type != "Recital":
                document_references["recitals"].append({
                    "number": recital_num,
                    "text": f"Recital {recital_num}",
                    "url": link.get('href')
                })
        
        # Extract all article references
        article_links = soup.find_all('a', href=re.compile(r'/article/'))
        for link in article_links:
            if ("Next" in link.get_text()) or ("Previous" in link.get_text()):
                continue
            article_num = self._extract_number_from_url(link.get('href'), 'article')
            if article_num and not self._is_reference_in_paragraphs(article_num, "articles", paragraph_refs):
                document_references["articles"].append({
                    "number": article_num,
                    "text": link.get_text(),
                    "url": link.get('href')
                })
        
        # Extract annex references 
        annex_links = soup.find_all('a', href=re.compile(r'/annex/'))
        for link in annex_links:
            href = link.get('href')
            # Extract identifier from URL (could be Roman or Arabic)
            annex_match = re.search(r'/annex/([ivx\d]+)/?$', href, re.IGNORECASE)
            if annex_match:
                identifier = annex_match.group(1).upper()
                if content_type != "Recital" and not self._is_annex_reference_in_paragraphs(identifier, paragraph_refs):
                    document_references["annexes"].append({
                        "identifier": identifier,
                        "text": link.get_text(),
                        "url": href
                    })
        
        # Remove duplicates
        for ref_type in document_references:
            document_references[ref_type] = self._remove_duplicate_references(document_references[ref_type], ref_type)
        
        return document_references
    
    def _collect_paragraph_references(self, paragraphs: List[Dict[str, Any]]) -> Dict[str, set]:
        """Collect all references found in paragraphs for deduplication."""
        paragraph_refs = {
            "articles": set(),
            "chapters": set(),
            "annexes": set(),
            "sections": set(),
            "recitals": set()
        }
        
        for paragraph in paragraphs:
            if "references" in paragraph:
                refs = paragraph["references"]
                
                # Collect article references
                for ref in refs.get("articles", []):
                    if "number" in ref:
                        paragraph_refs["articles"].add(ref["number"])
                
                # Collect chapter references
                for ref in refs.get("chapters", []):
                    if "number" in ref:
                        paragraph_refs["chapters"].add(ref["number"])
                
                # Collect annex references
                for ref in refs.get("annexes", []):
                    if "identifier" in ref:
                        paragraph_refs["annexes"].add(ref["identifier"])
                
                # Collect section references
                for ref in refs.get("sections", []):
                    if "number" in ref:
                        paragraph_refs["sections"].add(ref["number"])
                
                # Collect recital references
                for ref in refs.get("recitals", []):
                    if "number" in ref:
                        paragraph_refs["recitals"].add(ref["number"])
        
        return paragraph_refs
    
    def _is_reference_in_paragraphs(self, reference_id: int, ref_type: str, paragraph_refs: Dict[str, set]) -> bool:
        """Check if a reference is already present in paragraph-level references."""
        return reference_id in paragraph_refs.get(ref_type, set())
    
    def _is_annex_reference_in_paragraphs(self, identifier: str, paragraph_refs: Dict[str, set]) -> bool:
        """Check if an annex reference is already present in paragraph-level references."""
        return identifier in paragraph_refs.get("annexes", set())
    
    def _extract_annex_number(self, title: str) -> Optional[int]:
        """Extract annex number from title text."""
        # Try to match Roman numerals first
        roman_match = re.search(r'Annex\s+([IVX]+)', title, re.IGNORECASE)
        if roman_match:
            return fromRoman(roman_match.group(1))
        
        # Try to match Arabic numerals
        arabic_match = re.search(r'Annex\s+(\d+)', title, re.IGNORECASE)
        if arabic_match:
            return int(arabic_match.group(1))
        
        return None
    
    def _extract_annex_identifier(self, title: str) -> Optional[str]:
        """Extract annex identifier (Roman or Arabic) from title text."""
        match = re.search(r'Annex\s+([IVX]+|\d+)', title, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _extract_article_number(self, title: str) -> Optional[int]:
        """Extract article number from title text."""
        match = re.search(r'Article\s+(\d+)', title, re.IGNORECASE)
        return int(match.group(1)) if match else None
    
    def _extract_recital_number(self, title: str) -> Optional[int]:
        """Extract recital number from title text."""
        match = re.search(r'Recital\s+(\d+)', title, re.IGNORECASE)
        return int(match.group(1)) if match else None
    
    def _extract_number_from_url(self, url: str, url_type: str) -> Optional[int]:
        """Extract number from URL path."""
        pattern = rf'/{url_type}/(\d+)/?'
        match = re.search(pattern, url, re.IGNORECASE)
        return int(match.group(1)) if match else None
    
    def _remove_duplicate_references(self, references: List[Dict[str, Any]], ref_type: str) -> List[Dict[str, Any]]:
        """Remove duplicate references based on number or identifier."""
        seen = set()
        unique_refs = []
        
        for ref in references:
            key = str(ref.get('number') or ref.get('identifier'))

            if ref_type == 'articles':
                key = str(ref["number"])+"_"+str(ref.get("paragraph",""))+"_"+str(ref.get("points",""))

            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)
        
        return unique_refs


def main():
    """Command line interface for the scraper."""
    parser = argparse.ArgumentParser(
        description="Extract structured data from AI Act HTML content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file article2.html --output article2.json
  %(prog)s --url https://artificialintelligenceact.eu/article/2/
  %(prog)s --text "<html>...</html>" --pretty
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', '-f', help='HTML file to process')
    input_group.add_argument('--url', '-u', help='URL to scrape')
    input_group.add_argument('--text', '-t', help='HTML text content')
    
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    parser.add_argument('--pretty', action='store_true', help='Pretty print JSON output')
    parser.add_argument('--stats', action='store_true', help='Show extraction statistics')
    
    args = parser.parse_args()
    
    try:
        scraper = AIActScraper()
        
        # Extract data based on input method
        if args.file:
            result = scraper.scrape_from_file(args.file)
        elif args.url:
            result = scraper.scrape_from_url(args.url)
        else:  # args.text
            result = scraper.scrape_from_text(args.text)
        
        # Show statistics if requested
        if args.stats:
            content_type = result['metadata'].get('contentType', 'Article')
            
            if content_type == 'Article':
                stats = {
                    'paragraphs': len(result['paragraphs']),
                    'article_refs': len(result['references']['articles']),
                    'chapter_refs': len(result['references']['chapters']),
                    'annex_refs': len(result['references']['annexes']),
                    'recital_refs': len(result['references']['recitals'])
                }
                
                print("Article Extraction Statistics:", file=sys.stderr)
                print(f"  Paragraphs: {stats['paragraphs']}", file=sys.stderr)
            else:
                stats = {
                    'sections': len(result['sections']),
                    'paragraphs': len(result['paragraphs']),
                    'article_refs': len(result['references']['articles']),
                    'chapter_refs': len(result['references']['chapters']),
                    'annex_refs': len(result['references']['annexes']),
                    'recital_refs': len(result['references']['recitals'])
                }
                
                print("Annex Extraction Statistics:", file=sys.stderr)
                print(f"  Sections: {stats['sections']}", file=sys.stderr)
                print(f"  Paragraphs: {stats['paragraphs']}", file=sys.stderr)
            
            print(f"  Article References: {stats['article_refs']}", file=sys.stderr)
            print(f"  Chapter References: {stats['chapter_refs']}", file=sys.stderr)
            print(f"  Annex References: {stats['annex_refs']}", file=sys.stderr)
            print(f"  Recital References: {stats['recital_refs']}", file=sys.stderr)
            print("", file=sys.stderr)
        
        # Format JSON output
        json_kwargs = {'ensure_ascii': False}
        if args.pretty:
            json_kwargs.update({'indent': 2, 'sort_keys': True})
        
        json_output = json.dumps(result, **json_kwargs)
        
        # Write output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"Results saved to {args.output}", file=sys.stderr)
        else:
            print(json_output)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


# Example usage as a library:
"""
from ai_act_scraper import AIActScraper

# Initialize scraper
scraper = AIActScraper()

# Scrape Article from file
article_result = scraper.scrape_from_file('article2.html')

# Scrape Annex from file
annex_result = scraper.scrape_from_file('annex1.html')

# Scrape from URL
result = scraper.scrape_from_url('https://artificialintelligenceact.eu/article/2/')

# Scrape from HTML string
html_content = "<html>...</html>"
result = scraper.scrape_from_text(html_content)

# Access extracted data for Articles
if result['metadata']['contentType'] == 'Article':
    article_info = result['article']
    paragraphs = result['paragraphs']
    print(f"Article: {article_info['name']}")
    print(f"Paragraphs found: {len(paragraphs)}")

# Access extracted data for Annexes
elif result['metadata']['contentType'] == 'Annex':
    annex_info = result['annex']
    sections = result['sections']
    paragraphs = result['paragraphs']
    print(f"Annex: {annex_info['name']}")
    print(f"Sections found: {len(sections)}")
    print(f"Paragraphs found: {len(paragraphs)}")

# Common reference access
references = result['references']
print(f"Article references: {len(references['articles'])}")
print(f"Recital references: {len(references['recitals'])}")
"""