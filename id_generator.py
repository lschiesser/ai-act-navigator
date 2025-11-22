from roman import toRoman

ACT_ID = None # In case we want to refer to more specific Regulations e.g. f"EU-AI-ACT-2024-1689"


# -----------------------------------------------------------------------------
# ID-Schema
#  -----------------------------------------------------------------------------

def id_act() -> str:
    return ACT_ID

def id_article(art_num: str|int) -> str:
    return f"{ACT_ID}.Art.{art_num}" if ACT_ID else f"Art.{art_num}"

def id_article_paragraph(art_num: str|int, p_idx: int) -> str:
    return f"{id_article(art_num)}.{p_idx}"

def id_annex(an_num: int) -> str:
    return f"{ACT_ID}.Annex.{toRoman(an_num)}" if ACT_ID else f"Annex.{toRoman(an_num)}"

def id_annex_section(an_num: int, sec: str) -> str:
    return f"{id_annex(an_num)}.Sect.{sec}" if ACT_ID else f"Sect.{sec}"

def id_annex_paragraph(annex_identifier: str, p_number: str|int, section_key: str=None) -> str:
    if section_key is None:
        return f"{id_annex(annex_identifier)}.{p_number}"
    else:
        return f"{id_annex_section(annex_identifier, section_key)}.{p_number}"

def id_recital(rec_num: int|str) -> str:
    return f"{ACT_ID}.Rec.{rec_num}" if ACT_ID else f"Rec.{rec_num}"

def id_recital_paragraph(rec_num: int|str, p_idx: int) -> str:
    return f"{id_recital(rec_num)}.{p_idx}"

def id_chapter(ch_roman: str) -> str:
    return f"{ACT_ID}.Chapt{ch_roman.upper()}" if ACT_ID else f"Chapt{ch_roman.upper()}"

def id_chapter_section(ch_roman: str, sec: int) -> str:
    return f"{id_chapter(ch_roman)}.Sect-{sec}"