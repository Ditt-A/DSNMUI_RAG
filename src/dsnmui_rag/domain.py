from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

from .constants import STOPWORDS_ID, IMPORTANT_TERMS, DALIL_KEYWORDS, UMKM_DOMAIN_TERMS
from .utils import normalize_text, text_preview, count_arabic, clean_output_text, smart_truncate


def normalize_nomor(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    m = re.search(r"\d{1,4}", s)
    return m.group(0) if m else s


def infer_nomor_from_payload(p: Dict[str, Any]) -> str:
    direct = normalize_nomor(p.get("nomor"))
    if direct:
        return direct
    candidates = [p.get("title"), p.get("doc_id"), p.get("pdf_filename"), p.get("pdf_path"), p.get("chunk_id")]
    joined = " ".join(str(x or "") for x in candidates)
    patterns = [
        r"fatwa__([0-9]{1,4})(?:[-_]|\b)",
        r"fatwa[_\s-]+([0-9]{1,4})(?:[-_]|\b)",
        r"([0-9]{1,4})-dsn-mui",
        r"no\.?\s*([0-9]{1,4})",
        r"fatwa\s+(?:dsn[-\s]*mui\s*)?(?:no\.?\s*)?([0-9]{1,4})",
    ]
    for pat in patterns:
        m = re.search(pat, joined, flags=re.I)
        if m:
            return normalize_nomor(m.group(1))
    return ""


def infer_category_from_payload(p: Dict[str, Any]) -> str:
    cat = str(p.get("category") or "").strip()
    if cat:
        return cat
    text = " ".join(str(p.get(k) or "") for k in ["title", "doc_id", "pdf_filename", "chunk_id"]).lower()
    for c in ["fatwa", "pedoman_implementasi", "talimat"]:
        if c in text:
            return c
    return ""


def extract_requested_nomor(query: str) -> str:
    q = str(query or "")
    patterns = [
        r"fatwa\s+(?:dsn[-\s]*mui\s*)?(?:no\.?\s*)?([0-9]{1,4})",
        r"fatwa\s+dsn[-\s]*mui\s+no\.?\s*([0-9]{1,4})",
        r"no\.?\s*([0-9]{1,4})\s*(?:tentang|terkait|dalam|fatwa|dsn|mui)",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.I)
        if m:
            return normalize_nomor(m.group(1))
    return ""


def short_title(title: Any, max_len: int = 120) -> str:
    s = str(title or "").strip()
    s = re.sub(r"^[a-z_]+__", "", s)
    s = s.replace("__", " - ").replace("_", " ")
    s = re.sub(r"-0bxtl.*$", "", s)
    return text_preview(s, max_len)


def payload_text(p: Dict[str, Any]) -> str:
    return normalize_text(p.get("text_original") or p.get("text_for_embedding") or p.get("text") or "")


def query_keywords(query: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9\u0600-\u06FF]+", str(query).lower())
    clean: List[str] = []
    for t in toks:
        if len(t) < 3 or t in STOPWORDS_ID:
            continue
        if t not in clean:
            clean.append(t)
    clean = sorted(clean, key=lambda x: (x not in IMPORTANT_TERMS, -len(x)))
    return clean[:14]


def lexical_score_for_payload(query: str, p: Dict[str, Any]) -> float:
    text = (short_title(p.get("title"), 240) + "\n" + payload_text(p)).lower()
    kws = query_keywords(query)
    if not kws:
        return 0.0
    score = 0.0
    for kw in kws:
        cnt = text.count(kw.lower())
        if cnt:
            score += min(cnt, 5) * (2.0 if kw in IMPORTANT_TERMS else 1.0)
    qlow = str(query).lower()
    if any(x in qlow for x in ["definisi", "dimaksud", "apa definisi"]):
        if re.search(r"(yang\s+dimaksud\s+dengan|adalah|merupakan)", text):
            score += 4.0
    for term, boost in [("pemasaran", 4.0), ("wakalah", 4.0), ("ujrah", 3.0), ("ijarah", 3.0)]:
        if term in qlow and term in text:
            score += boost
    if "ju" in qlow and ("ju'alah" in text or "jualah" in text or "ju’alah" in text):
        score += 3.0
    return float(score)


def contains_dalil(text: str) -> bool:
    low = normalize_text(text).lower()
    return any(k in low for k in DALIL_KEYWORDS) or count_arabic(text) >= 20


def should_run_arabic_evidence(query: str) -> bool:
    low = normalize_text(query).lower()
    return any(k in low for k in ["arab", "hadis", "hadits", "dalil", "ayat", "q.s", "quran", "qur'an", "al-qur", "riwayat", "firman", "kaidah"])


def split_text_units(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    return [p.strip(" -•\t") for p in parts if len(p.strip()) > 3]


def extract_dalil_snippets(text: str, max_chars: int = 1200) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    units = split_text_units(text)
    chosen: list[str] = []
    for u in units:
        low = u.lower()
        if count_arabic(u) >= 8 or any(k in low for k in DALIL_KEYWORDS):
            chosen.append(u)
        if sum(len(x) for x in chosen) >= max_chars:
            break
    return smart_truncate(clean_output_text(normalize_text(" ".join(chosen))), max_chars)
