from __future__ import annotations
import json, re, time
from pathlib import Path
from typing import Any, Dict, List
import regex as reg

AR_RE = re.compile(r"[\u0600-\u06FF]")
PUA_RE = re.compile(r"[\ue000-\uf8ff]")
TOKEN_PATTERN = reg.compile(r"[\p{L}\p{N}_]+")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    if not text:
        return ""
    text = str(text).replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text(text).lower())


def count_arabic(text: str) -> int:
    return len(AR_RE.findall(text or ""))


def count_pua(text: str) -> int:
    return len(PUA_RE.findall(text or ""))


def arabic_quality_score(text: str) -> Dict[str, Any]:
    text = normalize_text(text or "")
    n = max(len(text), 1)
    arabic_count = count_arabic(text)
    pua_count = count_pua(text)
    latin_count = len(re.findall(r"[A-Za-z]", text))
    digit_count = len(re.findall(r"[0-9]", text))
    arabic_ratio = arabic_count / n
    pua_ratio = pua_count / n
    latin_ratio = latin_count / n
    score = arabic_count - 2.5 * pua_count - 0.5 * latin_count - 0.2 * digit_count
    if arabic_count < 8:
        label = "none"
    elif pua_count > 0 or pua_ratio > 0.01:
        label = "bad_font_encoded"
    elif arabic_ratio < 0.25:
        label = "mixed_or_noisy"
    else:
        label = "usable"
    return {
        "label": label,
        "score": float(score),
        "length": len(text),
        "arabic_count": int(arabic_count),
        "pua_count": int(pua_count),
        "latin_count": int(latin_count),
        "digit_count": int(digit_count),
        "arabic_ratio": float(arabic_ratio),
        "pua_ratio": float(pua_ratio),
        "latin_ratio": float(latin_ratio),
        "newline_count": int(text.count("\n")),
    }


def safe_name(s: Any, max_len: int = 120) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s)).strip("_")
    return s[:max_len] or "item"


def text_preview(text: Any, n: int = 350) -> str:
    return normalize_text(text)[:n]


def now_ms() -> float:
    return time.perf_counter() * 1000


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object_text(raw: str) -> str:
    raw = strip_code_fence(raw or "")
    start = raw.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]
    return ""


def extract_json_object(raw: str) -> Dict[str, Any] | None:
    txt = extract_first_json_object_text(raw)
    if not txt:
        return None
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
    
import re

def clean_output_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    replacements = {
        "[OCR_VISUAL_TEXT]": "",
        "[VISUAL_OCR]": "",
        "[OCR_TEXT]": "",
        "[ARABIC_OCR]": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Samakan line ending
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    for line in text.split("\n"):
        # Rapikan spasi berlebih per baris, tapi jangan hapus line break
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Hapus newline berlebih tapi tetap sisakan struktur
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Rapikan spasi sebelum tanda baca
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    
    text = re.sub(r"Crop:\s*[^\n]+", "", text)

    return text.strip()


def smart_truncate(text: str, max_chars: int = 1200) -> str:
    if not isinstance(text, str):
        return ""

    text = clean_output_text(text)

    if len(text) <= max_chars:
        return text

    cut = text.rfind(" ", 0, max_chars)

    if cut < int(max_chars * 0.7):
        cut = max_chars

    return text[:cut].rstrip() + " ..."

def format_answer_markdown(text: str) -> str:
    text = clean_output_text(text)

    # Pecah bagian-bagian utama biar rapi
    sections = [
        "Status Jawaban:",
        "Ditemukan Jawaban Praktis untuk UMKM:",
        "Konteks DSN-MUI yang terambil memuat dasar yang relevan untuk menjawab kasus ini.",
        "Bukti utama dari dokumen:",
        "Bukti tambahan:",
        "Akad/Ketentuan yang Relevan:",
        "Dalil/Kaidah dari Dokumen:",
        "Sumber Dokumen:",
        "Catatan Batasan:",
        "Mode jawaban:",
    ]

    for sec in sections:
        text = text.replace(sec, f"\n\n**{sec}**\n")

    # Rapikan bullet dash
    text = re.sub(r"\n\s*-\s+", "\n- ", text)

    # Tambah line break sebelum source [S1], [S2], dst
    text = re.sub(r"\s(\[S\d+\])", r"\n\1", text)

    return text.strip()