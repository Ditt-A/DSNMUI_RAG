from __future__ import annotations
from typing import Any, Dict, List
import re
import numpy as np

from .config import RAGConfig
from .constants import STOPWORDS_ID, CORE_CONCEPT_TERMS, UNSUPPORTED_QUERY_TERMS, UMKM_DOMAIN_TERMS
from .domain import (
    payload_text, short_title, infer_nomor_from_payload, infer_category_from_payload,
    query_keywords, contains_dalil, extract_dalil_snippets, split_text_units,
)
from .utils import normalize_text, count_arabic, clean_output_text, smart_truncate


def query_core_concepts(query: str) -> List[str]:
    low = normalize_text(query).lower()
    hits = [t for t in CORE_CONCEPT_TERMS if t in low]
    toks = [t for t in query_keywords(query) if len(t) >= 6 and t not in STOPWORDS_ID]
    for t in toks:
        if t in UMKM_DOMAIN_TERMS and t not in hits:
            hits.append(t)
    return list(dict.fromkeys(hits))[:20]


def has_unsupported_specific_terms(query: str) -> List[str]:
    low = normalize_text(query).lower()
    return [t for t in UNSUPPORTED_QUERY_TERMS if t in low]


def support_score_for_results(query: str, results: List[Dict[str, Any]], config: RAGConfig) -> Dict[str, Any]:
    q_norm = normalize_text(query)
    q_terms = [t for t in query_keywords(q_norm) if t not in STOPWORDS_ID and len(t) >= 3]
    q_terms = list(dict.fromkeys(q_terms))
    if not q_terms or not results:
        return {
            "support_score": 0.0,
            "support_level": "very_low",
            "matched_terms": "",
            "missing_terms": ", ".join(q_terms[:30]),
            "core_concepts": "",
            "matched_core_concepts": "",
            "unsupported_specific_terms": "",
        }

    top_texts = []
    for r in results[: config.final_context_k]:
        p = r.get("payload", {})
        top_texts.append(" ".join([str(p.get("title") or ""), payload_text(p)]).lower())
    joined = " ".join(top_texts)

    matched = [t for t in q_terms if t.lower() in joined]
    missing = [t for t in q_terms if t not in matched]
    concepts = query_core_concepts(q_norm)
    matched_concepts = [c for c in concepts if c.lower() in joined]
    unsupported = has_unsupported_specific_terms(q_norm)
    unsupported_missing = [t for t in unsupported if t.lower() not in joined]

    term_cov = len(matched) / max(len(q_terms), 1)
    concept_cov = len(matched_concepts) / max(len(concepts), 1) if concepts else 0.0

    top_scores = []
    for r in results[: min(3, len(results))]:
        val = r.get("score_reranker") or r.get("statistical_score") or r.get("score_hybrid") or 0
        try:
            top_scores.append(float(val))
        except Exception:
            pass
    # Notebook capped the score signal to 0..1; keep the same conservative effect.
    score_signal = min(max(float(np.mean(top_scores)) if top_scores else 0.0, 0.0), 1.0)
    score = (0.45 * term_cov) + (0.40 * concept_cov) + (0.15 * score_signal)

    if unsupported_missing:
        score *= 0.45
    q_low = q_norm.lower()
    if ("dalil arab" in q_low or "teks arab" in q_low or "hadis arab" in q_low) and concept_cov < 0.35:
        score *= 0.65

    if score >= 0.58 and (not concepts or concept_cov >= 0.45) and not unsupported_missing:
        level = "high"
    elif score >= 0.34 and (not concepts or concept_cov >= 0.20):
        level = "medium"
    elif score >= 0.18:
        level = "low"
    else:
        level = "very_low"

    return {
        "support_score": round(float(score), 4),
        "support_level": level,
        "matched_terms": ", ".join(matched[:30]),
        "missing_terms": ", ".join(missing[:30]),
        "core_concepts": ", ".join(concepts[:20]),
        "matched_core_concepts": ", ".join(matched_concepts[:20]),
        "unsupported_specific_terms": ", ".join(unsupported_missing[:20]),
    }


def _candidate_windows(text: str, centers: List[int], max_chars: int) -> List[tuple[int, int]]:
    n = len(text)
    windows: List[tuple[int, int]] = []
    half = max(120, max_chars // 2)
    for c in centers[:50]:
        start = max(0, c - half)
        end = min(n, start + max_chars)
        start = max(0, end - max_chars)
        windows.append((start, end))
    if not windows:
        windows.append((0, min(n, max_chars)))
    return windows


def focused_snippet(text: str, query: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    low = text.lower()
    kws = query_keywords(query)
    centers: List[int] = []
    for kw in kws:
        start = 0
        while True:
            pos = low.find(kw.lower(), start)
            if pos < 0:
                break
            centers.append(pos)
            start = pos + max(1, len(kw))
    windows = _candidate_windows(text, centers, max_chars)
    best = None
    phrases = ["yang dimaksud dengan", "akad antara", "untuk pemasaran", "wakalah bil ujrah", "ijarah", "ju'alah", "bepergianlah", "hotel syariah"]
    for start, end in windows:
        seg = low[start:end]
        score = 0.0
        for kw in kws:
            cnt = seg.count(kw.lower())
            if cnt:
                score += cnt * (3.0 if kw in {"akad", "dalil", "hadis", "ayat", "ijarah", "wakalah", "ujrah"} else 1.0)
        for ph in phrases:
            if ph in seg:
                score += 4.0
        if best is None or score > best[0]:
            best = (score, start, end)
    _, start, end = best
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(text):
        snippet += " ..."
    return snippet


def build_umkm_context(item: Dict[str, Any], results: List[Dict[str, Any]], config: RAGConfig, include_arabic: bool = False) -> str:
    query = item.get("retrieval_query") or item.get("query") or ""
    parts: List[str] = []
    total = 0
    for i, r in enumerate(results[: config.final_context_k], start=1):
        p = r.get("payload", {})
        source_id = f"S{i}"
        text = focused_snippet(clean_output_text(payload_text(p)), query, max_chars=config.max_context_chars_per_source)
        total += len(text)
        if total > config.max_context_total_chars:
            text = text[: max(0, config.max_context_total_chars - (total - len(text)))]
        nomor = infer_nomor_from_payload(p) or "tidak-terbaca"
        category = infer_category_from_payload(p) or "tidak-terbaca"
        title = short_title(p.get("title"))
        page = p.get("page") or "tidak-terbaca"
        arabic_block = "Tidak ditampilkan karena pertanyaan tidak meminta Arab/hadis/ayat/OCR."
        if include_arabic:
            vd = r.get("visual_dalil") or {}
            raw = vd.get("arabic_text") or p.get("arabic_text") or ""
            arabic_block = raw[: config.max_dalil_chars_per_source] if raw else "Tidak tersedia teks Arab/OCR pada source ini."
        block = f"""
[{source_id}] {category} | Fatwa/No: {nomor} | Halaman: {page}
Judul: {title}
Bukti teks relevan:
{text}
Teks Arab/OCR jika relevan:
{arabic_block}
""".strip()
        parts.append(block)
        if total >= config.max_context_total_chars:
            break
    return "\n\n" + ("\n\n" + "-" * 72 + "\n\n").join(parts)


def build_umkm_generator_prompt(item: Dict[str, Any], context: str) -> str:
    query = item.get("query") or ""
    focus = item.get("expected_focus") or ""

    return f"""
Anda adalah asisten DSN-MUI untuk membantu pelaku UMKM memahami pilihan akad dan ketentuan syariah secara praktis.

Tugas utama:
1. Pahami dulu masalah bisnis user, bukan hanya mencocokkan satu kata kunci.
2. Jawab hanya berdasarkan KONTEKS DSN-MUI yang diberikan.
3. Jangan membuat fatwa, nomor fatwa, halaman, dalil, atau aturan baru di luar konteks.
4. Jika konteks tidak cukup relevan dengan kasus user, gunakan status "tidak_cukup".
5. Jika konteks cukup relevan, gunakan status "ada".
6. Jelaskan alasan dengan bahasa sederhana, natural, dan cocok untuk pelaku UMKM.
7. Jangan terlalu formal seperti menyalin teks fatwa.
8. Jangan menyimpulkan akad secara mutlak jika bentuk transaksinya belum jelas.
9. Jika ada beberapa kemungkinan akad, jelaskan syarat kapan akad tersebut relevan.
10. Wajib sertakan rujukan source seperti [S1], [S2] pada bagian jawaban dan poin.
11. Output hanya JSON valid. Jangan markdown. Jangan teks di luar JSON.

Gaya jawaban:
- Bahasa Indonesia natural, ringkas, dan mudah dipahami.
- Mulai dari jawaban praktis.
- Lanjutkan dengan alasan mengapa akad/ketentuan itu relevan.
- Tambahkan catatan batasan jika kasus user masih kurang detail.
- Hindari kalimat terlalu kaku seperti "konteks yang terambil memuat dasar".

Schema JSON wajib:
{{
  "status": "ada" atau "tidak_cukup",
  "jawab": "jawaban praktis 1-2 paragraf untuk UMKM, wajib mencantumkan [S1]/[S2] jika menyebut aturan",
  "poin": [
    "poin akad atau ketentuan relevan, singkat, dengan [Sx]",
    "poin berikutnya jika ada"
  ],
  "batasan": "catatan singkat tentang batasan jawaban atau kondisi transaksi yang perlu diperjelas"
}}

Aturan status:
- Gunakan "ada" hanya jika konteks benar-benar membahas akad/ketentuan yang relevan dengan kasus user.
- Gunakan "tidak_cukup" jika konteks membahas topik lain, terlalu umum, atau tidak cukup spesifik.
- Jangan memaksakan jawaban jika source yang tersedia tidak sesuai dengan masalah user.

Fokus tambahan:
{focus}

[KONTEKS DSN-MUI]
{context}

[PERTANYAAN USER]
{query}
""".strip()


def deterministic_sources(results: List[Dict[str, Any]], max_sources: int = 4) -> List[str]:
    lines = []
    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get("payload", {})
        lines.append(
            f"- [S{i}] {infer_category_from_payload(p)} No. {infer_nomor_from_payload(p) or 'tidak-terbaca'}, "
            f"{short_title(p.get('title'), 160)}, halaman {p.get('page') or 'tidak-terbaca'}."
        )
    return lines


def visual_dalil_line_from_result(result: Dict[str, Any], source_idx: int) -> str | None:
    vd = result.get("visual_dalil") or {}
    if not vd:
        return None
    p = result.get("payload", {})
    source = f"[S{source_idx}] {infer_category_from_payload(p)} No. {infer_nomor_from_payload(p)}, halaman {p.get('page')}"
    crop = vd.get("best_crop_image_path") or ""
    dtype = vd.get("dalil_type") or "unknown"
    conf = vd.get("confidence") or "medium"
    if vd.get("accepted") and vd.get("arabic_text"):
        terjemahan = extract_document_translation_for_dalil(
            payload=p,
            dalil_type=dtype,
            max_len=320,
        )

        terjemahan_line = (
            f"\n  Terjemahan dokumen: {terjemahan}"
            if terjemahan
            else ""
        )

        return (
            f"- {source}\n"
            f"  Jenis: {dtype}\n"
            f"  Teks Arab: {normalize_text(vd.get('arabic_text'))}"
            f"{terjemahan_line}"
        )
    if vd.get("status") == "visual_needs_verification":
        return (
            f"- {source}\n"
            f"  Ditemukan kandidat {dtype} pada gambar, tetapi teks Arab belum cukup presisi untuk ditampilkan sebagai kutipan resmi. "
            f"Verifikasi crop/halaman asli: {crop}"
        )
    if vd.get("status") in ["low_quality", "error"]:
        return f"- {source}\n  Ditemukan kandidat teks Arab pada gambar, tetapi kualitasnya belum cukup untuk dikutip. Verifikasi crop/halaman asli: {crop}"
    return None

def extract_document_translation_for_dalil(
    payload: Dict[str, Any],
    dalil_type: str = "",
    max_len: int = 320,
) -> str:
    """
    Mengambil terjemahan/penjelasan Indonesia dari teks dokumen,
    bukan menerjemahkan dengan model.

    Jika tidak ada terjemahan eksplisit/kalimat Indonesia yang relevan
    pada payload_text, return string kosong.
    """
    text = clean_output_text(payload_text(payload))
    if not text:
        return ""

    units = split_text_units(text)

    # Marker umum yang biasanya muncul pada terjemahan/penjelasan dalil
    common_markers = [
        "artinya",
        "maksudnya",
        "terjemah",
        "terjemahan",
        "yang dimaksud",
    ]

    # Marker khusus kaidah muamalah
    kaidah_markers = [
        "pada dasarnya",
        "semua bentuk muamalah",
        "muamalah boleh",
        "boleh dilakukan",
        "kecuali ada dalil",
        "mengharamkannya",
        "dilarang",
    ]

    # Marker hadis umum
    hadis_markers = [
        "perdamaian",
        "orang muslim",
        "kaum muslimin",
        "syarat-syarat",
        "menghalalkan yang haram",
        "mengharamkan yang halal",
    ]

    # Marker ayat umum
    ayat_markers = [
        "hai orang",
        "wahai orang",
        "allah",
        "jual beli",
        "riba",
        "halal",
        "haram",
    ]

    dtype = (dalil_type or "").lower()

    markers = list(common_markers)

    if dtype == "kaidah":
        markers += kaidah_markers
    elif dtype == "hadis":
        markers += hadis_markers
    elif dtype == "ayat":
        markers += ayat_markers
    else:
        markers += kaidah_markers + hadis_markers + ayat_markers

    candidates = []

    for idx, unit in enumerate(units):
        u = normalize_text(unit)
        low = u.lower()

        if len(u) < 20:
            continue

        # Hindari unit yang dominan Arab, karena yang dicari terjemahan Indonesia
        arabic_count = count_arabic(u)
        if arabic_count > 20:
            continue

        score = 0
        for marker in markers:
            if marker in low:
                score += 3

        # Prioritaskan kalimat Indonesia yang natural
        if any(k in low for k in ["adalah", "boleh", "tidak boleh", "kecuali", "harus", "dapat"]):
            score += 1

        if score > 0:
            candidates.append((score, idx, u))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, idx, first = candidates[0]

    # Ambil 1-2 kalimat sekitar kandidat supaya tidak kepotong
    chosen = [first]

    if idx + 1 < len(units):
        nxt = normalize_text(units[idx + 1])
        if 20 <= len(nxt) <= 220 and count_arabic(nxt) <= 10:
            chosen.append(nxt)

    out = " ".join(chosen)
    return smart_truncate(out, max_len)

def deterministic_dalil_lines(results: List[Dict[str, Any]], item: Dict[str, Any], config: RAGConfig, max_sources: int = 3, allow: bool = True) -> List[str]:
    if not allow:
        return []
    lines: List[str] = []
    query = item.get("retrieval_query") or item.get("query") or ""
    q_terms = set(query_keywords(query))

    seen_dalil_keys = set()

    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get("payload", {}) or {}
        vd = r.get("visual_dalil") or {}

        dalil_key = (
            infer_category_from_payload(p),
            infer_nomor_from_payload(p),
            p.get("page"),
            vd.get("dalil_type"),
            normalize_text(vd.get("arabic_text") or "")[:80],
        )

        if dalil_key in seen_dalil_keys:
            continue

        line = visual_dalil_line_from_result(r, i)
        if line:
            seen_dalil_keys.add(dalil_key)
            lines.append(line)

    if lines:
        return lines

    any_visual_not_dalil = any((r.get("visual_dalil") or {}).get("status") == "arabic_detected_not_dalil" for r in results[:max_sources])
    if any_visual_not_dalil:
        return ["- Terdeteksi teks Arab pada source visual, tetapi tidak diklasifikasikan sebagai ayat/hadis/kaidah sehingga tidak ditampilkan sebagai dalil."]

    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get("payload", {})
        doc_text = payload_text(p)
        dalil = extract_dalil_snippets(doc_text, max_chars=config.max_dalil_chars_per_source)
        if not dalil:
            continue
        combined = (str(p.get("title") or "") + " " + doc_text).lower()
        overlap = sum(1 for t in q_terms if t in combined and t not in STOPWORDS_ID)
        if overlap < 2:
            continue
        lines.append(f"- [S{i}] {dalil}")
    return lines


def extract_concepts_from_results(results: List[Dict[str, Any]], query: str = "", limit: int = 6) -> List[str]:
    joined = (query + " " + " ".join(short_title((r.get("payload") or {}).get("title"), 200) + " " + clean_output_text(payload_text(r.get("payload", {})))[:1200] for r in results)).lower()
    concepts = [c for c in CORE_CONCEPT_TERMS if c in joined]
    return list(dict.fromkeys(concepts))[:limit]


def collect_evidence_points(item: Dict[str, Any], results: List[Dict[str, Any]], max_sources: int = 3) -> List[str]:
    query = item.get("query") or ""
    points: List[str] = []
    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get("payload", {})
        text = focused_snippet(clean_output_text(payload_text(p)), query, max_chars=1200)
        if text:
            points.append(f"- [S{i}] {text}")
    return points


def summarize_evidence_point(point: str, max_len: int = 260) -> str:
    point = re.sub(r"^- \[S\d+\]\s*", "", normalize_text(point))
    point = re.sub(r"\s+", " ", point).strip()
    units = split_text_units(point)
    chosen: List[str] = []
    for u in units:
        if len(u) < 20:
            continue
        chosen.append(u)
        if sum(len(x) for x in chosen) >= max_len:
            break
    text = " ".join(chosen) if chosen else point[:max_len]
    return text[:max_len].strip()
