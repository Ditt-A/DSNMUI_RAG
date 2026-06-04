from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import gc, json, re

from .config import RAGConfig
from .constants import V10_BAD_TERMS
from .context import (
    deterministic_sources, deterministic_dalil_lines, extract_concepts_from_results,
    collect_evidence_points, summarize_evidence_point,
)
from .domain import query_keywords
from .utils import normalize_text, extract_json_object, strip_code_fence, clean_output_text, smart_truncate


def clean_json_value(v: Any) -> Any:
    if isinstance(v, str):
        v = strip_code_fence(v)
        v = re.sub(r"\s+", " ", v).strip()
        return v
    if isinstance(v, list):
        return [clean_json_value(x) for x in v if x not in [None, ""]]
    if isinstance(v, dict):
        return {k: clean_json_value(x) for k, x in v.items()}
    return v


def normalize_structured_obj(obj: Dict[str, Any] | None, support_info: Dict[str, Any]) -> Dict[str, Any]:
    obj = obj or {}
    raw_status = str(obj.get("status", "")).lower().strip()
    if raw_status in ["ada", "ditemukan", "found", "supported"]:
        status = "ditemukan"
    elif raw_status in ["tidak_cukup", "tidak_cukup_konteks", "tidak cukup konteks", "no_context", "not_enough_context"]:
        status = "tidak_cukup_konteks"
    else:
        status = "ditemukan" if support_info.get("support_level") in ["high", "medium"] else "tidak_cukup_konteks"
    jawab = obj.get("jawab", obj.get("jawaban_praktis", ""))
    poin = obj.get("poin", obj.get("akad_ketentuan_relevan", []))
    batasan = obj.get("batasan", obj.get("catatan_batasan", ""))
    out = {
        "status": status,
        "jawaban_praktis": clean_json_value(jawab),
        "akad_ketentuan_relevan": clean_json_value(poin),
        "catatan_batasan": clean_json_value(batasan),
    }
    if isinstance(out["akad_ketentuan_relevan"], str):
        out["akad_ketentuan_relevan"] = [out["akad_ketentuan_relevan"]] if out["akad_ketentuan_relevan"] else []
    if not isinstance(out["akad_ketentuan_relevan"], list):
        out["akad_ketentuan_relevan"] = []
    return out


def structured_obj_is_bad(obj: Optional[Dict[str, Any]], raw: str = "") -> bool:
    if not isinstance(obj, dict):
        return True
    has_short = all(k in obj for k in ["status", "jawab", "poin", "batasan"])
    has_old = all(k in obj for k in ["status", "jawaban_praktis", "akad_ketentuan_relevan", "catatan_batasan"])
    if not (has_short or has_old):
        return True
    combined = json.dumps(obj, ensure_ascii=False) + " " + (raw or "")
    bad_pattern = r"[\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0400-\u04ff]|chunk_id|reranker_score|pdf_path|assistant\n|user\n"
    if re.search(bad_pattern, combined, flags=re.I):
        return True
    low = combined.lower()
    return any(t in low for t in V10_BAD_TERMS)


def normalize_answer_domain_terms(answer: str, query: str = "") -> str:
    # Keep minimal; notebook used it for domain wording cleanup. Safe hook for future rules.
    return normalize_text(answer)


def render_structured_answer(obj: Dict[str, Any], item: Dict[str, Any], results: List[Dict[str, Any]], support_info: Dict[str, Any], config: RAGConfig, mode: str = "json_generated") -> str:
    obj = normalize_structured_obj(obj, support_info)
    is_found = obj.get("status") == "ditemukan"
    status_text = "Ditemukan" if is_found else "Tidak cukup konteks"
    jawaban = obj.get("jawaban_praktis") or "Dokumen yang terambil belum cukup kuat untuk menjawab secara aman."
    akad_list = obj.get("akad_ketentuan_relevan") or ["Tidak dapat ditentukan dari konteks yang terambil."]
    akad_block = "\n".join(f"- {x}" for x in akad_list[:5])
    if is_found:
        dalil_lines = deterministic_dalil_lines(results, item, config, max_sources=3, allow=True)
        dalil_block = "\n".join(dalil_lines) if dalil_lines else "- Tidak ditemukan dalil/kaidah eksplisit dalam konteks yang terambil."
    else:
        dalil_block = "- Tidak ditampilkan karena konteks tidak cukup mendukung jawaban."
    sources = deterministic_sources(results, max_sources=4) if is_found else deterministic_sources(results, max_sources=2)
    source_block = "\n".join(sources) if sources else "- Tidak ada sumber yang cukup kuat."
    catatan = obj.get("catatan_batasan") or "Jawaban dibatasi pada corpus DSN-MUI precomputed dan konteks yang terambil."
    answer = (
        f"Status Jawaban:\n- {status_text}\n\n"
        f"Jawaban Praktis untuk UMKM:\n- {jawaban}\n\n"
        f"Akad/Ketentuan yang Relevan:\n{akad_block}\n\n"
        f"Dalil/Kaidah dari Dokumen:\n{dalil_block}\n\n"
        f"Sumber Dokumen:\n{source_block}\n\n"
        f"Catatan Batasan:\n- {catatan}"
    )
    if mode != "json_generated":
        answer += f"\n- Mode jawaban: {mode}."
    return normalize_answer_domain_terms(answer, query=item.get("query", ""))


def build_extractive_safe_answer(item: Dict[str, Any], results: List[Dict[str, Any]], support_info: Dict[str, Any], config: RAGConfig, reason: str = "") -> str:
    support_level = support_info.get("support_level", "unknown")
    if not results or support_level in ["very_low", "low"]:
        obj = {
            "status": "tidak_cukup",
            "jawab": "Dokumen DSN-MUI yang terambil belum cukup spesifik untuk menjawab pertanyaan ini secara aman.",
            "poin": ["Tidak dapat ditentukan dari konteks yang terambil."],
            "batasan": "Sistem hanya memakai corpus DSN-MUI precomputed; jawaban tidak dipaksakan jika konteks tidak cukup.",
        }
        return render_structured_answer(obj, item, results, support_info, config, mode="NO_CONTEXT")

    concepts = extract_concepts_from_results(results, query=item.get("query", ""), limit=6)
    evidence_points = collect_evidence_points(item, results, max_sources=3)
    ev1 = summarize_evidence_point(evidence_points[0], max_len=300) if evidence_points else ""
    ev2 = summarize_evidence_point(evidence_points[1], max_len=220) if len(evidence_points) > 1 else ""
    focus = normalize_text(item.get("expected_focus") or "")
    practical = "Konteks DSN-MUI yang terambil memuat dasar yang relevan untuk menjawab kasus ini."
    if concepts:
        practical += f" Konsep yang paling terkait adalah {', '.join(concepts[:4])}."
    if ev1:
        practical += f" Bukti utama dari dokumen: {ev1}"
    if ev2:
        practical += f" Bukti tambahan: {ev2}"
    obj = {
        "status": "ada",
        "jawab": smart_truncate(practical, 2200),
        "poin": concepts[:5] or [focus or "Gunakan bukti ekstraktif dari sumber terambil."],
        "batasan": "Jawaban ini fallback ekstraktif karena generator gagal menghasilkan JSON valid; review manual tetap disarankan.",
    }
    return render_structured_answer(obj, item, results, support_info, config, mode=f"CONTEXT_FOUND_GENERATION_FAILED:{reason or 'fallback_extractive_v13'}")


def answer_query_umkm_safely(
    item: Dict[str, Any],
    results: List[Dict[str, Any]],
    prompt: str,
    support_info: Dict[str, Any],
    config: RAGConfig,
    generator: Optional[Any] = None,
) -> tuple[str, Dict[str, Any]]:
    support_level = support_info.get("support_level")
    if config.force_abstain_on_low_support and support_level == "very_low":
        ans = build_extractive_safe_answer(item, results, support_info, config, reason="very_low_support")
        return ans, {"generation_mode": "NO_CONTEXT", "raw_generation": "", "json_parse_ok": False, "schema_version": config.generation_schema_version}

    if generator is None or generator.model is None:
        ans = build_extractive_safe_answer(item, results, support_info, config, reason="generator_not_loaded")
        return ans, {"generation_mode": "extractive_fallback_generator_not_loaded", "raw_generation": "", "json_parse_ok": False, "schema_version": config.generation_schema_version}

    obj, raw = generator.generate_structured_json_once(prompt, max_new_tokens=config.max_new_tokens)
    if not structured_obj_is_bad(obj, raw):
        obj_norm = normalize_structured_obj(obj, support_info)
        if support_level in ["high", "medium"] and obj_norm.get("status") == "tidak_cukup_konteks":
            retry_prompt = prompt + '\n\nKonteks dinilai cukup oleh retriever. Jika ada aturan relevan, gunakan status "ada". Output tetap JSON valid schema pendek.'
            obj2, raw2 = generator.generate_structured_json_once(retry_prompt, max_new_tokens=min(config.max_new_tokens, 420))
            if not structured_obj_is_bad(obj2, raw2):
                obj2_norm = normalize_structured_obj(obj2, support_info)
                if obj2_norm.get("status") == "ditemukan":
                    ans = render_structured_answer(obj2_norm, item, results, support_info, config, mode="json_generated_retry")
                    return ans, {"generation_mode": "json_generated_retry", "raw_generation": raw2, "json_parse_ok": True, "schema_version": config.generation_schema_version}
        ans = render_structured_answer(obj_norm, item, results, support_info, config, mode="json_generated")
        return ans, {"generation_mode": "json_generated", "raw_generation": raw, "json_parse_ok": True, "schema_version": config.generation_schema_version}

    retry_prompt = prompt + '\n\nOutput hanya JSON. Contoh format: {"status":"ada","jawab":"... [S1]","poin":["... [S1]"],"batasan":"..."}'
    obj2, raw2 = generator.generate_structured_json_once(retry_prompt, max_new_tokens=min(config.max_new_tokens, 420))
    if not structured_obj_is_bad(obj2, raw2):
        obj2_norm = normalize_structured_obj(obj2, support_info)
        ans = render_structured_answer(obj2_norm, item, results, support_info, config, mode="json_generated_retry")
        return ans, {"generation_mode": "json_generated_retry", "raw_generation": raw2, "json_parse_ok": True, "schema_version": config.generation_schema_version}

    ans = build_extractive_safe_answer(item, results, support_info, config, reason="json_invalid_v13")
    return ans, {"generation_mode": "extractive_fallback_json_invalid_v13", "raw_generation": raw2 or raw, "json_parse_ok": False, "schema_version": config.generation_schema_version}
