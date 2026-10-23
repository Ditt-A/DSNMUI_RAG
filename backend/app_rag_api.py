from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dsnmui_rag import RAGConfig, DSNMUIRAGPipeline
from dsnmui_rag.domain import infer_category_from_payload, infer_nomor_from_payload, short_title, payload_text
from dsnmui_rag.utils import clean_output_text, format_answer_markdown, text_preview


API_TITLE = "DSN-MUI UMKM RAG API"
API_VERSION = "1.0.0"

# Kosongkan RAG_API_KEY kalau endpoint hanya untuk local testing.
# Kalau expose via ngrok, set RAG_API_KEY agar endpoint tidak terbuka bebas.
API_KEY = os.getenv("RAG_API_KEY", "").strip()
LOAD_PIPELINE = os.getenv("RAG_LOAD_PIPELINE", "0").lower() in {"1", "true", "yes", "y"}
PRECOMP_DIR = os.getenv("PRECOMP_DIR", "dsnmui_rag_precompute_arabic_fix/precomputed")
FINAL_CONTEXT_K = int(os.getenv("FINAL_CONTEXT_K", "4"))
PRELOAD_BGE = os.getenv("PRELOAD_BGE", "0").lower() in {"1", "true", "yes", "y"}
PRELOAD_VL = os.getenv("PRELOAD_VL", "0").lower() in {"1", "true", "yes", "y"}

# Untuk demo FE/BE lintas domain. Untuk production, ganti dengan domain FE/BE temanmu.
CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [x.strip() for x in CORS_ORIGINS_RAW.split(",") if x.strip()]

app = FastAPI(title=API_TITLE, version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe: DSNMUIRAGPipeline | None = None
request_lock = threading.Lock()


class AskRequest(BaseModel):
    query: str = Field(..., description="Pertanyaan user/UMKM.")
    expected_focus: str = Field("", description="Catatan fokus opsional untuk retrieval/generation.")
    category_filter: Optional[str] = Field(None, description="fatwa | pedoman_implementasi | talimat | unknown | null")
    use_visual_dalil: bool = Field(False, description="Aktifkan Qwen3-VL untuk membaca crop dalil Arab dari PDF.")
    use_visual_as_generator: bool = Field(False, description="Gunakan Qwen3-VL yang sama untuk reasoning final text-only.")
    include_debug: bool = Field(False, description="Sertakan context/prompt/raw metadata untuk debugging.")


class AskResponse(BaseModel):
    answer: str
    answer_markdown: str
    support_info: dict[str, Any]
    generation_meta: dict[str, Any]
    timing: dict[str, Any]
    sources: list[dict[str, Any]]
    elapsed_seconds: float
    debug: Optional[dict[str, Any]] = None


def _check_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")


def _ensure_pipe() -> DSNMUIRAGPipeline:
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline belum siap. Aktifkan RAG_LOAD_PIPELINE=1 dan siapkan artifact precomputed.")
    return pipe


def _build_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for i, r in enumerate(results, start=1):
        p = r.get("payload", {}) or {}
        vd = r.get("visual_dalil") or {}
        source_text = clean_output_text(payload_text(p))
        sources.append({
            "source": f"S{i}",
            "rank_final": r.get("rank_final"),
            "category": infer_category_from_payload(p),
            "nomor": infer_nomor_from_payload(p),
            "title": short_title(p.get("title"), 120),
            "page": p.get("page"),
            "score_reranker": r.get("score_reranker"),
            "score_vector": r.get("score_vector"),
            "score_bm25": r.get("score_bm25"),
            "visual_dalil_status": vd.get("status"),
            "dalil_type": vd.get("dalil_type"),
            "arabic_text": vd.get("arabic_text") or "",
            "confidence": vd.get("confidence"),
            "crop": vd.get("best_crop_image_path"),
            "preview": text_preview(source_text, 800),
        })
    return sources


@app.on_event("startup")
def startup_event() -> None:
    global pipe
    if not LOAD_PIPELINE:
        print("[RAG API] Preview mode: pipeline dan model tidak dimuat (RAG_LOAD_PIPELINE=0)")
        return

    cfg = RAGConfig(precomp_dir=Path(PRECOMP_DIR), final_context_k=FINAL_CONTEXT_K)
    pipe = DSNMUIRAGPipeline(cfg)

    if PRELOAD_BGE:
        pipe.retriever.load_embedder()
        pipe.retriever.embed_query("warmup query")

    if PRELOAD_VL:
        pipe.visual.load()

    print("[RAG API] Ready")
    print("[RAG API] PRECOMP_DIR=", PRECOMP_DIR)
    print("[RAG API] PRELOAD_BGE=", PRELOAD_BGE)
    print("[RAG API] PRELOAD_VL=", PRELOAD_VL)
    print("[RAG API] API_KEY enabled=", bool(API_KEY))


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "ask": "/rag/ask",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    p = pipe
    return {
        "status": "healthy" if p is not None else ("starting" if LOAD_PIPELINE else "disabled"),
        "pipeline_enabled": LOAD_PIPELINE,
        "pipeline_loaded": p is not None,
        "precomp_dir": PRECOMP_DIR,
        "preload_bge": PRELOAD_BGE,
        "preload_vl": PRELOAD_VL,
        "vl_loaded": bool(p and p.visual.model is not None and p.visual.processor is not None),
        "api_key_enabled": bool(API_KEY),
    }


@app.get("/summary")
def summary(x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _check_api_key(x_api_key)
    p = _ensure_pipe()
    return p.summary()


@app.post("/warmup/vl")
def warmup_vl(x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _check_api_key(x_api_key)
    p = _ensure_pipe()
    t0 = time.time()
    with request_lock:
        p.visual.load()
    return {
        "status": "ok",
        "message": "Qwen3-VL ready",
        "elapsed_seconds": round(time.time() - t0, 3),
    }


@app.post("/rag/ask", response_model=AskResponse)
def rag_ask(req: AskRequest, x_api_key: Optional[str] = Header(default=None)) -> AskResponse:
    _check_api_key(x_api_key)
    p = _ensure_pipe()

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query kosong")

    t0 = time.time()
    try:
        # Heavy model inference should be serialized to avoid VRAM contention.
        with request_lock:
            ans = p.answer(
                query,
                expected_focus=req.expected_focus,
                category_filter=req.category_filter,
                use_visual_dalil=req.use_visual_dalil,
                use_visual_as_generator=req.use_visual_as_generator,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG inference failed: {type(e).__name__}: {e}") from e

    answer_text = clean_output_text(ans.answer)
    answer_markdown = format_answer_markdown(ans.answer)
    debug = None
    if req.include_debug:
        debug = {
            "context": clean_output_text(ans.context),
            "prompt": clean_output_text(ans.prompt),
            "item": ans.item,
        }

    return AskResponse(
        answer=answer_text,
        answer_markdown=answer_markdown,
        support_info=ans.support_info,
        generation_meta=ans.generation_meta,
        timing=ans.timing,
        sources=_build_sources(ans.results),
        elapsed_seconds=round(time.time() - t0, 3),
        debug=debug,
    )


# Alias sederhana agar test lama POST /ask tetap bisa dipakai.
@app.post("/ask", response_model=AskResponse)
def ask_alias(req: AskRequest, x_api_key: Optional[str] = Header(default=None)) -> AskResponse:
    return rag_ask(req, x_api_key=x_api_key)
