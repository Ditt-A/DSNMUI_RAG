from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional
import os


@dataclass
class RAGConfig:
    """Central config for the DSN-MUI UMKM RAG prototype.

    Defaults mirror the final notebook direction:
    - precomputed chunks/embeddings/BM25 are reused
    - dense BGE-M3 + BM25 + RRF + statistical reranker
    - Qwen3-VL can be used for visual dalil and text-only reasoning via adapter
    - optional chunk-aligned Qwen3-VL visual dalil quality gate
    """

    project_dir: Path = Path(os.getenv("DSNMUI_PROJECT_DIR", "dsnmui_rag_precompute_arabic_fix"))
    precomp_dir: Optional[Path] = None
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Required precomputed artifacts
    chunks_filename: str = "chunks.jsonl"
    embeddings_filename: str = "embeddings_bge_m3.npy"
    bm25_filename: str = "bm25_index.pkl"

    # Embedding/retrieval
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
    embed_max_length: int = int(os.getenv("EMBED_MAX_LENGTH", "8192"))
    use_fp16: bool = os.getenv("USE_FP16", "1").lower() in {"1", "true", "yes", "y"}
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "60"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "60"))
    hybrid_top_k: int = int(os.getenv("HYBRID_TOP_K", "36"))
    final_context_k: int = int(os.getenv("FINAL_CONTEXT_K", "4"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    debug_retrieval_steps: bool = os.getenv("DEBUG_RETRIEVAL_STEPS", "0").lower() in {"1", "true", "yes", "y"}

    # Statistical reranker
    reranker_type: str = "statistical_v12_v13_conservative_support"
    rerank_max_chars_per_passage: int = int(os.getenv("RERANK_MAX_CHARS_PER_PASSAGE", "1600"))
    direct_nomor_candidates_k: int = int(os.getenv("DIRECT_NOMOR_CANDIDATES_K", "18"))

    # Context/prompt/generation
    generation_schema_version: str = "v13_short_json"
    force_abstain_on_low_support: bool = True
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "520"))
    max_prompt_tokens: int = int(os.getenv("MAX_PROMPT_TOKENS", "6144"))
    max_context_chars_per_source: int = int(os.getenv("MAX_CONTEXT_CHARS_PER_SOURCE", "2200"))
    max_context_total_chars: int = int(os.getenv("MAX_CONTEXT_TOTAL_CHARS", "3600"))
    max_dalil_chars_per_source: int = int(os.getenv("MAX_DALIL_CHARS_PER_SOURCE", "1200"))

    # Visual dalil v16
    visual_dalil_enabled: bool = os.getenv("VISUAL_DALIL_ENABLED", "0").lower() in {"1", "true", "yes", "y"}
    visual_dalil_model_name: str = os.getenv("VISUAL_DALIL_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    visual_dalil_max_sources: int = int(os.getenv("VISUAL_DALIL_MAX_SOURCES", "2"))
    visual_dalil_max_crops_per_source: int = int(os.getenv("VISUAL_DALIL_MAX_CROPS_PER_SOURCE", "4"))
    visual_dalil_render_dpi: int = int(os.getenv("VISUAL_DALIL_RENDER_DPI", "300"))
    visual_dalil_max_new_tokens: int = int(os.getenv("VISUAL_DALIL_MAX_NEW_TOKENS", "420"))
    visual_dalil_temperature: float = float(os.getenv("VISUAL_DALIL_TEMPERATURE", "0.0"))
    visual_dalil_min_arabic_in_chunk: int = int(os.getenv("VISUAL_DALIL_MIN_ARABIC_IN_CHUNK", "8"))
    visual_dalil_min_arabic_count: int = int(os.getenv("VISUAL_DALIL_MIN_ARABIC_COUNT", "12"))
    visual_dalil_min_quality_score: float = float(os.getenv("VISUAL_DALIL_MIN_QUALITY_SCORE", "7.5"))
    visual_dalil_accept_low_confidence: bool = False
    visual_dalil_only_for_dalil_queries: bool = True
    visual_dalil_force_for_source_with_dalil: bool = True
    visual_dalil_require_arabic_in_chunk: bool = True
    visual_dalil_fallback_heuristic_clip: bool = True
    visual_dalil_enable_wider_same_page_fallback: bool = True
    visual_dalil_clip_padding_pt: float = 18.0
    visual_dalil_heuristic_clips: tuple[str, ...] = ("upper_middle", "middle", "lower_middle")
    visual_dalil_wider_clips: tuple[str, ...] = ("upper_70", "middle_80", "lower_75")

    def __post_init__(self) -> None:
        if self.precomp_dir is None:
            self.precomp_dir = self.project_dir / "precomputed"
        self.precomp_dir = Path(self.precomp_dir)

    @property
    def chunks_path(self) -> Path:
        return self.precomp_dir / self.chunks_filename

    @property
    def embeddings_path(self) -> Path:
        return self.precomp_dir / self.embeddings_filename

    @property
    def bm25_path(self) -> Path:
        return self.precomp_dir / self.bm25_filename

    @property
    def evidence_dir(self) -> Path:
        return self.precomp_dir / "arabic_evidence"

    @property
    def crop_dir(self) -> Path:
        return self.evidence_dir / "crops"

    @property
    def run_dir(self) -> Path:
        return self.precomp_dir / "rag_umkm_test_runs" / self.run_id

    def ensure_dirs(self) -> None:
        for path in [self.evidence_dir, self.crop_dir, self.run_dir]:
            path.mkdir(parents=True, exist_ok=True)
