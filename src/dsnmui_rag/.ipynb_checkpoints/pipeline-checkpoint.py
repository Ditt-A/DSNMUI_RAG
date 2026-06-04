from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import RAGConfig
from .store import PrecomputedStore
from .retrieval import HybridRetriever
from .context import build_umkm_context, build_umkm_generator_prompt, support_score_for_results
from .generation import answer_query_umkm_safely
from .visual_dalil import VisualDalilExtractor


@dataclass
class RAGAnswer:
    answer: str
    item: Dict[str, Any]
    results: list[dict[str, Any]]
    context: str
    prompt: str
    support_info: Dict[str, Any]
    timing: Dict[str, Any]
    generation_meta: Dict[str, Any]


class VisualTextGeneratorAdapter:
    """
    Adapter agar Qwen3-VL bisa dipakai sebagai generator final
    dengan interface yang sama seperti QwenJSONGenerator.

    answer_query_umkm_safely() membutuhkan method:
    - generate_structured_json_once(prompt, max_new_tokens=None)
    """

    def __init__(self, visual: VisualDalilExtractor):
        self.visual = visual
        self.model = visual.model

    def load(self) -> None:
        self.visual.load()
        self.model = self.visual.model

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        self.load()
        return self.visual.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
        )

    def generate_structured_json_once(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ):
        """
        Dibuat kompatibel dengan QwenJSONGenerator.
        Return wajib: (obj, raw)
        """
        from .utils import (
            strip_code_fence,
            extract_first_json_object_text,
            extract_json_object,
        )
        import json

        raw = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
        )

        raw_clean = strip_code_fence(raw or "").strip()

        obj = None

        # Coba parse langsung
        try:
            parsed = json.loads(raw_clean)
            if isinstance(parsed, dict):
                obj = parsed
        except Exception:
            pass

        # Coba ambil JSON object pertama
        if obj is None:
            try:
                jtxt = extract_first_json_object_text(raw_clean)
                if jtxt:
                    parsed = json.loads(jtxt)
                    if isinstance(parsed, dict):
                        obj = parsed
            except Exception:
                pass

        # Fallback extractor JSON existing
        if obj is None:
            try:
                extracted = extract_json_object(raw_clean)
                if isinstance(extracted, dict):
                    obj = extracted
            except Exception:
                pass

        if obj is None:
            obj = {}

        return obj, raw
    
class DSNMUIRAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.config.ensure_dirs()
        self.store = PrecomputedStore.load(self.config)
        self.retriever = HybridRetriever(self.store, self.config)
        self.visual = VisualDalilExtractor(self.config)

    def answer(
    self,
    query: str,
    *,
    domain: str = "interactive",
    expected_focus: str = "",
    category_filter: Optional[str] = None,
    use_visual_dalil: Optional[bool] = None,
    use_visual_as_generator: bool = False,
) -> RAGAnswer:
        item = {
            "id": "interactive",
            "domain": domain,
            "query": query,
            "retrieval_query": query,
            "expected_focus": expected_focus,
            "needs_dalil": any(k in query.lower() for k in ["dalil", "hadis", "hadits", "ayat", "arab", "kaidah"]),
        }
        results, timing = self.retriever.retrieve(query, category_filter=category_filter)
        old_visual = self.config.visual_dalil_enabled
        if use_visual_dalil is not None:
            self.config.visual_dalil_enabled = bool(use_visual_dalil)
        try:
            if self.config.visual_dalil_enabled:
                results = self.visual.enrich_results(item, results)
        finally:
            if use_visual_dalil is not None:
                self.config.visual_dalil_enabled = old_visual

        support_info = support_score_for_results(query, results, self.config)
        include_arabic = item["needs_dalil"] or any((r.get("visual_dalil") or {}).get("arabic_text") for r in results)
        context = build_umkm_context(item, results, self.config, include_arabic=include_arabic)
        prompt = build_umkm_generator_prompt(item, context)

        gen_obj = None

        if use_visual_as_generator:
            # Pakai Qwen3-VL yang sama untuk reasoning final.
            # Tidak load Qwen3-4B.
            self.visual.load()
            gen_obj = VisualTextGeneratorAdapter(self.visual)

        answer, generation_meta = answer_query_umkm_safely(
            item,
            results,
            prompt,
            support_info,
            self.config,
            generator=gen_obj,
        )
        return RAGAnswer(
            answer=answer,
            item=item,
            results=results,
            context=context,
            prompt=prompt,
            support_info=support_info,
            timing=timing,
            generation_meta=generation_meta,
        )

    def retrieve_only(self, query: str, category_filter: Optional[str] = None):
        return self.retriever.retrieve(query, category_filter=category_filter)

    def summary(self) -> dict[str, Any]:
        return self.store.summary()
