from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np

from .config import RAGConfig
from .constants import STAT_STOPWORDS, STAT_CONCEPT_SYNONYMS, TITLE_BOOST_TERMS
from .domain import (
    infer_category_from_payload, infer_nomor_from_payload, extract_requested_nomor,
    lexical_score_for_payload, payload_text, short_title, query_keywords,
)
from .store import PrecomputedStore
from .utils import normalize_text, tokenize_for_bm25, now_ms


class HybridRetriever:
    def __init__(self, store: PrecomputedStore, config: RAGConfig):
        self.store = store
        self.config = config
        self.embed_model = None

    def _dbg(self, msg: str) -> None:
        if self.config.debug_retrieval_steps:
            print(msg, flush=True)

    def load_embedder(self) -> None:
        if self.embed_model is not None:
            return
        from FlagEmbedding import BGEM3FlagModel
        self.embed_model = BGEM3FlagModel(self.config.embed_model_name, use_fp16=self.config.use_fp16)

    def embed_query(self, query: str) -> np.ndarray:
        self.load_embedder()
        out = self.embed_model.encode(
            [query],
            batch_size=1,
            max_length=self.config.embed_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
        limit = min(limit, len(scores))
        if limit <= 0:
            return np.array([], dtype=int)
        if limit < len(scores):
            idx = np.argpartition(scores, -limit)[-limit:]
            return idx[np.argsort(scores[idx])[::-1]]
        return np.argsort(scores)[::-1]

    def vector_search_numpy(self, query: str, limit: Optional[int] = None, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        limit = limit or self.config.vector_top_k
        q = self.embed_query(query)
        sims = self.store.emb_norm @ q
        if category_filter:
            mask = self.store.category_arr == category_filter
            sims = np.where(mask, sims, -1e9)
        idx = self._top_indices(sims, limit)
        rows: List[Dict[str, Any]] = []
        for rank, i in enumerate(idx, start=1):
            if sims[i] <= -1e8:
                continue
            p = self.store.chunks[int(i)]
            rows.append({"chunk_id": p.get("chunk_id"), "score_vector": float(sims[i]), "rank_vector": rank, "payload": p})
        return rows

    def bm25_search(self, query: str, limit: Optional[int] = None, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        limit = limit or self.config.bm25_top_k
        tokens = tokenize_for_bm25(query)
        scores = np.asarray(self.store.bm25.get_scores(tokens), dtype=np.float32)
        if category_filter:
            bm25_categories = np.array([self.store.chunk_by_id.get(cid, {}).get("category") for cid in self.store.bm25_chunk_ids], dtype=object)
            scores = np.where(bm25_categories == category_filter, scores, -1e9)
        idx = self._top_indices(scores, limit)
        rows: List[Dict[str, Any]] = []
        for rank, i in enumerate(idx, start=1):
            if scores[i] <= -1e8:
                continue
            p = self.store.chunk_by_id.get(self.store.bm25_chunk_ids[int(i)], {})
            rows.append({"chunk_id": p.get("chunk_id"), "score_bm25": float(scores[int(i)]), "rank_bm25": rank, "payload": p})
        return rows

    def reciprocal_rank_fusion(self, vector_hits: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = limit or self.config.hybrid_top_k
        score: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}
        details: dict[str, dict[str, Any]] = {}
        for h in vector_hits:
            cid = h.get("chunk_id")
            if not cid:
                continue
            score[cid] = score.get(cid, 0.0) + 1.0 / (self.config.rrf_k + h["rank_vector"])
            payloads[cid] = h.get("payload", {})
            details.setdefault(cid, {}).update({"rank_vector": h["rank_vector"], "score_vector": h["score_vector"]})
        for h in bm25_hits:
            cid = h.get("chunk_id")
            if not cid:
                continue
            score[cid] = score.get(cid, 0.0) + 1.0 / (self.config.rrf_k + h["rank_bm25"])
            payloads[cid] = h.get("payload", {})
            details.setdefault(cid, {}).update({"rank_bm25": h["rank_bm25"], "score_bm25": h["score_bm25"]})
        ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)[:limit]
        rows: List[Dict[str, Any]] = []
        for rank, (cid, s) in enumerate(ranked, start=1):
            rows.append({
                "rank_hybrid": rank,
                "rank_final": rank,
                "chunk_id": cid,
                "score_hybrid": float(s),
                "score_reranker": None,
                "reranker_used": False,
                **details.get(cid, {}),
                "payload": payloads.get(cid, {}),
            })
        return rows

    def direct_nomor_candidates(self, query: str, nomor: str, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not nomor:
            return rows
        for p in self.store.chunks:
            if category_filter and infer_category_from_payload(p) != category_filter:
                continue
            if infer_nomor_from_payload(p) != nomor:
                continue
            s = lexical_score_for_payload(query, p)
            if s > 0 or len(rows) < self.config.direct_nomor_candidates_k:
                rows.append({
                    "rank_hybrid": 9999,
                    "rank_final": 9999,
                    "chunk_id": p.get("chunk_id"),
                    "score_hybrid": s / 1000.0,
                    "score_direct_nomor": s,
                    "score_reranker": None,
                    "reranker_used": False,
                    "payload": p,
                })
        rows = sorted(rows, key=lambda x: x.get("score_direct_nomor", 0.0), reverse=True)[: self.config.direct_nomor_candidates_k]
        for i, r in enumerate(rows, start=1):
            r["rank_direct_nomor"] = i
        return rows

    @staticmethod
    def merge_hits(primary: List[Dict[str, Any]], extra: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for source in [primary, extra]:
            for h in source:
                cid = h.get("chunk_id") or (h.get("payload") or {}).get("chunk_id")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                out.append(h)
        return out[:limit]

    @staticmethod
    def stat_tokenize(text: str) -> List[str]:
        text = normalize_text(text).lower().replace("’", "'").replace("`", "'")
        toks = re.findall(r"[a-z0-9][a-z0-9'\-]{1,}", text)
        return [t for t in toks if t not in STAT_STOPWORDS and len(t) >= 3]

    def stat_query_terms(self, query: str) -> List[str]:
        toks = self.stat_tokenize(query)
        terms = list(dict.fromkeys(toks))
        qlow = normalize_text(query).lower()
        for concept, variants in STAT_CONCEPT_SYNONYMS.items():
            if any(v in qlow for v in variants):
                for v in variants[:4]:
                    terms.extend(self.stat_tokenize(v))
                terms.extend(self.stat_tokenize(concept))
        return list(dict.fromkeys(terms))[:80]

    def stat_concept_score(self, query: str, combined_low: str, title_low: str) -> float:
        qlow = normalize_text(query).lower()
        score = 0.0
        for _, variants in STAT_CONCEPT_SYNONYMS.items():
            if not any(v in qlow for v in variants):
                continue
            doc_hits = sum(1 for v in variants if v in combined_low)
            title_hits = sum(1 for v in variants if v in title_low)
            if doc_hits:
                score += min(3.0, 0.65 * doc_hits)
            if title_hits:
                score += min(2.5, 1.15 * title_hits)
        return score

    def stat_phrase_score(self, query: str, text_low: str, title_low: str) -> float:
        qlow = normalize_text(query).lower()
        score = 0.0
        toks = self.stat_tokenize(query)
        phrases: list[str] = []
        for n in [4, 3, 2]:
            for i in range(0, max(0, len(toks) - n + 1)):
                ph = " ".join(toks[i:i + n])
                if len(ph) >= 8:
                    phrases.append(ph)
        for ph in list(dict.fromkeys(phrases))[:30]:
            if ph in title_low:
                score += 1.4
            elif ph in text_low:
                score += 0.7
        for term in TITLE_BOOST_TERMS:
            if term in qlow and term in title_low:
                score += 2.0
            elif term in qlow and term in text_low:
                score += 0.7
        return score

    def stat_keyword_coverage(self, query: str, text: str, title: str = "") -> tuple[float, int, int]:
        terms = self.stat_query_terms(query)
        if not terms:
            return 0.0, 0, 0
        text_low = normalize_text(text).lower()
        title_low = normalize_text(title).lower()
        hits = 0
        weighted = 0.0
        for t in terms:
            in_title = t in title_low
            in_text = t in text_low
            if in_title or in_text:
                hits += 1
                weighted += 2.0 if in_title else 1.0
        return hits / max(1, len(terms)) + 0.03 * weighted, hits, len(terms)

    def statistical_score_hit(self, query: str, hit: Dict[str, Any]) -> Dict[str, float]:
        p = hit.get("payload", {}) or {}
        title = str(p.get("title") or p.get("doc_title") or p.get("pdf_filename") or "")
        text = payload_text(p)
        combined = f"{title}\n{text}"
        combined_low = normalize_text(combined).lower()
        title_low = normalize_text(title).lower()
        coverage_score, kw_hits, kw_total = self.stat_keyword_coverage(query, text, title)
        concept_score = self.stat_concept_score(query, combined_low, title_low)
        phrase_score = self.stat_phrase_score(query, combined_low, title_low)
        rrf_score = float(hit.get("score_hybrid") or hit.get("score_rrf") or 0.0)
        vector_score = float(hit.get("score_vector") or 0.0)
        bm25_rank = hit.get("rank_bm25")
        vector_rank = hit.get("rank_vector")
        bm25_rank_score = 1.0 / (1.0 + float(bm25_rank)) if bm25_rank else 0.0
        vector_rank_score = 1.0 / (1.0 + float(vector_rank)) if vector_rank else 0.0
        title_score = 0.0
        qlow = normalize_text(query).lower()
        for term in TITLE_BOOST_TERMS:
            if term in qlow and term in title_low:
                title_score += 1.2
        title_score = min(title_score, 5.0)
        length_score = 0.5 if len(normalize_text(text)) >= 180 else -0.8
        total = (
            10.0 * rrf_score + 1.2 * vector_score + 1.4 * bm25_rank_score + 1.0 * vector_rank_score +
            6.0 * coverage_score + 2.2 * concept_score + 1.3 * phrase_score + 1.8 * title_score + length_score
        )
        return {
            "score_reranker": float(total),
            "statistical_score": float(total),
            "score_stat_rrf": float(rrf_score),
            "score_stat_vector": float(vector_score),
            "score_stat_bm25_rank": float(bm25_rank_score),
            "score_stat_vector_rank": float(vector_rank_score),
            "score_stat_keyword_coverage": float(coverage_score),
            "score_stat_keyword_hits": float(kw_hits),
            "score_stat_keyword_total": float(kw_total),
            "score_stat_concept": float(concept_score),
            "score_stat_phrase": float(phrase_score),
            "score_stat_title": float(title_score),
            "score_stat_length": float(length_score),
        }

    def rerank_hits(self, query: str, hits: List[Dict[str, Any]], final_k: Optional[int] = None) -> List[Dict[str, Any]]:
        final_k = final_k or self.config.final_context_k
        scored: List[Dict[str, Any]] = []
        for h in hits:
            hh = dict(h)
            hh.update(self.statistical_score_hit(query, hh))
            hh["reranker_used"] = True
            hh["reranker_type"] = self.config.reranker_type
            scored.append(hh)
        ranked = sorted(scored, key=lambda x: x.get("score_reranker", -1e9), reverse=True)[:final_k]
        for i, h in enumerate(ranked, start=1):
            h["rank_final"] = i
        return ranked

    def retrieve(self, query: str, category_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = now_ms()
        self._dbg("[1/5] vector search")
        vh = self.vector_search_numpy(query, limit=self.config.vector_top_k, category_filter=category_filter)
        t1 = now_ms()
        self._dbg("[2/5] BM25 search")
        bh = self.bm25_search(query, limit=self.config.bm25_top_k, category_filter=category_filter)
        t2 = now_ms()
        self._dbg("[3/5] RRF fusion")
        hybrid = self.reciprocal_rank_fusion(vh, bh, limit=self.config.hybrid_top_k)
        requested_nomor = extract_requested_nomor(query)
        direct_count = 0
        if requested_nomor:
            direct = self.direct_nomor_candidates(query, requested_nomor, category_filter=category_filter)
            direct_count = len(direct)
            hybrid = self.merge_hits(hybrid, direct, limit=self.config.hybrid_top_k + self.config.direct_nomor_candidates_k)
            filtered = [h for h in hybrid if infer_nomor_from_payload(h.get("payload", {})) == requested_nomor]
            if filtered:
                hybrid = filtered
        if category_filter:
            hybrid = [h for h in hybrid if infer_category_from_payload(h.get("payload", {})) == category_filter]
        for i, h in enumerate(hybrid, start=1):
            h["rank_hybrid_filtered"] = i
        t3 = now_ms()
        self._dbg("[4/5] statistical rerank")
        final = self.rerank_hits(query, hybrid, final_k=self.config.final_context_k)
        t4 = now_ms()
        timing = {
            "vector_ms": round(t1 - t0, 2),
            "bm25_ms": round(t2 - t1, 2),
            "rrf_ms": round(t3 - t2, 2),
            "rerank_ms": round(t4 - t3, 2),
            "retrieval_total_ms": round(t4 - t0, 2),
            "requested_nomor": requested_nomor,
            "direct_nomor_candidates": direct_count,
            "reranker_type": self.config.reranker_type,
        }
        return final, timing
