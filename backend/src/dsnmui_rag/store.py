from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import pickle
import numpy as np

from .config import RAGConfig
from .utils import read_jsonl


@dataclass
class PrecomputedStore:
    config: RAGConfig
    chunks: list[dict[str, Any]]
    chunk_by_id: dict[str, dict[str, Any]]
    embeddings: np.ndarray
    emb_norm: np.ndarray
    bm25: Any
    bm25_chunk_ids: list[str]
    category_arr: np.ndarray
    chunk_id_arr: np.ndarray

    @classmethod
    def load(cls, config: RAGConfig) -> "PrecomputedStore":
        missing = [p for p in [config.chunks_path, config.embeddings_path, config.bm25_path] if not Path(p).exists()]
        if missing:
            msg = "File precomputed belum ditemukan:\n" + "\n".join(f"- {p}" for p in missing)
            msg += "\nPastikan folder precomputed berisi chunks.jsonl, embeddings_bge_m3.npy, dan bm25_index.pkl."
            raise FileNotFoundError(msg)

        chunks = read_jsonl(config.chunks_path)
        chunk_by_id = {c.get("chunk_id"): c for c in chunks if c.get("chunk_id")}
        embeddings = np.load(config.embeddings_path).astype(np.float32)
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        with open(config.bm25_path, "rb") as f:
            bm25_pack = pickle.load(f)
        bm25 = bm25_pack["bm25"]
        bm25_chunk_ids = bm25_pack["chunk_ids"]
        return cls(
            config=config,
            chunks=chunks,
            chunk_by_id=chunk_by_id,
            embeddings=embeddings,
            emb_norm=emb_norm,
            bm25=bm25,
            bm25_chunk_ids=bm25_chunk_ids,
            category_arr=np.array([c.get("category") for c in chunks], dtype=object),
            chunk_id_arr=np.array([c.get("chunk_id") for c in chunks], dtype=object),
        )

    def summary(self) -> dict[str, Any]:
        cats: dict[str, int] = {}
        arabic = 0
        for c in self.chunks:
            cats[str(c.get("category"))] = cats.get(str(c.get("category")), 0) + 1
            if c.get("arabic_text"):
                arabic += 1
        return {
            "chunks": len(self.chunks),
            "embeddings_shape": list(self.embeddings.shape),
            "bm25_chunk_ids": len(self.bm25_chunk_ids),
            "chunks_with_arabic_text": arabic,
            "categories": cats,
            "precomp_dir": str(self.config.precomp_dir),
        }
