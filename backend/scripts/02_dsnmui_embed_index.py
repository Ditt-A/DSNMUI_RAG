#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2 DSN-MUI RAG precompute: pages.jsonl -> chunks -> BGE-M3 -> Qdrant + BM25."""
from __future__ import annotations
import argparse, json, os, pickle, re, sys, time, unicodedata
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import regex as reg
from tqdm.auto import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ARABIC_PATTERN = reg.compile(r"[\p{Script=Arabic}]")
TOKEN_PATTERN = reg.compile(r"[\p{L}\p{N}_]+")
SECTION_MARKERS = ["Menimbang", "Mengingat", "Memperhatikan", "MEMUTUSKAN", "Menetapkan", "Ketentuan Umum", "Ketentuan Hukum", "Pertama", "Kedua", "Ketiga", "Keempat", "Kelima", "Keenam", "Ketujuh", "Kedelapan", "Kesembilan", "Kesepuluh", "Ditetapkan di", "Dewan Syariah Nasional"]

def parse_bool(x):
    return x if isinstance(x, bool) else str(x).lower() in {"1", "true", "yes", "y", "on"}

def normalize_text(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize("NFKC", text).replace("\x00", " ").replace("\ufeff", " ")
    text = reg.sub(r"[\t\r\f\v]+", " ", text)
    text = reg.sub(r" *\n *", "\n", text)
    text = reg.sub(r"\n{3,}", "\n\n", text)
    text = reg.sub(r"[ ]{2,}", " ", text)
    return text.strip()

def count_arabic_chars(text: str) -> int:
    return len(ARABIC_PATTERN.findall(text or ""))

def extract_arabic_text(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if count_arabic_chars(ln) >= 2]
    return normalize_text("\n".join(lines))

def tokenize_for_bm25(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text(text).lower())

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists(): return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def guess_section(text: str) -> str:
    lower = (text or "").lower()
    return " | ".join([m for m in SECTION_MARKERS if m.lower() in lower][:3])

def smart_cut_end(text: str, start: int, target_end: int, max_end: int) -> int:
    if target_end >= max_end: return max_end
    ws = max(start + 200, target_end - 250)
    window = text[ws:target_end]
    best = max([window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "), window.rfind("; ")])
    return ws + best + 1 if best > 50 else target_end

def split_text_chunks(text: str, chunk_size: int, overlap: int, min_chars: int) -> List[str]:
    text = normalize_text(text)
    if not text: return []
    if len(text) <= chunk_size: return [text] if len(text) >= min_chars else []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = smart_cut_end(text, start, min(start + chunk_size, n), n)
        chunk = normalize_text(text[start:end])
        if len(chunk) >= min_chars: chunks.append(chunk)
        if end >= n: break
        start = max(end - overlap, start + 1)
    return chunks

def build_chunks(pages: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    chunks = []
    for r in tqdm(pages, desc="Chunking pages"):
        if r.get("parse_status") != "ok": continue
        for j, piece in enumerate(split_text_chunks(r.get("text_combined") or "", args.chunk_size_chars, args.chunk_overlap_chars, args.min_chunk_chars)):
            chunk_id_raw = f"{r.get('doc_id')}_p{r.get('page')}_c{j+1}"
            chunk_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", chunk_id_raw)[:220]
            chunks.append({
                "chunk_id": chunk_id, "doc_id": r.get("doc_id"), "category": r.get("category"),
                "nomor": r.get("nomor"), "title": r.get("title"), "page": r.get("page"),
                "page_count": r.get("page_count"), "section_guess": guess_section(piece),
                "text_for_embedding": normalize_text(piece), "text_original": piece,
                "arabic_text": extract_arabic_text(piece), "arabic_char_count": count_arabic_chars(piece),
                "source_url": r.get("source_url", ""), "pdf_filename": r.get("pdf_filename"),
                "pdf_path": r.get("pdf_path"), "pdf_sha256": r.get("pdf_sha256"),
                "chunk_index_on_page": j + 1, "chunk_char_len": len(piece),
            })
    return chunks

def embed_chunks(chunks: List[Dict[str, Any]], args, embed_path: Path, chunk_ids_path: Path) -> np.ndarray:
    from FlagEmbedding import BGEM3FlagModel
    import torch
    texts = [c["text_for_embedding"] for c in chunks]
    print("Total texts to embed:", len(texts))
    if torch.cuda.is_available():
        print("Torch:", torch.__version__)
        print("Torch CUDA:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA memory total GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    else:
        print("WARNING: CUDA tidak tersedia. Embedding akan lambat di CPU.")
    model = BGEM3FlagModel(args.embed_model_name, use_fp16=args.use_fp16)
    out_vecs = []
    for start in tqdm(range(0, len(texts), args.embed_batch_size), desc="Embedding BGE-M3"):
        batch = texts[start:start + args.embed_batch_size]
        out = model.encode(batch, batch_size=len(batch), max_length=args.embed_max_length, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        out_vecs.append(np.asarray(out["dense_vecs"], dtype=np.float32))
    embeddings = np.vstack(out_vecs).astype(np.float32)
    np.save(embed_path, embeddings)
    chunk_ids_path.write_text(json.dumps([c["chunk_id"] for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8")
    print("Embeddings shape:", embeddings.shape)
    print("Saved:", embed_path)
    return embeddings

def build_qdrant(chunks, embeddings, qdrant_dir: Path, collection_name: str, recreate: bool):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    client = QdrantClient(path=str(qdrant_dir))
    try: exists = client.collection_exists(collection_name)
    except Exception: exists = False
    if exists and recreate:
        client.delete_collection(collection_name); exists = False
    if not exists:
        client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE))
    for start in tqdm(range(0, len(chunks), 128), desc="Upsert Qdrant"):
        points = [PointStruct(id=i, vector=embeddings[i].tolist(), payload=dict(chunks[i])) for i in range(start, min(start + 128, len(chunks)))]
        client.upsert(collection_name=collection_name, points=points)
    print(client.get_collection(collection_name))

def build_bm25(chunks, bm25_path: Path):
    from rank_bm25 import BM25Okapi
    corpus = [tokenize_for_bm25(c["text_for_embedding"]) for c in tqdm(chunks, desc="Tokenize BM25")]
    bm25 = BM25Okapi(corpus)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in chunks], "tokenizer": "regex_unicode_letters_numbers_v1"}, f)
    print("BM25 saved:", bm25_path)

def retrieval_test(precomp_dir: Path, collection_name: str, embed_model_name: str, use_fp16: bool, embed_max_length: int):
    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client import QdrantClient
    chunks = read_jsonl(precomp_dir / "chunks.jsonl")
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    client = QdrantClient(path=str(precomp_dir / "qdrant_local"))
    with open(precomp_dir / "bm25_index.pkl", "rb") as f: pack = pickle.load(f)
    bm25, bm25_chunk_ids = pack["bm25"], pack["chunk_ids"]
    model = BGEM3FlagModel(embed_model_name, use_fp16=use_fp16)
    query = "Apa dalil hadis tentang bepergian dalam fatwa pariwisata syariah?"
    vec = np.asarray(model.encode([query], batch_size=1, max_length=embed_max_length, return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0], dtype=np.float32)
    try:
        vh = client.query_points(collection_name=collection_name, query=vec.tolist(), limit=30, with_payload=True).points
    except Exception:
        vh = client.search(collection_name=collection_name, query_vector=vec.tolist(), limit=30, with_payload=True)
    scores = bm25.get_scores(tokenize_for_bm25(query))
    idx = np.argsort(scores)[::-1][:30]
    rrf, payloads = {}, {}
    for rank, hit in enumerate(vh, 1):
        cid = (hit.payload or {}).get("chunk_id")
        if cid: rrf[cid] = rrf.get(cid, 0) + 1/(60+rank); payloads[cid] = hit.payload
    for rank, i in enumerate(idx, 1):
        cid = bm25_chunk_ids[int(i)]
        rrf[cid] = rrf.get(cid, 0) + 1/(60+rank); payloads.setdefault(cid, chunk_by_id.get(cid, {}))
    print("\nRetrieval test query:", query)
    for n, (cid, _) in enumerate(sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:5], 1):
        p = payloads[cid]
        print("="*100)
        print(f"#{n} | {p.get('category')} | {p.get('nomor')} | {p.get('title')} | page {p.get('page')}")
        print((p.get("text_original") or p.get("text_for_embedding") or "")[:900])
        if p.get("arabic_text"):
            print("\n[Arabic sample]"); print(p["arabic_text"][:500])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default="dsnmui_rag_precompute")
    parser.add_argument("--chunk-size-chars", type=int, default=1700)
    parser.add_argument("--chunk-overlap-chars", type=int, default=250)
    parser.add_argument("--min-chunk-chars", type=int, default=80)
    parser.add_argument("--embed-model-name", default="BAAI/bge-m3")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--embed-max-length", type=int, default=8192)
    parser.add_argument("--use-fp16", type=parse_bool, default=True)
    parser.add_argument("--collection-name", default="dsnmui_chunks")
    parser.add_argument("--recreate-qdrant", type=parse_bool, default=True)
    parser.add_argument("--skip-retrieval-test", type=parse_bool, default=False)
    args = parser.parse_args()
    precomp_dir = Path(args.project_dir) / "precomputed"
    qdrant_dir = precomp_dir / "qdrant_local"; qdrant_dir.mkdir(parents=True, exist_ok=True)
    pages_path = precomp_dir / "pages.jsonl"
    chunks_path, embed_path, chunk_ids_path, bm25_path, stats_path = precomp_dir/"chunks.jsonl", precomp_dir/"embeddings_bge_m3.npy", precomp_dir/"chunk_ids.json", precomp_dir/"bm25_index.pkl", precomp_dir/"stats.json"
    print("Python:", sys.version); print("PAGES_PATH:", pages_path)
    pages = read_jsonl(pages_path)
    if not pages: raise FileNotFoundError(f"pages.jsonl kosong/tidak ditemukan: {pages_path}. Jalankan 01_dsnmui_ocr_pages.py dulu.")
    t0 = time.time()
    chunks = build_chunks(pages, args)
    if not chunks: raise RuntimeError("Tidak ada chunk yang terbentuk. Cek pages.jsonl/text_combined.")
    write_jsonl(chunks_path, chunks)
    print("Total chunks:", len(chunks))
    df = pd.DataFrame([{k:v for k,v in c.items() if k not in ["text_for_embedding","text_original","arabic_text"]} for c in chunks])
    if len(df): print(df.groupby("category").agg(docs=("doc_id","nunique"), chunks=("chunk_id","count"), arabic_chunks=("arabic_char_count", lambda s:int((s.fillna(0)>0).sum())), avg_chars=("chunk_char_len","mean")).reset_index().to_string(index=False))
    embeddings = embed_chunks(chunks, args, embed_path, chunk_ids_path)
    build_qdrant(chunks, embeddings, qdrant_dir, args.collection_name, args.recreate_qdrant)
    build_bm25(chunks, bm25_path)
    stats = {"num_pages_rows": len(pages), "num_chunks": len(chunks), "embedding_model": args.embed_model_name, "embedding_shape": list(embeddings.shape), "chunk_size_chars": args.chunk_size_chars, "chunk_overlap_chars": args.chunk_overlap_chars, "qdrant_dir": str(qdrant_dir), "qdrant_collection": args.collection_name, "bm25_path": str(bm25_path), "pages_path": str(pages_path), "chunks_path": str(chunks_path), "elapsed_seconds_stage2": round(time.time()-t0,2)}
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not args.skip_retrieval_test:
        retrieval_test(precomp_dir, args.collection_name, args.embed_model_name, args.use_fp16, args.embed_max_length)
    print("DONE Stage 2 embedding/index.")

if __name__ == "__main__": main()
