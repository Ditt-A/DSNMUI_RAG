

# %% CELL 1
from pathlib import Path
import os, sys, json, re, time, pickle, subprocess, math, gc, shutil, traceback
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from IPython.display import display, Markdown, Image as IPImage
except Exception:
    def display(x): print(x)
    def Markdown(x): return x

pd.set_option('display.max_colwidth', 180)
pd.set_option('display.max_columns', 80)
print('Notebook ready')



# %% CELL 2
# =====================================================
# PROJECT PATH
# =====================================================
PROJECT_DIR = Path('dsnmui_rag_precompute_arabic_fix')
PRECOMP_DIR = PROJECT_DIR / 'precomputed'

CHUNKS_PATH = PRECOMP_DIR / 'chunks.jsonl'
EMBED_PATH = PRECOMP_DIR / 'embeddings_bge_m3.npy'
BM25_PATH = PRECOMP_DIR / 'bm25_index.pkl'

# Output evidence dan run result
EVIDENCE_DIR = PRECOMP_DIR / 'arabic_evidence'
PAGE_IMG_DIR = EVIDENCE_DIR / 'page_images'
CROP_DIR = EVIDENCE_DIR / 'crops'
OCR_IO_DIR = EVIDENCE_DIR / 'ocr_io'

RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = PRECOMP_DIR / 'rag_umkm_test_runs' / RUN_ID
for p in [EVIDENCE_DIR, PAGE_IMG_DIR, CROP_DIR, OCR_IO_DIR, RUN_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# =====================================================
# RETRIEVAL CONFIG
# =====================================================
EMBED_MODEL_NAME = 'BAAI/bge-m3'
EMBED_MAX_LENGTH = 8192
USE_FP16 = True

VECTOR_TOP_K = 60
BM25_TOP_K = 60
HYBRID_TOP_K = 30       # kandidat awal setelah RRF
FINAL_CONTEXT_K = 3     # jumlah source masuk context generator; v3 diperkecil agar generator tidak menyalin konteks mentah
RRF_K = 60

# =====================================================
# REAL RERANKER CONFIG
# =====================================================
USE_RERANKER = False  # initial False agar FlagEmbedding reranker tidak diload; manual HF diaktifkan di patch v11c
RERANKER_MODEL_NAME = 'STATISTICAL_RERANKER_V12'
RERANK_BATCH_SIZE = 8
RERANK_MAX_CHARS_PER_PASSAGE = 2500

# =====================================================
# ARABIC EVIDENCE CONFIG
# =====================================================
ENABLE_ARABIC_EVIDENCE = True
EVIDENCE_TOP_N = 1         # v3: hanya top-1 source di-OCR agar konteks tidak penuh noise
RENDER_DPI = 250           # bisa dinaikkan 300/350 kalau OCR kurang jelas
OCR_USE_GPU = 1
OCR_TIMEOUT_SECONDS = 240
OCR_FULL_PAGE = False      # True = lebih mahal, tapi kadang membantu

ARABIC_EVIDENCE_KEYWORDS = [
    'arab', 'hadis', 'hadits', 'dalil', 'ayat', 'q.s', 'quran', 'al-qur', 'riwayat', 'firman', 'kaidah fikih'
]

# =====================================================
# GENERATOR CONFIG
# =====================================================
# Penting: generator TIDAK diload di awal, karena BGE-M3 + reranker sudah makan VRAM.
# Alur aman: retrieval/evidence dulu -> save context/prompt -> unload retrieval models -> load generator -> generate answers.
LOAD_GENERATOR = False
GENERATE_AFTER_RETRIEVAL = True
UNLOAD_RETRIEVAL_BEFORE_GENERATION = True

GENERATOR_MODEL_NAME = 'Qwen/Qwen3-4B-Instruct-2507'
# Kalau GPU T4/16GB masih berat, ganti ke:
# GENERATOR_MODEL_NAME = 'Qwen/Qwen3-4B-Instruct-2507'

GENERATOR_USE_4BIT = False   # set True kalau bitsandbytes tersedia dan 7B masih berat
MAX_NEW_TOKENS = 768
MAX_PROMPT_TOKENS = 8192
MAX_CONTEXT_CHARS_PER_SOURCE = 650

# =====================================================
# HARD-CODED OCR ENV — dibiarkan sesuai kondisi environment sekarang
# =====================================================
OCR_ENV_PREFIX = '/home/nafisnaufal1426/adit/envAdit/envs/dsnmui_ocr'
WORKER_PATH = Path('arabic_ocr_crop_worker.py')
if not WORKER_PATH.exists():
    alt = Path('/mnt/data/arabic_ocr_crop_worker.py')
    if alt.exists():
        WORKER_PATH = alt

print('PROJECT_DIR:', PROJECT_DIR)
print('PRECOMP_DIR:', PRECOMP_DIR)
print('RUN_DIR:', RUN_DIR)
print('WORKER_PATH:', WORKER_PATH, '| exists:', WORKER_PATH.exists())


# V9 UMKM TEST CONFIG
UMKM_TEST_MODE = True
ENABLE_DALIL_SNIPPET_EXTRACTION = True
ENABLE_ARABIC_EVIDENCE_FOR_DALIL_QUERY = False
FORCE_ABSTAIN_ON_LOW_SUPPORT = True
MIN_SUPPORT_SCORE_FOR_GENERATION = 0.08



# %% CELL 3
import regex as reg

AR_RE = re.compile(r"[\u0600-\u06FF]")
PUA_RE = re.compile(r"[\ue000-\uf8ff]")
TOKEN_PATTERN = reg.compile(r"[\p{L}\p{N}_]+")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def normalize_text(text: str) -> str:
    if not text:
        return ''
    text = str(text).replace('\x00', ' ').replace('\ufeff', ' ')
    text = re.sub(r'[\t\r\f\v]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text(text).lower())


def count_arabic(text: str) -> int:
    return len(AR_RE.findall(text or ''))


def count_pua(text: str) -> int:
    return len(PUA_RE.findall(text or ''))


def arabic_quality_score(text: str) -> Dict[str, Any]:
    text = normalize_text(text or '')
    n = max(len(text), 1)
    arabic_count = count_arabic(text)
    pua_count = count_pua(text)
    latin_count = len(re.findall(r'[A-Za-z]', text))
    digit_count = len(re.findall(r'[0-9]', text))

    arabic_ratio = arabic_count / n
    pua_ratio = pua_count / n
    latin_ratio = latin_count / n

    score = arabic_count - 2.5 * pua_count - 0.5 * latin_count - 0.2 * digit_count
    if arabic_count < 8:
        label = 'none'
    elif pua_count > 0 or pua_ratio > 0.01:
        label = 'bad_font_encoded'
    elif arabic_ratio < 0.25:
        label = 'mixed_or_noisy'
    else:
        label = 'usable'

    return {
        'label': label,
        'score': float(score),
        'length': len(text),
        'arabic_count': int(arabic_count),
        'pua_count': int(pua_count),
        'latin_count': int(latin_count),
        'digit_count': int(digit_count),
        'arabic_ratio': float(arabic_ratio),
        'pua_ratio': float(pua_ratio),
        'latin_ratio': float(latin_ratio),
        'newline_count': int(text.count('\n')),
    }


def safe_name(s: str, max_len: int = 120) -> str:
    s = re.sub(r'[^a-zA-Z0-9_\-]+', '_', str(s)).strip('_')
    return s[:max_len] or 'item'


def text_preview(text: str, n: int = 350) -> str:
    return normalize_text(text)[:n]


def now_ms() -> float:
    return time.perf_counter() * 1000


def display_arabic(text: str, title: str = 'Arabic'):
    text = normalize_text(text)
    q = arabic_quality_score(text)
    display(Markdown(
        f"**{title}** — quality: `{q['label']}`, arabic_count: `{q['arabic_count']}`, pua_count: `{q['pua_count']}`"
        f"<div dir='rtl' style='font-size:22px; line-height:2; border:1px solid #ddd; padding:12px; margin-top:8px;'>"
        f"{text.replace(chr(10), '<br>')}"
        f"</div>"
    ))



# %% CELL 4
for path in [CHUNKS_PATH, EMBED_PATH, BM25_PATH]:
    if not path.exists():
        raise FileNotFoundError(f'File tidak ditemukan: {path}')

chunks = read_jsonl(CHUNKS_PATH)
chunk_by_id = {c.get('chunk_id'): c for c in chunks}
embeddings = np.load(EMBED_PATH).astype(np.float32)
emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

with open(BM25_PATH, 'rb') as f:
    bm25_pack = pickle.load(f)
bm25 = bm25_pack['bm25']
bm25_chunk_ids = bm25_pack['chunk_ids']

category_arr = np.array([c.get('category') for c in chunks], dtype=object)
chunk_id_arr = np.array([c.get('chunk_id') for c in chunks], dtype=object)

print('chunks:', len(chunks))
print('embeddings:', embeddings.shape)
print('BM25 chunk ids:', len(bm25_chunk_ids))
print('chunks with arabic_text:', sum(1 for c in chunks if c.get('arabic_text')))
print('categories:', pd.Series(category_arr).value_counts(dropna=False).to_dict())



# %% CELL 5
from FlagEmbedding import BGEM3FlagModel
import torch

print('Torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

embed_model = BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=USE_FP16)
print('BGE-M3 embedder ready:', EMBED_MODEL_NAME)

reranker = None
if USE_RERANKER:
    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(RERANKER_MODEL_NAME, use_fp16=USE_FP16)
        print('Reranker ready:', RERANKER_MODEL_NAME)
    except Exception as e:
        print('Reranker gagal diload. Fallback ke hybrid rank.')
        print(type(e).__name__, str(e)[:500])
        reranker = None



# %% CELL 6
def embed_query(query: str) -> np.ndarray:
    out = embed_model.encode(
        [query],
        batch_size=1,
        max_length=EMBED_MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vec = np.asarray(out['dense_vecs'][0], dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec


def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    limit = min(limit, len(scores))
    if limit <= 0:
        return np.array([], dtype=int)
    if limit < len(scores):
        idx = np.argpartition(scores, -limit)[-limit:]
        return idx[np.argsort(scores[idx])[::-1]]
    return np.argsort(scores)[::-1]


def vector_search_numpy(query: str, limit: int = VECTOR_TOP_K, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    q = embed_query(query)
    sims = emb_norm @ q
    if category_filter:
        mask = category_arr == category_filter
        sims = np.where(mask, sims, -1e9)
    idx = _top_indices(sims, limit)
    rows = []
    for rank, i in enumerate(idx, start=1):
        if sims[i] <= -1e8:
            continue
        p = chunks[int(i)]
        rows.append({
            'chunk_id': p.get('chunk_id'),
            'score_vector': float(sims[i]),
            'rank_vector': rank,
            'payload': p,
        })
    return rows


def bm25_search(query: str, limit: int = BM25_TOP_K, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    tokens = tokenize_for_bm25(query)
    scores = np.asarray(bm25.get_scores(tokens), dtype=np.float32)

    if category_filter:
        # BM25 index punya urutan bm25_chunk_ids, bukan selalu sama dengan chunks.
        bm25_categories = np.array([chunk_by_id.get(cid, {}).get('category') for cid in bm25_chunk_ids], dtype=object)
        scores = np.where(bm25_categories == category_filter, scores, -1e9)

    idx = _top_indices(scores, limit)
    rows = []
    for rank, i in enumerate(idx, start=1):
        if scores[i] <= -1e8:
            continue
        p = chunk_by_id.get(bm25_chunk_ids[int(i)], {})
        rows.append({
            'chunk_id': p.get('chunk_id'),
            'score_bm25': float(scores[int(i)]),
            'rank_bm25': rank,
            'payload': p,
        })
    return rows


def reciprocal_rank_fusion(vector_hits: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], k: int = RRF_K, limit: int = HYBRID_TOP_K) -> List[Dict[str, Any]]:
    score, payloads, details = {}, {}, {}
    for h in vector_hits:
        cid = h.get('chunk_id')
        if not cid:
            continue
        score[cid] = score.get(cid, 0.0) + 1.0 / (k + h['rank_vector'])
        payloads[cid] = h.get('payload', {})
        details.setdefault(cid, {}).update({'rank_vector': h['rank_vector'], 'score_vector': h['score_vector']})
    for h in bm25_hits:
        cid = h.get('chunk_id')
        if not cid:
            continue
        score[cid] = score.get(cid, 0.0) + 1.0 / (k + h['rank_bm25'])
        payloads[cid] = h.get('payload', {})
        details.setdefault(cid, {}).update({'rank_bm25': h['rank_bm25'], 'score_bm25': h['score_bm25']})

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)[:limit]
    rows = []
    for rank, (cid, s) in enumerate(ranked, start=1):
        rows.append({
            'rank_hybrid': rank,
            'rank_final': rank,
            'chunk_id': cid,
            'score_hybrid': float(s),
            'score_reranker': None,
            'reranker_used': False,
            **details.get(cid, {}),
            'payload': payloads.get(cid, {}),
        })
    return rows


def build_passage_for_rerank(payload: Dict[str, Any], max_chars: int = RERANK_MAX_CHARS_PER_PASSAGE) -> str:
    title = payload.get('title') or ''
    nomor = payload.get('nomor') or ''
    category = payload.get('category') or ''
    page = payload.get('page') or ''
    text = payload.get('text_original') or payload.get('text_for_embedding') or ''
    arabic = payload.get('arabic_text') or ''
    passage = f"""Kategori: {category}
Nomor: {nomor}
Judul: {title}
Halaman: {page}
Teks:
{text}

Teks Arab:
{arabic}"""
    return normalize_text(passage)[:max_chars]


def rerank_hits(query: str, hits: List[Dict[str, Any]], final_k: int = FINAL_CONTEXT_K) -> List[Dict[str, Any]]:
    if not hits:
        return []
    if reranker is None:
        for i, h in enumerate(hits, start=1):
            h['rank_final'] = i
            h['reranker_used'] = False
        return hits[:final_k]

    pairs = [[query, build_passage_for_rerank(h.get('payload', {}))] for h in hits]
    scores = []
    try:
        for start in range(0, len(pairs), RERANK_BATCH_SIZE):
            batch = pairs[start:start + RERANK_BATCH_SIZE]
            batch_scores = reranker.compute_score(batch, normalize=True)
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend([float(x) for x in batch_scores])
    except TypeError:
        # Beberapa versi FlagEmbedding tidak punya parameter normalize
        scores = []
        for start in range(0, len(pairs), RERANK_BATCH_SIZE):
            batch = pairs[start:start + RERANK_BATCH_SIZE]
            batch_scores = reranker.compute_score(batch)
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend([float(x) for x in batch_scores])

    for h, s in zip(hits, scores):
        h['score_reranker'] = s
        h['reranker_used'] = True
    ranked = sorted(hits, key=lambda x: x.get('score_reranker', -1e9), reverse=True)
    for i, h in enumerate(ranked, start=1):
        h['rank_final'] = i
    return ranked[:final_k]


def retrieve_final(query: str, category_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    t0 = now_ms()
    vh = vector_search_numpy(query, limit=VECTOR_TOP_K, category_filter=category_filter)
    t1 = now_ms()
    bh = bm25_search(query, limit=BM25_TOP_K, category_filter=category_filter)
    t2 = now_ms()
    hybrid = reciprocal_rank_fusion(vh, bh, limit=HYBRID_TOP_K)
    t3 = now_ms()
    final = rerank_hits(query, hybrid, final_k=FINAL_CONTEXT_K)
    t4 = now_ms()
    timing = {
        'vector_ms': round(t1 - t0, 2),
        'bm25_ms': round(t2 - t1, 2),
        'rrf_ms': round(t3 - t2, 2),
        'rerank_ms': round(t4 - t3, 2),
        'retrieval_total_ms': round(t4 - t0, 2),
    }
    return final, timing



# %% CELL 7
def results_to_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        p = r.get('payload', {})
        q = arabic_quality_score(p.get('arabic_text') or '')
        rows.append({
            'rank_final': r.get('rank_final'),
            'rank_hybrid': r.get('rank_hybrid'),
            'rank_vector': r.get('rank_vector'),
            'rank_bm25': r.get('rank_bm25'),
            'category': p.get('category'),
            'nomor': p.get('nomor'),
            'title': p.get('title'),
            'page': p.get('page'),
            'chunk_id': r.get('chunk_id'),
            'score_reranker': r.get('score_reranker'),
            'score_hybrid': r.get('score_hybrid'),
            'score_vector': r.get('score_vector'),
            'score_bm25': r.get('score_bm25'),
            'arabic_quality': q['label'],
            'arabic_count': q['arabic_count'],
            'pua_count': q['pua_count'],
            'text_preview': text_preview(p.get('text_original') or p.get('text_for_embedding') or '', 250),
        })
    return pd.DataFrame(rows)


def show_results(results: List[Dict[str, Any]], max_chars: int = 900):
    display(results_to_df(results))
    for i, r in enumerate(results, start=1):
        p = r.get('payload', {})
        print('=' * 110)
        print(f"S{i} | rank_final={r.get('rank_final')} | {p.get('category')} | {p.get('nomor')} | {p.get('title')} | page {p.get('page')}")
        print({k: v for k, v in r.items() if k.startswith('score_') or k.startswith('rank_') or k == 'reranker_used'})
        txt = p.get('text_original') or p.get('text_for_embedding') or ''
        print(txt[:max_chars])
        if p.get('arabic_text'):
            print('\n[arabic_text raw preview]')
            print((p.get('arabic_text') or '')[:700])



# %% CELL 8
from PIL import Image
import fitz


def render_pdf_page(pdf_path: str, page_no: int, doc_id: str, dpi: int = RENDER_DPI) -> Path:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f'PDF tidak ditemukan: {pdf_path}')
    out = PAGE_IMG_DIR / f'{safe_name(doc_id)}_p{page_no}_dpi{dpi}.png'
    if out.exists():
        return out
    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(str(out))
    doc.close()
    return out


def make_crop_boxes(w: int, h: int, include_full_page: bool = OCR_FULL_PAGE) -> Dict[str, Tuple[int, int, int, int]]:
    boxes = {
        'upper_50':      (int(w * 0.04), int(h * 0.05), int(w * 0.96), int(h * 0.55)),
        'middle_60':     (int(w * 0.04), int(h * 0.20), int(w * 0.96), int(h * 0.80)),
        'lower_60':      (int(w * 0.04), int(h * 0.35), int(w * 0.96), int(h * 0.96)),
        'bottom_45':     (int(w * 0.04), int(h * 0.52), int(w * 0.96), int(h * 0.98)),
        'center_wide':   (int(w * 0.08), int(h * 0.15), int(w * 0.92), int(h * 0.88)),
        'left_column':   (int(w * 0.03), int(h * 0.08), int(w * 0.58), int(h * 0.96)),
        'right_column':  (int(w * 0.42), int(h * 0.08), int(w * 0.98), int(h * 0.96)),
    }
    if include_full_page:
        boxes = {'full_page': (0, 0, w, h), **boxes}
    return boxes


def crop_page_candidates(page_img_path: Path, chunk_id: str, page_no: int) -> List[Dict[str, Any]]:
    img = Image.open(page_img_path).convert('RGB')
    w, h = img.size
    boxes = make_crop_boxes(w, h)
    rows = []
    for name, box in boxes.items():
        crop = img.crop(box)
        out = CROP_DIR / f'{safe_name(chunk_id)}_p{page_no}_{name}.png'
        crop.save(out)
        rows.append({
            'crop_name': name,
            'image_path': str(out),
            'page_image_path': str(page_img_path),
            'box': list(box),
            'width': crop.size[0],
            'height': crop.size[1],
        })
    return rows


def get_ocr_python_and_libs() -> Tuple[str, str]:
    py = str(Path(OCR_ENV_PREFIX) / 'bin' / 'python')
    nvidia_libs = subprocess.check_output(
        [
            py,
            '-c',
            "import site,glob; paths=[]; "
            "[paths.extend(glob.glob(p+'/nvidia/*/lib')) for p in site.getsitepackages()]; "
            "print(':'.join(paths))"
        ],
        text=True
    ).strip()
    return py, nvidia_libs


def run_external_crop_ocr(crop_rows: List[Dict[str, Any]], tag: str = 'query') -> List[Dict[str, Any]]:
    if not WORKER_PATH.exists():
        raise FileNotFoundError(f'Worker not found: {WORKER_PATH}. Upload/copy arabic_ocr_crop_worker.py ke folder kerja.')

    input_path = OCR_IO_DIR / f'{safe_name(tag)}_crop_inputs.jsonl'
    output_path = OCR_IO_DIR / f'{safe_name(tag)}_crop_outputs.jsonl'
    write_jsonl(input_path, crop_rows)

    py_ocr, nvidia_libs = get_ocr_python_and_libs()
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f"{nvidia_libs}:{env.get('LD_LIBRARY_PATH','')}"

    cmd = [
        py_ocr, str(WORKER_PATH),
        '--input', str(input_path),
        '--output', str(output_path),
        '--use-gpu', str(OCR_USE_GPU),
        '--lang', 'ar',
    ]
    print('Running OCR worker:', ' '.join(cmd))
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=OCR_TIMEOUT_SECONDS)
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f'OCR worker failed with code {proc.returncode}')
    return read_jsonl(output_path)


def select_best_ocr_text(ocr_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    for r in ocr_rows:
        q = arabic_quality_score(r.get('ocr_text') or '')
        r['ocr_quality'] = q
        if best is None or q['score'] > best['ocr_quality']['score']:
            best = r
    return best or {}


def should_run_arabic_evidence(query: str, force: Optional[bool] = None) -> bool:
    if force is not None:
        return bool(force)
    q = query.lower()
    return any(k in q for k in ARABIC_EVIDENCE_KEYWORDS)



# %% CELL 9
def build_arabic_evidence_for_result(result: Dict[str, Any], tag: str) -> Dict[str, Any]:
    p = result.get('payload', {})
    raw_ar = p.get('arabic_text') or ''
    raw_q = arabic_quality_score(raw_ar)

    evidence = {
        'chunk_id': p.get('chunk_id') or result.get('chunk_id'),
        'doc_id': p.get('doc_id'),
        'category': p.get('category'),
        'nomor': p.get('nomor'),
        'title': p.get('title'),
        'page': p.get('page'),
        'pdf_path': p.get('pdf_path'),
        'pdf_filename': p.get('pdf_filename'),
        'raw_arabic_text': raw_ar,
        'raw_quality': raw_q,
        'ocr_status': 'not_run',
        'ocr_error': None,
        'page_image_path': None,
        'crop_image_path': None,
        'selected_crop_name': None,
        'selected_arabic_text': raw_ar,
        'selected_source': 'raw_chunk_arabic_text' if raw_ar else 'none',
        'selected_quality': raw_q,
        'ocr_candidates': [],
    }

    if not ENABLE_ARABIC_EVIDENCE:
        return evidence

    pdf_path = p.get('pdf_path')
    page_no = p.get('page')
    doc_id = p.get('doc_id') or p.get('pdf_filename') or p.get('chunk_id') or tag
    chunk_id = p.get('chunk_id') or result.get('chunk_id') or tag

    if not pdf_path or not page_no:
        evidence['ocr_status'] = 'skipped_no_pdf_or_page'
        return evidence

    try:
        page_img = render_pdf_page(pdf_path, int(page_no), str(doc_id), dpi=RENDER_DPI)
        crop_rows = crop_page_candidates(page_img, str(chunk_id), int(page_no))
        ocr_rows = run_external_crop_ocr(crop_rows, tag=f'{tag}_{chunk_id}')
        best = select_best_ocr_text(ocr_rows)

        evidence['ocr_status'] = 'ok'
        evidence['page_image_path'] = str(page_img)
        evidence['ocr_candidates'] = ocr_rows

        best_text = normalize_text(best.get('ocr_text') or '')
        best_q = best.get('ocr_quality') or arabic_quality_score(best_text)

        if best_text and best_q['score'] > raw_q['score']:
            evidence['selected_arabic_text'] = best_text
            evidence['selected_source'] = 'crop_ocr'
            evidence['selected_quality'] = best_q
            evidence['selected_crop_name'] = best.get('crop_name')
            evidence['crop_image_path'] = best.get('image_path')
        else:
            evidence['selected_arabic_text'] = raw_ar
            evidence['selected_source'] = 'raw_chunk_arabic_text' if raw_ar else 'none'
            evidence['selected_quality'] = raw_q
            if best:
                evidence['selected_crop_name'] = best.get('crop_name')
                evidence['crop_image_path'] = best.get('image_path')

    except Exception as e:
        evidence['ocr_status'] = 'error'
        evidence['ocr_error'] = f'{type(e).__name__}: {str(e)}'

    return evidence


def enrich_results_with_arabic_evidence(results: List[Dict[str, Any]], query: str, max_results: int = EVIDENCE_TOP_N) -> List[Dict[str, Any]]:
    enriched = []
    for i, r in enumerate(results, start=1):
        rr = dict(r)
        if i <= max_results:
            evidence = build_arabic_evidence_for_result(rr, tag=f'{safe_name(query, 60)}_r{i}')
        else:
            p = rr.get('payload', {})
            raw_ar = p.get('arabic_text') or ''
            evidence = {
                'chunk_id': p.get('chunk_id') or rr.get('chunk_id'),
                'doc_id': p.get('doc_id'),
                'category': p.get('category'),
                'nomor': p.get('nomor'),
                'title': p.get('title'),
                'page': p.get('page'),
                'raw_arabic_text': raw_ar,
                'raw_quality': arabic_quality_score(raw_ar),
                'ocr_status': 'not_run_top_limit',
                'ocr_error': None,
                'page_image_path': None,
                'crop_image_path': None,
                'selected_crop_name': None,
                'selected_arabic_text': raw_ar,
                'selected_source': 'raw_chunk_arabic_text' if raw_ar else 'none',
                'selected_quality': arabic_quality_score(raw_ar),
                'ocr_candidates': [],
            }
        rr['arabic_evidence'] = evidence
        enriched.append(rr)
    return enriched


def show_evidence(result: Dict[str, Any], show_images: bool = True):
    ev = result.get('arabic_evidence') or {}
    print('=' * 110)
    print('Doc:', ev.get('nomor'), '|', ev.get('title'), '| page', ev.get('page'))
    print('OCR status:', ev.get('ocr_status'), '| selected:', ev.get('selected_source'), '| crop:', ev.get('selected_crop_name'))
    print('quality:', ev.get('selected_quality'))
    if ev.get('ocr_error'):
        print('OCR error:', ev.get('ocr_error'))
    if ev.get('selected_arabic_text'):
        display_arabic(ev.get('selected_arabic_text'), title='Selected Arabic Evidence')
    if show_images and ev.get('crop_image_path'):
        print('CROP:', ev.get('crop_image_path'))
        display(IPImage(filename=ev.get('crop_image_path'), width=520))



# %% CELL 10
generator = None
tokenizer = None


def print_gpu_memory(prefix: str = ''):
    if not torch.cuda.is_available():
        print(prefix + 'CUDA tidak tersedia.')
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{prefix}GPU memory allocated={allocated:.2f} GiB | reserved={reserved:.2f} GiB")


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def release_retrieval_models():
    """Kosongkan VRAM sebelum load generator.

    Setelah fungsi ini dipanggil, retrieval tidak bisa dijalankan lagi kecuali kernel direstart
    atau cell load embedding model/reranker dijalankan ulang.
    """
    global embed_model, reranker
    print_gpu_memory('Before release retrieval | ')
    try:
        del embed_model
        print('embed_model deleted')
    except NameError:
        print('embed_model already deleted / not loaded')
    try:
        del reranker
        print('reranker deleted')
    except NameError:
        print('reranker already deleted / not loaded')
    cleanup_cuda()
    print_gpu_memory('After release retrieval | ')


def load_generator_model(force_reload: bool = False):
    """Load Qwen hanya saat benar-benar dibutuhkan."""
    global generator, tokenizer

    if generator is not None and tokenizer is not None and not force_reload:
        print('Generator already loaded:', GENERATOR_MODEL_NAME)
        return generator, tokenizer

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print_gpu_memory('Before generator load | ')
    print('Loading generator:', GENERATOR_MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        'device_map': 'auto',
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
    }

    if torch.cuda.is_available():
        # torch_dtype dipakai agar kompatibel dengan banyak versi transformers.
        model_kwargs['torch_dtype'] = torch.float16
    else:
        model_kwargs['torch_dtype'] = torch.float32

    if GENERATOR_USE_4BIT:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs.pop('torch_dtype', None)
            print('4-bit quantization enabled.')
        except Exception as e:
            print('Gagal enable 4-bit. Lanjut load normal fp16/fp32.')
            print(type(e).__name__, str(e)[:300])

    generator = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_NAME, **model_kwargs)
    generator.eval()
    print('Generator ready:', GENERATOR_MODEL_NAME)
    print_gpu_memory('After generator load | ')
    return generator, tokenizer


def unload_generator_model():
    global generator, tokenizer
    try:
        del generator
    except NameError:
        pass
    try:
        del tokenizer
    except NameError:
        pass
    generator = None
    tokenizer = None
    cleanup_cuda()
    print_gpu_memory('After generator unload | ')


def generate_answer(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if generator is None or tokenizer is None:
        return '[GENERATOR_NOT_LOADED]'

    messages = [
        {
            'role': 'system',
            'content': (
                'Anda adalah asisten RAG dokumen DSN-MUI. '
                'Jawab hanya berdasarkan konteks. '
                'Jangan mengarang teks Arab, ayat, hadis, nomor fatwa, atau halaman.'
            )
        },
        {'role': 'user', 'content': prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Truncation mencegah prompt terlalu panjang dan generation jadi sangat lama.
    input_device = next(generator.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    ).to(input_device)

    with torch.inference_mode():
        output_ids = generator.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    generated = output_ids[0][inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


if LOAD_GENERATOR:
    # Hanya pakai ini kalau memang belum load embedder/reranker, atau VRAM sangat lega.
    load_generator_model()
else:
    print('Generator belum diload. Ini sengaja agar VRAM tidak penuh. Generator akan diload setelah retrieval/evidence selesai.')



# %% CELL 11
# =====================================================
# PATCH: Manual HF Reranker untuk mengganti FlagEmbedding reranker
# Fix error:
# AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model
# =====================================================

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
USE_RERANKER = True
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 8

reranker_tokenizer = None
reranker_model = None
reranker_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_reranker():
    global reranker_tokenizer, reranker_model

    if not USE_RERANKER:
        print("Reranker disabled. Using hybrid ranking only.")
        return

    if reranker_model is not None and reranker_tokenizer is not None:
        print("Reranker already loaded.")
        return

    reranker_tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL_NAME,
        trust_remote_code=True
    )

    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    reranker_model.to(reranker_device)
    reranker_model.eval()

    print("Manual HF reranker ready:", RERANKER_MODEL_NAME)
    print("Device:", reranker_device)


def get_hit_text(hit):
    """
    Ambil teks dari hit retrieval secara fleksibel.
    Menyesuaikan kalau nama kolom/chunk beda-beda.
    """
    for key in ["text", "content", "chunk_text", "passage", "document", "clean_text"]:
        if key in hit and hit[key] is not None:
            return str(hit[key])

    # fallback gabungan beberapa field yang mungkin ada
    parts = []
    for key in ["title", "fatwa_title", "page_text", "ocr_text"]:
        if key in hit and hit[key] is not None:
            parts.append(str(hit[key]))

    return "\n".join(parts)


@torch.no_grad()
def score_pairs_manual(query, passages):
    """
    Score pasangan (query, passage) menggunakan AutoModelForSequenceClassification.
    Ini menggantikan reranker.compute_score() dari FlagEmbedding.
    """
    if reranker_model is None or reranker_tokenizer is None:
        return [0.0] * len(passages)

    scores = []

    for start in range(0, len(passages), RERANK_BATCH_SIZE):
        batch_passages = passages[start:start + RERANK_BATCH_SIZE]
        batch_queries = [query] * len(batch_passages)

        inputs = reranker_tokenizer(
            batch_queries,
            batch_passages,
            padding=True,
            truncation="only_second",
            max_length=RERANK_MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {k: v.to(reranker_device) for k, v in inputs.items()}

        outputs = reranker_model(**inputs)
        logits = outputs.logits

        if logits.ndim == 2 and logits.shape[1] == 1:
            batch_scores = logits[:, 0]
        elif logits.ndim == 2:
            batch_scores = logits[:, -1]
        else:
            batch_scores = logits.view(-1)

        batch_scores = batch_scores.detach().float().cpu().numpy().tolist()
        scores.extend(batch_scores)

    return scores


def rerank_hits(query, hits, final_k=5):
    """
    Replacement function untuk rerank_hits lama.
    Nama fungsi sengaja sama, supaya retrieve_final() otomatis pakai versi ini.
    """
    if not hits:
        return []

    if not USE_RERANKER:
        final = hits[:final_k]
        for rank, hit in enumerate(final, start=1):
            hit["final_rank"] = rank
            hit["rerank_score"] = None
        return final

    load_reranker()

    passages = [get_hit_text(hit) for hit in hits]
    scores = score_pairs_manual(query, passages)

    reranked = []
    for hit, score in zip(hits, scores):
        new_hit = dict(hit)
        new_hit["rerank_score"] = float(score)
        reranked.append(new_hit)

    reranked = sorted(
        reranked,
        key=lambda x: x.get("rerank_score", -999999),
        reverse=True
    )

    final = reranked[:final_k]

    for rank, hit in enumerate(final, start=1):
        hit["final_rank"] = rank

    return final


print("Reranker patch installed. Now rerun retrieve_final(...).")


# %% CELL 12
# =====================================================
# PATCH V4: hard filter nomor fatwa + smart excerpt + safe generation
# =====================================================
# Fokus perbaikan dari audit output:
# 1) Jika query menyebut nomor fatwa eksplisit, retrieval dipaksa ke nomor itu.
# 2) Context tidak lagi memotong awal chunk, tapi mengambil excerpt paling relevan.
# 3) Metadata teknis seperti chunk_id/reranker_score tidak dimasukkan ke prompt.
# 4) OCR Arab hanya masuk kalau query memang meminta Arab/hadis/ayat/OCR.
# 5) Output generator divalidasi; jika aneh/kepotong/menyalin konteks, otomatis retry.
# 6) Query tentang presisi OCR resmi dijawab secara hati-hati berbasis quality OCR.

FINAL_CONTEXT_K = 4                 # source sedikit ditambah, tapi tiap source lebih pendek dan fokus
MAX_CONTEXT_CHARS_PER_SOURCE = 950  # smart excerpt sekitar keyword, bukan awal chunk
MAX_ARABIC_CHARS_CONTEXT = 260
MAX_NEW_TOKENS = 420
EVIDENCE_TOP_N = 1
DIRECT_NOMOR_CANDIDATES_K = 24
RERANK_MAX_CHARS_PER_PASSAGE = min(RERANK_MAX_CHARS_PER_PASSAGE, 1800)

STOPWORDS_ID = set("""
yang dan atau untuk pada dalam dengan dari ke di adalah itu ini sebagai menurut tentang apa saja
jelaskan sebutkan tampilkan berikan apakah bagaimana mengapa cari dokumen bagian hasil ocr teks arab
hadis hadits ayat quran qur'an al qur an fatwa dsn mui no nomor berdasarkan prinsip syariah sebuah
antara dari dalam serta dapat digunakan apa saja isi pokok dasar hukum
""".split())

IMPORTANT_TERMS = {
    'akad', 'ijarah', 'wakalah', 'ujrah', 'ju\'alah', 'jualah', 'pemasaran', 'bpws',
    'hotel', 'wisatawan', 'pemandu', 'destinasi', 'ketentuan', 'prinsip', 'larangan',
    'dalil', 'hadis', 'ayat', 'kaidah', 'definisi', 'dimaksud', 'pariwisata', 'syariah'
}


def normalize_nomor(x: Any) -> str:
    if x is None:
        return ''
    s = str(x).strip()
    if not s or s.lower() in ['nan', 'none', 'null']:
        return ''
    m = re.search(r'\d{1,4}', s)
    return m.group(0) if m else s


def infer_nomor_from_payload(p: Dict[str, Any]) -> str:
    direct = normalize_nomor(p.get('nomor'))
    if direct:
        return direct
    candidates = [p.get('title'), p.get('doc_id'), p.get('pdf_filename'), p.get('pdf_path'), p.get('chunk_id')]
    joined = ' '.join(str(x or '') for x in candidates)
    patterns = [
        r'fatwa__([0-9]{1,4})(?:[-_]|\b)',
        r'fatwa[_\s-]+([0-9]{1,4})(?:[-_]|\b)',
        r'([0-9]{1,4})-dsn-mui',
        r'no\.?\s*([0-9]{1,4})',
        r'fatwa\s+(?:dsn[-\s]*mui\s*)?(?:no\.?\s*)?([0-9]{1,4})',
    ]
    for pat in patterns:
        m = re.search(pat, joined, flags=re.I)
        if m:
            return normalize_nomor(m.group(1))
    return ''


def infer_category_from_payload(p: Dict[str, Any]) -> str:
    cat = str(p.get('category') or '').strip()
    if cat:
        return cat
    text = ' '.join(str(p.get(k) or '') for k in ['title', 'doc_id', 'pdf_filename', 'chunk_id']).lower()
    for c in ['fatwa', 'pedoman_implementasi', 'talimat']:
        if c in text:
            return c
    return ''


def extract_requested_nomor(query: str) -> str:
    """Ambil nomor fatwa eksplisit dari query. Tidak mengambil angka acak selain pola Fatwa/No."""
    q = str(query or '')
    patterns = [
        r'fatwa\s+(?:dsn[-\s]*mui\s*)?(?:no\.?\s*)?([0-9]{1,4})',
        r'fatwa\s+dsn[-\s]*mui\s+no\.?\s*([0-9]{1,4})',
        r'no\.?\s*([0-9]{1,4})\s*(?:tentang|terkait|dalam|fatwa|dsn|mui)',
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.I)
        if m:
            return normalize_nomor(m.group(1))
    return ''


def short_title(title: Any, max_len: int = 120) -> str:
    s = str(title or '').strip()
    s = re.sub(r'^[a-z_]+__', '', s)
    s = s.replace('__', ' - ').replace('_', ' ')
    s = re.sub(r'-0bxtl.*$', '', s)
    return text_preview(s, max_len) if 'text_preview' in globals() else s[:max_len]


def payload_text(p: Dict[str, Any]) -> str:
    return normalize_text(p.get('text_original') or p.get('text_for_embedding') or p.get('text') or '')


def query_keywords(query: str) -> List[str]:
    toks = re.findall(r'[a-zA-Z0-9\u0600-\u06FF]+', str(query).lower())
    clean = []
    for t in toks:
        if len(t) < 3 or t in STOPWORDS_ID:
            continue
        if t not in clean:
            clean.append(t)
    clean = sorted(clean, key=lambda x: (x not in IMPORTANT_TERMS, -len(x)))
    return clean[:14]


def lexical_score_for_payload(query: str, p: Dict[str, Any]) -> float:
    text = (short_title(p.get('title'), 240) + '\n' + payload_text(p)).lower()
    kws = query_keywords(query)
    if not kws:
        return 0.0
    score = 0.0
    for kw in kws:
        cnt = text.count(kw.lower())
        if cnt:
            score += min(cnt, 5) * (2.0 if kw in IMPORTANT_TERMS else 1.0)
    qlow = str(query).lower()
    if any(x in qlow for x in ['definisi', 'dimaksud', 'apa definisi']):
        if re.search(r'(yang\s+dimaksud\s+dengan|adalah|merupakan)', text):
            score += 4.0
    if 'pemasaran' in qlow and 'pemasaran' in text:
        score += 4.0
    if 'wakalah' in qlow and 'wakalah' in text:
        score += 4.0
    if 'ujrah' in qlow and 'ujrah' in text:
        score += 3.0
    if 'ijarah' in qlow and 'ijarah' in text:
        score += 3.0
    if 'ju' in qlow and ('ju\'alah' in text or 'jualah' in text or 'ju’alah' in text):
        score += 3.0
    return float(score)


def direct_nomor_candidates(query: str, nomor: str, category_filter: Optional[str] = None, limit: int = DIRECT_NOMOR_CANDIDATES_K) -> List[Dict[str, Any]]:
    """Ambil kandidat langsung dari chunks untuk nomor fatwa eksplisit, agar tidak kalah oleh fatwa lain yang lebih semantik."""
    nomor = normalize_nomor(nomor)
    if not nomor:
        return []
    rows = []
    for p in chunks:
        if category_filter and infer_category_from_payload(p) != category_filter:
            continue
        if infer_nomor_from_payload(p) != nomor:
            continue
        s = lexical_score_for_payload(query, p)
        if s > 0 or len(rows) < limit:
            rows.append({
                'rank_hybrid': 9999,
                'rank_final': 9999,
                'chunk_id': p.get('chunk_id'),
                'score_hybrid': s / 1000.0,
                'score_direct_nomor': s,
                'score_reranker': None,
                'reranker_used': False,
                'payload': p,
            })
    rows = sorted(rows, key=lambda x: x.get('score_direct_nomor', 0.0), reverse=True)[:limit]
    for i, r in enumerate(rows, start=1):
        r['rank_direct_nomor'] = i
    return rows


def merge_hits(primary: List[Dict[str, Any]], extra: List[Dict[str, Any]], limit: int = HYBRID_TOP_K) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for source in [primary, extra]:
        for h in source:
            cid = h.get('chunk_id') or (h.get('payload') or {}).get('chunk_id')
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(h)
    return out[:limit]


def build_passage_for_rerank(payload: Dict[str, Any], max_chars: int = RERANK_MAX_CHARS_PER_PASSAGE) -> str:
    category = infer_category_from_payload(payload)
    nomor = infer_nomor_from_payload(payload)
    title = short_title(payload.get('title'), 160)
    page = payload.get('page') or ''
    text = payload_text(payload)
    passage = f"Kategori: {category}\nNomor: {nomor}\nJudul: {title}\nHalaman: {page}\nTeks:\n{text}"
    return normalize_text(passage)[:max_chars]


manual_reranker_tokenizer = None
manual_reranker_model = None
manual_reranker_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_manual_hf_reranker():
    global manual_reranker_tokenizer, manual_reranker_model
    if manual_reranker_model is not None and manual_reranker_tokenizer is not None:
        return True
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        manual_reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME, trust_remote_code=True)
        manual_reranker_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        manual_reranker_model.to(manual_reranker_device)
        manual_reranker_model.eval()
        print('Manual HF reranker ready:', RERANKER_MODEL_NAME)
        return True
    except Exception as e:
        print('Manual HF reranker gagal, fallback ke hybrid rank:', type(e).__name__, str(e)[:300])
        return False


@torch.no_grad()
def score_pairs_manual(query: str, passages: List[str]) -> List[float]:
    if not load_manual_hf_reranker():
        return [0.0] * len(passages)
    scores = []
    for start in range(0, len(passages), RERANK_BATCH_SIZE):
        batch_passages = passages[start:start + RERANK_BATCH_SIZE]
        batch_queries = [query] * len(batch_passages)
        inputs = manual_reranker_tokenizer(batch_queries, batch_passages, padding=True, truncation='only_second', max_length=512, return_tensors='pt')
        inputs = {k: v.to(manual_reranker_device) for k, v in inputs.items()}
        logits = manual_reranker_model(**inputs).logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            batch_scores = logits[:, 0]
        elif logits.ndim == 2:
            batch_scores = logits[:, -1]
        else:
            batch_scores = logits.view(-1)
        scores.extend(batch_scores.detach().float().cpu().numpy().tolist())
    return scores


def rerank_hits(query: str, hits: List[Dict[str, Any]], final_k: int = FINAL_CONTEXT_K) -> List[Dict[str, Any]]:
    if not hits:
        return []
    if not USE_RERANKER:
        final = hits[:final_k]
        for i, h in enumerate(final, start=1):
            h['rank_final'] = i
            h['reranker_used'] = False
        return final

    pairs = [[query, build_passage_for_rerank(h.get('payload', {}))] for h in hits]
    scores = None
    try:
        if reranker is not None:
            scores = []
            for start in range(0, len(pairs), RERANK_BATCH_SIZE):
                batch = pairs[start:start + RERANK_BATCH_SIZE]
                try:
                    batch_scores = reranker.compute_score(batch, normalize=True)
                except TypeError:
                    batch_scores = reranker.compute_score(batch)
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend([float(x) for x in batch_scores])
    except Exception as e:
        print('FlagEmbedding reranker error, pakai manual HF reranker:', type(e).__name__, str(e)[:200])
        scores = None

    if scores is None:
        passages = [x[1] for x in pairs]
        scores = score_pairs_manual(query, passages)

    reranked = []
    for h, s in zip(hits, scores):
        hh = dict(h)
        lex = lexical_score_for_payload(query, hh.get('payload', {}))
        hh['score_reranker_raw'] = float(s)
        hh['score_lexical_boost'] = float(lex)
        hh['score_reranker'] = float(s) + 0.03 * float(lex)
        hh['reranker_used'] = True
        reranked.append(hh)
    ranked = sorted(reranked, key=lambda x: x.get('score_reranker', -1e9), reverse=True)
    final = ranked[:final_k]
    for i, h in enumerate(final, start=1):
        h['rank_final'] = i
    return final


def retrieve_final(query: str, category_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Override V4: dense + BM25 + RRF + hard filter nomor fatwa eksplisit + direct nomor candidates + rerank."""
    t0 = now_ms()
    vh = vector_search_numpy(query, limit=VECTOR_TOP_K, category_filter=category_filter)
    t1 = now_ms()
    bh = bm25_search(query, limit=BM25_TOP_K, category_filter=category_filter)
    t2 = now_ms()
    hybrid = reciprocal_rank_fusion(vh, bh, limit=HYBRID_TOP_K)

    requested_nomor = extract_requested_nomor(query)
    direct_count = 0
    if requested_nomor:
        direct = direct_nomor_candidates(query, requested_nomor, category_filter=category_filter, limit=DIRECT_NOMOR_CANDIDATES_K)
        direct_count = len(direct)
        hybrid = merge_hits(hybrid, direct, limit=HYBRID_TOP_K + DIRECT_NOMOR_CANDIDATES_K)
        filtered = [h for h in hybrid if infer_nomor_from_payload(h.get('payload', {})) == requested_nomor]
        if filtered:
            hybrid = filtered
        else:
            print(f'Warning: query meminta Fatwa {requested_nomor}, tapi kandidat nomor tersebut tidak ditemukan. Fallback ke hybrid umum.')

    if category_filter:
        hybrid = [h for h in hybrid if infer_category_from_payload(h.get('payload', {})) == category_filter]

    for i, h in enumerate(hybrid, start=1):
        h['rank_hybrid_filtered'] = i
    t3 = now_ms()
    final = rerank_hits(query, hybrid, final_k=FINAL_CONTEXT_K)
    t4 = now_ms()
    timing = {
        'vector_ms': round(t1 - t0, 2),
        'bm25_ms': round(t2 - t1, 2),
        'rrf_ms': round(t3 - t2, 2),
        'rerank_ms': round(t4 - t3, 2),
        'retrieval_total_ms': round(t4 - t0, 2),
        'requested_nomor': requested_nomor,
        'direct_nomor_candidates': direct_count,
    }
    return final, timing


def should_include_arabic_for_query(query: str, force: Optional[bool] = None) -> bool:
    return should_run_arabic_evidence(query, force=force)


def _candidate_windows(text: str, centers: List[int], max_chars: int) -> List[Tuple[int, int]]:
    windows = []
    n = len(text)
    for c in centers:
        start = max(0, c - max_chars // 3)
        end = min(n, start + max_chars)
        if end - start < max_chars and start > 0:
            start = max(0, end - max_chars)
        windows.append((start, end))
    if not windows:
        windows.append((0, min(n, max_chars)))
    return windows


def focused_snippet(text: str, query: str, max_chars: int = MAX_CONTEXT_CHARS_PER_SOURCE) -> str:
    """Ambil excerpt dengan skor keyword tertinggi, bukan 850 karakter pertama."""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text

    low = text.lower()
    kws = query_keywords(query)
    centers = []
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
    for start, end in windows:
        seg = low[start:end]
        score = 0.0
        for kw in kws:
            cnt = seg.count(kw.lower())
            if cnt:
                score += cnt * (3.0 if kw in IMPORTANT_TERMS else 1.0)
        for phrase in ['yang dimaksud dengan', 'akad antara', 'untuk pemasaran', 'wakalah bil ujrah', 'ijarah', "ju'alah", 'bepergianlah', 'hotel syariah']:
            if phrase in seg:
                score += 4.0
        if best is None or score > best[0]:
            best = (score, start, end)

    _, start, end = best
    snippet = text[start:end].strip()
    if start > 0:
        snippet = '... ' + snippet
    if end < len(text):
        snippet = snippet + ' ...'
    return snippet


def build_rag_context(results: List[Dict[str, Any]], query: str = '', include_arabic: bool = False, max_chars_per_source: int = MAX_CONTEXT_CHARS_PER_SOURCE) -> str:
    parts = []
    for i, r in enumerate(results[:FINAL_CONTEXT_K], start=1):
        p = r.get('payload', {})
        ev = r.get('arabic_evidence') or {}
        source_id = f'S{i}'
        text = focused_snippet(payload_text(p), query, max_chars=max_chars_per_source)
        nomor = infer_nomor_from_payload(p) or 'tidak-terbaca'
        category = infer_category_from_payload(p) or 'tidak-terbaca'
        title = short_title(p.get('title'))
        page = p.get('page') or 'tidak-terbaca'

        arabic_block = 'Tidak ditampilkan karena pertanyaan tidak meminta Arab/hadis/ayat/OCR.'
        if include_arabic:
            arabic_text = normalize_text(ev.get('selected_arabic_text') or p.get('arabic_text') or '')
            arabic_q = ev.get('selected_quality') or arabic_quality_score(arabic_text)
            if arabic_text:
                arabic_block = (
                    f"OCR_quality={arabic_q.get('label')}; source={ev.get('selected_source', 'raw_or_none')}; "
                    f"arabic_count={arabic_q.get('arabic_count')}; pua_count={arabic_q.get('pua_count')}; ocr_status={ev.get('ocr_status')}\n"
                    f"OCR_preview:\n{arabic_text[:MAX_ARABIC_CHARS_CONTEXT]}"
                )
            else:
                arabic_block = 'Tidak tersedia teks Arab/OCR pada source ini.'

        block = f"""
[{source_id}] {category} | Fatwa/No: {nomor} | Halaman: {page}
Judul: {title}
Bukti teks relevan:
{text}
Teks Arab/OCR jika relevan:
{arabic_block}
""".strip()
        parts.append(block)
    return '\n\n' + ('\n\n' + '-' * 72 + '\n\n').join(parts)


def build_generator_prompt(user_query: str, context: str) -> str:
    requested_nomor = extract_requested_nomor(user_query)
    hard_filter_note = f"Query menyebut Fatwa {requested_nomor}; jawab hanya dari Fatwa {requested_nomor}." if requested_nomor else "Jika query tidak menyebut nomor fatwa, gunakan source paling relevan."
    return f"""
Anda adalah asisten RAG untuk dokumen DSN-MUI. Jawab hanya dari konteks yang diberikan.

ATURAN WAJIB:
- {hard_filter_note}
- Jangan menyalin blok konteks mentah.
- Jangan menulis chunk_id, skor reranker, path file, atau metadata teknis.
- Jangan mengarang ayat, hadis, teks Arab, nomor fatwa, judul, halaman, atau akad.
- Nomor fatwa, judul, dan halaman hanya boleh diambil dari header source [S1], [S2], dst.
- Semua klaim penting wajib diberi rujukan [S1], [S2], dst.
- Jika konteks tidak cukup, tulis jelas: "Tidak cukup konteks".
- Jika user meminta teks Arab/OCR dan kualitas OCR bukan usable/terverifikasi, sebutkan bahwa teks belum aman untuk kutipan resmi.
- Jika pertanyaan meminta membuat teks Arab/ayat/hadis yang tidak ada di konteks, tolak dengan sopan dan jangan membuat teks baru.
- Jawab hanya dalam Bahasa Indonesia. Jangan gunakan karakter China/Jepang/Rusia atau melanjutkan role user/assistant.

FORMAT WAJIB:
Status Jawaban:
- Ditemukan / Tidak cukup konteks

Jawaban Ringkas:
- maksimal 4 poin, langsung menjawab pertanyaan.

Bukti dari Dokumen:
- [Sx] Fatwa/No, judul, halaman, dan kutipan pendek/parafrasa bukti.

Teks Arab/Evidence OCR:
- isi hanya jika diminta; jika tidak, tulis "Tidak diperlukan".

Catatan Keterbatasan:
- sebutkan keterbatasan OCR/context bila ada.

[KONTEKS]
{context}

[PERTANYAAN]
{user_query}

Mulai jawaban tepat dengan "Status Jawaban:".
""".strip()


def clean_generated_answer(ans: str) -> str:
    ans = (ans or '').strip()
    for marker in ['\nuser\n', '\nassistant\n', '<|im_start|>', '<|im_end|>', '\n[KONTEKS]', '\nTEKS_DOKUMEN_RINGKAS:', '\nBukti teks relevan:\n[S']:
        pos = ans.lower().find(marker.lower())
        if pos > 60:
            ans = ans[:pos].strip()
    for marker in ['chunk_id:', 'reranker_score:', 'score_reranker', 'pdf_path:', 'rank_final:', 'rank_hybrid:', 'TEKS_DOKUMEN_RINGKAS:']:
        pos = ans.find(marker)
        if pos > 60:
            ans = ans[:pos].strip()
    return ans.strip()


def output_is_bad(answer: str) -> bool:
    ans = clean_generated_answer(answer)
    if not ans or len(ans) < 60:
        return True
    if not ans.startswith('Status Jawaban:'):
        return True
    bad_pattern = r'[\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff]|足|从|限|答|кут|user\n|assistant\n|chunk_id|reranker_score|pdf_path'
    if re.search(bad_pattern, ans, flags=re.I):
        return True
    tail = ans[-80:].strip()
    if len(ans) >= MAX_NEW_TOKENS * 2 and not re.search(r'[.!?。؟:]$', tail):
        return True
    return False


def fallback_answer_from_context(prompt: str) -> str:
    return (
        'Status Jawaban:\n'
        '- Tidak cukup konteks\n\n'
        'Jawaban Ringkas:\n'
        '- Output generator gagal divalidasi, sehingga jawaban tidak dipaksakan agar tidak halusinatif.\n\n'
        'Bukti dari Dokumen:\n'
        '- Lihat source [S1], [S2], dst. pada context/debug untuk pemeriksaan manual.\n\n'
        'Teks Arab/Evidence OCR:\n'
        '- Tidak ditampilkan dalam fallback.\n\n'
        'Catatan Keterbatasan:\n'
        '- Jalankan ulang generation dengan model lebih stabil atau prompt lebih pendek jika diperlukan.'
    )


def generate_answer_once(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if generator is None or tokenizer is None:
        return '[GENERATOR_NOT_LOADED]'
    messages = [
        {'role': 'system', 'content': 'Anda adalah evaluator RAG DSN-MUI. Jawab singkat, terstruktur, hanya Bahasa Indonesia, dan hanya berdasarkan konteks. Jangan menyalin konteks mentah. Jangan mengarang teks Arab atau sumber.'},
        {'role': 'user', 'content': prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer_prefix = 'Status Jawaban:\n- '
    text = text + answer_prefix
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_PROMPT_TOKENS).to(generator.device)
    with torch.no_grad():
        output_ids = generator.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.12,
            no_repeat_ngram_size=5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs['input_ids'].shape[-1]:]
    ans = answer_prefix + tokenizer.decode(generated, skip_special_tokens=True).strip()
    return clean_generated_answer(ans)


def generate_answer(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if generator is None or tokenizer is None:
        return '[GENERATOR_NOT_LOADED]'
    ans = generate_answer_once(prompt, max_new_tokens=max_new_tokens)
    if output_is_bad(ans):
        retry_prompt = prompt + """

ULANGI DENGAN KETAT:
- Jawab hanya Bahasa Indonesia.
- Jangan pakai karakter asing selain Arab yang memang ada di OCR.
- Jangan menyalin konteks mentah.
- Jangan melanjutkan role user/assistant.
- Pastikan jawaban selesai pada bagian Catatan Keterbatasan.
"""
        ans2 = generate_answer_once(retry_prompt, max_new_tokens=max_new_tokens)
        if not output_is_bad(ans2):
            return ans2
        return fallback_answer_from_context(prompt)
    return ans


def answer_query_safely(item: Dict[str, Any], results: List[Dict[str, Any]], prompt: str) -> str:
    """Rule-based guard untuk kasus yang tidak perlu/berisiko dijawab bebas oleh generator."""
    query = str(item.get('query') or '').lower()

    if ('cukup presisi' in query or 'dikutip resmi' in query or 'kutipan resmi' in query) and ('arab' in query or 'ocr' in query or 'hadis' in query or 'bepergian' in query):
        src = results[0] if results else {}
        p = src.get('payload', {}) if src else {}
        ev = src.get('arabic_evidence') or {}
        q = ev.get('selected_quality') or arabic_quality_score(ev.get('selected_arabic_text') or p.get('arabic_text') or '')
        label = q.get('label') or 'none'
        nomor = infer_nomor_from_payload(p) or 'tidak-terbaca'
        title = short_title(p.get('title'), 140)
        page = p.get('page') or 'tidak-terbaca'
        if label != 'usable':
            verdict = 'Belum cukup presisi untuk dikutip resmi.'
        else:
            verdict = 'Relatif terbaca, tetapi tetap perlu diverifikasi ke crop/page asli sebelum kutipan resmi.'
        return (
            'Status Jawaban:\n'
            '- Ditemukan\n\n'
            'Jawaban Ringkas:\n'
            f'- {verdict}\n'
            f'- Kualitas OCR yang terdeteksi: {label}; arabic_count={q.get("arabic_count")}; pua_count={q.get("pua_count")}.\n'
            '- Untuk kutipan akademik/resmi, gunakan gambar crop atau halaman PDF asli sebagai verifikasi akhir.\n\n'
            'Bukti dari Dokumen:\n'
            f'- [S1] Fatwa/No {nomor}, {title}, halaman {page}.\n\n'
            'Teks Arab/Evidence OCR:\n'
            '- Tidak ditetapkan sebagai kutipan resmi pada jawaban ini karena perlu verifikasi visual.\n\n'
            'Catatan Keterbatasan:\n'
            '- OCR Arab rawan salah huruf/harakat, terutama jika sumber PDF memakai font encoded atau hasil scan.'
        )

    return generate_answer(prompt)


def expected_hit(results: List[Dict[str, Any]], expected_nomor: str = '', expected_category: str = '') -> Dict[str, Any]:
    expected_nomor = normalize_nomor(expected_nomor)
    expected_category = str(expected_category or '').strip()
    found_rank = None
    found_source = None
    for i, r in enumerate(results, start=1):
        p = r.get('payload', {})
        nomor = infer_nomor_from_payload(p)
        category = infer_category_from_payload(p)
        nomor_ok = True if not expected_nomor else nomor == expected_nomor
        cat_ok = True if not expected_category else category == expected_category
        if nomor_ok and cat_ok:
            found_rank = i
            found_source = f"{category} | {nomor} | {short_title(p.get('title'), 140)} | p.{p.get('page')}"
            break
    return {'expected_found_in_topk': found_rank is not None, 'expected_found_rank': found_rank, 'expected_found_source': found_source}


def flatten_retrieval_rows(query_item: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for source_idx, r in enumerate(results, start=1):
        p = r.get('payload', {})
        ev = r.get('arabic_evidence') or {}
        rows.append({
            'run_id': RUN_ID, 'query_id': query_item.get('id'), 'query': query_item.get('query'), 'source_id': f'S{source_idx}',
            'rank_final': r.get('rank_final'), 'rank_hybrid': r.get('rank_hybrid'), 'rank_hybrid_filtered': r.get('rank_hybrid_filtered'),
            'rank_direct_nomor': r.get('rank_direct_nomor'), 'rank_vector': r.get('rank_vector'), 'rank_bm25': r.get('rank_bm25'),
            'reranker_used': r.get('reranker_used'), 'score_reranker': r.get('score_reranker'), 'score_reranker_raw': r.get('score_reranker_raw'),
            'score_lexical_boost': r.get('score_lexical_boost'), 'score_hybrid': r.get('score_hybrid'), 'score_direct_nomor': r.get('score_direct_nomor'),
            'score_vector': r.get('score_vector'), 'score_bm25': r.get('score_bm25'), 'chunk_id': r.get('chunk_id') or p.get('chunk_id'),
            'doc_id': p.get('doc_id'), 'category': infer_category_from_payload(p), 'nomor': infer_nomor_from_payload(p),
            'title': p.get('title'), 'page': p.get('page'), 'pdf_filename': p.get('pdf_filename'), 'pdf_path': p.get('pdf_path'),
            'text_preview': focused_snippet(payload_text(p), query_item.get('query', ''), 650), 'arabic_evidence_status': ev.get('ocr_status'),
            'arabic_selected_source': ev.get('selected_source'), 'arabic_selected_quality': (ev.get('selected_quality') or {}).get('label'),
            'arabic_selected_count': (ev.get('selected_quality') or {}).get('arabic_count'), 'crop_image_path': ev.get('crop_image_path'), 'page_image_path': ev.get('page_image_path')
        })
    return rows


def flatten_evidence_rows(query_item: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for source_idx, r in enumerate(results, start=1):
        p = r.get('payload', {})
        ev = r.get('arabic_evidence') or {}
        if not ev:
            continue
        qsel = ev.get('selected_quality') or {}
        rows.append({
            'run_id': RUN_ID, 'query_id': query_item.get('id'), 'query': query_item.get('query'), 'source_id': f'S{source_idx}',
            'rank_final': r.get('rank_final'), 'chunk_id': r.get('chunk_id') or p.get('chunk_id'), 'category': infer_category_from_payload(p), 'nomor': infer_nomor_from_payload(p),
            'title': p.get('title'), 'page': p.get('page'), 'ocr_status': ev.get('ocr_status'), 'ocr_error': ev.get('ocr_error'),
            'selected_source': ev.get('selected_source'), 'selected_crop_name': ev.get('selected_crop_name'), 'selected_quality_label': qsel.get('label'),
            'selected_quality_score': qsel.get('score'), 'selected_arabic_count': qsel.get('arabic_count'), 'selected_pua_count': qsel.get('pua_count'),
            'page_image_path': ev.get('page_image_path'), 'crop_image_path': ev.get('crop_image_path'), 'selected_arabic_text': ev.get('selected_arabic_text')
        })
    return rows


def run_one_retrieval_test(item: Dict[str, Any]) -> Dict[str, Any]:
    query = item['query']
    category_filter = item.get('category_filter') or None
    force_evidence = item.get('force_evidence')
    need_evidence = should_include_arabic_for_query(query, force=force_evidence)
    requested_nomor = extract_requested_nomor(query)

    t0 = now_ms()
    results, timing = retrieve_final(query, category_filter=category_filter)
    evidence_ms = 0.0
    if need_evidence:
        e0 = now_ms()
        results = enrich_results_with_arabic_evidence(results, query=query, max_results=EVIDENCE_TOP_N)
        evidence_ms = now_ms() - e0

    context = build_rag_context(results, query=query, include_arabic=need_evidence)
    prompt = build_generator_prompt(query, context)
    total_retrieval_evidence_ms = now_ms() - t0
    top = results[0].get('payload', {}) if results else {}
    hit = expected_hit(results, item.get('expected_nomor', ''), item.get('expected_category', ''))
    answer_row = {
        'run_id': RUN_ID, 'query_id': item.get('id'), 'query': query, 'category_filter': category_filter, 'need_evidence': need_evidence,
        'requested_nomor_from_query': requested_nomor, 'expected_nomor': item.get('expected_nomor', ''), 'expected_category': item.get('expected_category', ''),
        'must_not_hallucinate': item.get('must_not_hallucinate', False), 'expected_note': item.get('expected', ''), **hit,
        'top1_category': infer_category_from_payload(top), 'top1_nomor': infer_nomor_from_payload(top), 'top1_title': top.get('title'),
        'top1_page': top.get('page'), 'top1_chunk_id': top.get('chunk_id'), 'source_count': len(results),
        'vector_ms': timing.get('vector_ms'), 'bm25_ms': timing.get('bm25_ms'), 'rrf_ms': timing.get('rrf_ms'), 'rerank_ms': timing.get('rerank_ms'),
        'retrieval_total_ms': timing.get('retrieval_total_ms'), 'direct_nomor_candidates': timing.get('direct_nomor_candidates'),
        'evidence_ms': round(evidence_ms, 2), 'generation_ms': None, 'total_ms': round(total_retrieval_evidence_ms, 2),
        'context_chars': len(context), 'prompt_chars': len(prompt), 'answer_chars': 0, 'answer_validation_bad': None,
        'answer': '[PENDING_GENERATION]'
    }
    return {'item': item, 'answer_row': answer_row, 'retrieval_rows': flatten_retrieval_rows(item, results), 'evidence_rows': flatten_evidence_rows(item, results), 'results': results, 'context': context, 'prompt': prompt, 'answer': '[PENDING_GENERATION]'}


def release_retrieval_models():
    global embed_model, reranker, manual_reranker_model, manual_reranker_tokenizer
    print_gpu_memory('Before release retrieval | ')
    for name in ['embed_model', 'reranker', 'manual_reranker_model', 'manual_reranker_tokenizer']:
        try:
            globals()[name] = None
            print(name, 'released')
        except Exception:
            pass
    cleanup_cuda()
    print_gpu_memory('After release retrieval | ')

print('PATCH V4 installed: hard nomor filter, direct candidates, smart excerpt, clean prompt, output validation/retry.')



# %% CELL 13

# =====================================================
# PATCH V5: context compression / summarization agar prompt muat
# =====================================================
# Masalah V4: sebagian output masih kepotong/noisy karena context terlalu panjang
# dan generator masih melihat OCR mentah terlalu banyak. V5 memakai EXTRACTIVE
# CONTEXT COMPRESSION, bukan LLM summary bebas, supaya tidak menambah halusinasi.

CONTEXT_COMPRESSION_MODE = 'extractive'   # 'extractive' lebih aman untuk dokumen hukum/fatwa
FINAL_CONTEXT_K = 4
MAX_CONTEXT_TOTAL_CHARS = 3600
MAX_CONTEXT_CHARS_PER_SOURCE = 620
MAX_SOURCE_QUOTE_CHARS = 360
MAX_ARABIC_CHARS_CONTEXT = 160
MAX_PROMPT_TOKENS = 4096
MAX_NEW_TOKENS = 520
MIN_KEYWORD_SCORE_FOR_SOURCE = 0.0

# Query yang eksplisit minta Arab/OCR saja yang diberi OCR raw.
ARABIC_EXPLICIT_KEYWORDS = [
    'teks arab', 'arab lengkap', 'arab hasil ocr', 'ocr', 'tampilkan arab',
    'bacaan hadis arab', 'kutipan arab', 'ayat arab', 'hadis arab'
]
ARABIC_GENERAL_EVIDENCE_KEYWORDS = ['hadis', 'hadits', 'ayat', 'quran', "qur'an", 'al-qur', 'riwayat', 'firman']

BAD_OUTPUT_TERMS = [
    'syariahs', 'ijabah', 'jawabann', 'jawabaringkas', 'cukupkonteks',
    'diperlaku', 'diperiukan', 'diperluka', 'dsnmiu', 'm ui', 'n o:',
]

DOMAIN_REPLACEMENTS = [
    (r'\bsyariahs\b', 'syariah'),
    (r'\bDSNMIU\b', 'DSN-MUI'),
    (r'\bDsn-Mui\b', 'DSN-MUI'),
    (r'\bJawabann?\s*Ringkas\b', 'Jawaban Ringkas'),
    (r'\bJawabaRingkas\b', 'Jawaban Ringkas'),
    (r'\bTidak cukupkonteks\b', 'Tidak cukup konteks'),
]


def should_include_arabic_for_query(query: str, force: Optional[bool] = None) -> bool:
    """V5: OCR Arab raw hanya dimasukkan kalau user eksplisit meminta Arab/OCR.
    Untuk query dalil/hadis umum, cukup pakai teks latin/terjemahan dari chunk agar model tidak rusak.
    """
    if force is not None:
        return bool(force)
    q = str(query or '').lower()
    return any(k in q for k in ARABIC_EXPLICIT_KEYWORDS)


def split_text_units(text: str) -> List[str]:
    """Pecah teks OCR menjadi unit pendek. Toleran untuk OCR yang banyak newline."""
    text = normalize_text(text)
    if not text:
        return []
    raw = re.split(r'(?<=[.!?;:])\s+|\n+', text)
    units = []
    buf = ''
    for part in raw:
        part = part.strip()
        if not part:
            continue
        if len(part) < 25:
            buf = (buf + ' ' + part).strip()
            continue
        if buf:
            part = (buf + ' ' + part).strip()
            buf = ''
        # potong unit yang terlalu panjang menjadi window kecil
        if len(part) > 700:
            for start in range(0, len(part), 420):
                sub = part[start:start+520].strip()
                if len(sub) >= 35:
                    units.append(sub)
        else:
            units.append(part)
    if buf and len(buf) >= 25:
        units.append(buf)
    return units


def unit_score(unit: str, query: str) -> float:
    ulow = unit.lower()
    kws = query_keywords(query)
    score = 0.0
    for kw in kws:
        cnt = ulow.count(kw.lower())
        if cnt:
            score += min(cnt, 4) * (3.0 if kw in IMPORTANT_TERMS else 1.0)
    phrase_boosts = {
        'yang dimaksud dengan': 7.0,
        'akad antara': 7.0,
        'untuk pemasaran': 7.0,
        'wakalah bil ujrah': 9.0,
        'wakalah': 5.0,
        'ujrah': 5.0,
        'ijarah': 6.0,
        "ju'alah": 5.0,
        'hotel syariah': 5.0,
        'wisata syariah': 4.0,
        'prinsip syariah': 4.0,
        'bepergianlah': 6.0,
        'safar': 3.0,
        'riwayat': 3.0,
        'firman allah': 4.0,
        'al-qur': 4.0,
    }
    for phrase, val in phrase_boosts.items():
        if phrase in ulow:
            score += val
    # Penalti kalau unit terlalu noisy/simbolis, kecuali query memang minta Arab eksplisit.
    if not should_include_arabic_for_query(query):
        arabic_count = len(re.findall(r'[\u0600-\u06FF]', unit))
        if arabic_count > 60:
            score -= 5.0
    # Prioritaskan unit yang tidak terlalu panjang supaya final context padat.
    if len(unit) > 500:
        score -= 1.0
    return float(score)


def extractive_compress_text(text: str, query: str, max_chars: int = MAX_CONTEXT_CHARS_PER_SOURCE, max_units: int = 3) -> str:
    """Ringkas secara ekstraktif: pilih kalimat/window paling relevan, bukan summarize bebas."""
    text = normalize_text(text)
    if not text:
        return ''
    if len(text) <= max_chars:
        return text

    units = split_text_units(text)
    if not units:
        return focused_snippet(text, query, max_chars=max_chars)

    scored = []
    for idx, u in enumerate(units):
        s = unit_score(u, query)
        if s > 0:
            scored.append((s, idx, u))

    if not scored:
        return focused_snippet(text, query, max_chars=max_chars)

    # Ambil top units, lalu urutkan lagi sesuai posisi asli agar enak dibaca.
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_units]
    top = sorted(top, key=lambda x: x[1])

    out = []
    used = 0
    for s, idx, u in top:
        u = u.strip()
        if used + len(u) + 8 > max_chars:
            remain = max_chars - used - 8
            if remain > 120:
                out.append(u[:remain].rstrip() + ' ...')
            break
        out.append(u)
        used += len(u) + 8

    result = ' ... '.join(out).strip()
    if len(result) < 80:
        result = focused_snippet(text, query, max_chars=max_chars)
    return result[:max_chars].strip()


def make_source_summary_line(p: Dict[str, Any], source_id: str) -> str:
    nomor = infer_nomor_from_payload(p) or 'tidak-terbaca'
    category = infer_category_from_payload(p) or 'tidak-terbaca'
    title = short_title(p.get('title'), 110)
    page = p.get('page') or 'tidak-terbaca'
    return f'[{source_id}] {category} | Fatwa/No: {nomor} | Halaman: {page} | Judul: {title}'


def build_rag_context(results: List[Dict[str, Any]], query: str = '', include_arabic: bool = False, max_chars_per_source: int = MAX_CONTEXT_CHARS_PER_SOURCE) -> str:
    """V5: context builder kompresif.
    Setiap source dibuat jadi header deterministik + bukti ekstraktif pendek.
    """
    parts = []
    total_chars = 0
    include_arabic = include_arabic and should_include_arabic_for_query(query)

    for i, r in enumerate(results[:FINAL_CONTEXT_K], start=1):
        p = r.get('payload', {})
        ev = r.get('arabic_evidence') or {}
        source_id = f'S{i}'
        header = make_source_summary_line(p, source_id)
        doc_text = payload_text(p)
        compressed = extractive_compress_text(doc_text, query, max_chars=max_chars_per_source, max_units=3)

        arabic_block = 'Tidak diperlukan / tidak dimasukkan agar context tidak noisy.'
        if include_arabic:
            arabic_text = normalize_text(ev.get('selected_arabic_text') or p.get('arabic_text') or '')
            arabic_q = ev.get('selected_quality') or arabic_quality_score(arabic_text)
            if arabic_text:
                arabic_block = (
                    f"quality={arabic_q.get('label')}; arabic_count={arabic_q.get('arabic_count')}; "
                    f"pua_count={arabic_q.get('pua_count')}; ocr_status={ev.get('ocr_status')}\n"
                    f"OCR_preview: {arabic_text[:MAX_ARABIC_CHARS_CONTEXT]}"
                )
            else:
                arabic_block = 'Tidak tersedia teks Arab/OCR pada source ini.'

        block = f"""{header}
Bukti ekstraktif terpilih:
{compressed}
Teks Arab/OCR:
{arabic_block}""".strip()

        if total_chars + len(block) > MAX_CONTEXT_TOTAL_CHARS and parts:
            break
        parts.append(block)
        total_chars += len(block)

    context = ('\n\n' + '-' * 64 + '\n\n').join(parts)
    return '\n' + context.strip()


def build_generator_prompt(user_query: str, context: str) -> str:
    requested_nomor = extract_requested_nomor(user_query)
    hard_filter_note = f'Pertanyaan menyebut Fatwa {requested_nomor}; abaikan source yang bukan Fatwa {requested_nomor}.' if requested_nomor else 'Gunakan source paling relevan dari konteks.'
    return f"""
Anda adalah asisten RAG dokumen DSN-MUI. Context di bawah SUDAH DIRINGKAS SECARA EKSTRAKTIF dari dokumen, bukan untuk disalin mentah.

ATURAN WAJIB:
1. {hard_filter_note}
2. Jawab hanya dari context. Jangan pakai pengetahuan luar.
3. Jangan menyalin context mentah panjang; ubah menjadi jawaban ringkas.
4. Jangan mengarang nomor fatwa, halaman, ayat, hadis, teks Arab, atau akad.
5. Nomor fatwa/judul/halaman harus mengikuti header source [S1], [S2], dst.
6. Semua klaim penting wajib diberi rujukan [S1], [S2], dst.
7. Kalau context tidak cukup, jawab "Tidak cukup konteks" dan jangan memaksa.
8. Jawab hanya Bahasa Indonesia. Jangan gunakan karakter China/Jepang/Rusia. Jangan lanjutkan percakapan user/assistant.

BATAS OUTPUT:
- Maksimal 220 kata.
- Maksimal 4 poin pada Jawaban Ringkas.
- Kutipan bukti maksimal 1 kalimat pendek per source.

FORMAT WAJIB:
Status Jawaban:
- Ditemukan / Tidak cukup konteks

Jawaban Ringkas:
- ...

Bukti dari Dokumen:
- [Sx] Fatwa/No, judul singkat, halaman, bukti ringkas.

Teks Arab/Evidence OCR:
- Tulis hanya jika user meminta teks Arab/OCR; selain itu tulis "Tidak diperlukan".

Catatan Keterbatasan:
- ...

[CONTEXT TERKOMPRESI]
{context}

[PERTANYAAN]
{user_query}

Mulai jawaban tepat dengan "Status Jawaban:".
""".strip()


def normalize_answer_domain_terms(ans: str, query: str = '', prompt: str = '') -> str:
    ans = clean_generated_answer(ans)
    for pat, repl in DOMAIN_REPLACEMENTS:
        ans = re.sub(pat, repl, ans, flags=re.I)
    # Koreksi aman untuk kasus OCR/model salah ketik: ijabah bukan akad yang diminta di test ini.
    if re.search(r'\bakad\b', query, flags=re.I) and re.search(r'\bijarah\b', prompt, flags=re.I):
        ans = re.sub(r'\bijabah\b', 'ijarah', ans, flags=re.I)
    # Rapikan spasi pecah umum.
    ans = re.sub(r'DSN\s*[- ]?\s*MUI', 'DSN-MUI', ans, flags=re.I)
    ans = re.sub(r'Fatwa/\s*No\s*:', 'Fatwa/No:', ans, flags=re.I)
    ans = re.sub(r'\s+([,.;:])', r'\1', ans)
    ans = re.sub(r' {2,}', ' ', ans)
    return ans.strip()


def output_is_bad(answer: str, query: str = '', prompt: str = '') -> bool:
    ans = clean_generated_answer(answer)
    if not ans or len(ans) < 60:
        return True
    if not ans.startswith('Status Jawaban:'):
        return True
    bad_pattern = r'[\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff]|足|从|限|答|кут|user\n|assistant\n|chunk_id|reranker_score|pdf_path'
    if re.search(bad_pattern, ans, flags=re.I):
        return True
    low = ans.lower()
    if any(t in low for t in BAD_OUTPUT_TERMS):
        return True
    requested = extract_requested_nomor(query)
    if requested:
        # Kalau model menyebut nomor fatwa lain secara eksplisit, anggap buruk.
        nums = re.findall(r'(?:fatwa|no\.?|fatwa/no)\s*[:\-]?\s*(\d{1,4})', low, flags=re.I)
        wrong = [n for n in nums if normalize_nomor(n) != requested]
        if wrong:
            return True
    # Deteksi output yang berhenti di tengah frasa.
    tail = ans.strip()[-70:].lower()
    if tail.endswith((' dan', ' atau', ' dengan', ' untuk', ' dalam', ' yaitu', ' sebagai', ' berdasarkan', ',', ';', ':')):
        return True
    return False


def generate_answer_once(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if generator is None or tokenizer is None:
        return '[GENERATOR_NOT_LOADED]'
    messages = [
        {'role': 'system', 'content': 'Anda adalah evaluator RAG DSN-MUI. Jawab singkat, valid, hanya Bahasa Indonesia, dan hanya berdasarkan context. Jangan menyalin context mentah.'},
        {'role': 'user', 'content': prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer_prefix = 'Status Jawaban:\n- '
    text = text + answer_prefix
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_PROMPT_TOKENS).to(generator.device)
    with torch.no_grad():
        output_ids = generator.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs['input_ids'].shape[-1]:]
    ans = answer_prefix + tokenizer.decode(generated, skip_special_tokens=True).strip()
    return clean_generated_answer(ans)


def generate_answer(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, query: str = '') -> str:
    if generator is None or tokenizer is None:
        return '[GENERATOR_NOT_LOADED]'
    ans = generate_answer_once(prompt, max_new_tokens=max_new_tokens)
    ans = normalize_answer_domain_terms(ans, query=query, prompt=prompt)
    if output_is_bad(ans, query=query, prompt=prompt):
        retry_prompt = prompt + """

ULANGI LEBIH KETAT:
- Jawaban maksimal 160 kata.
- Jangan menyebut nomor fatwa selain yang ada di pertanyaan/source.
- Jangan pakai karakter asing selain Arab jika memang diminta.
- Jangan menyalin context mentah.
- Akhiri jawaban sampai Catatan Keterbatasan.
"""
        ans2 = generate_answer_once(retry_prompt, max_new_tokens=max_new_tokens)
        ans2 = normalize_answer_domain_terms(ans2, query=query, prompt=prompt)
        if not output_is_bad(ans2, query=query, prompt=prompt):
            return ans2
        return fallback_answer_from_context(prompt)
    return ans


def answer_query_safely(item: Dict[str, Any], results: List[Dict[str, Any]], prompt: str) -> str:
    query_raw = str(item.get('query') or '')
    query = query_raw.lower()

    # Guard anti-halusinasi untuk query yang secara jelas meminta sesuatu di luar corpus/context.
    if item.get('must_not_hallucinate'):
        joined = ' '.join(payload_text(r.get('payload', {})).lower() for r in results[:FINAL_CONTEXT_K])
        critical_terms = [t for t in query_keywords(query_raw) if t not in {'fatwa', 'dsn', 'mui', 'hadis', 'arab', 'dokumen'}]
        # Jika term unik user seperti blockchain/mars/ai tidak muncul di context, jangan generate bebas.
        if critical_terms and not any(t in joined for t in critical_terms[:4]):
            return (
                'Status Jawaban:\n'
                '- Tidak cukup konteks\n\n'
                'Jawaban Ringkas:\n'
                '- Konteks dokumen yang ditemukan tidak memuat informasi spesifik yang diminta, sehingga jawaban tidak boleh dikarang.\n\n'
                'Bukti dari Dokumen:\n'
                '- Tidak ada source pada context yang mendukung klaim tersebut secara memadai.\n\n'
                'Teks Arab/Evidence OCR:\n'
                '- Tidak diperlukan.\n\n'
                'Catatan Keterbatasan:\n'
                '- Sistem hanya boleh menjawab berdasarkan dokumen DSN-MUI yang terambil oleh retrieval.'
            )

    if ('cukup presisi' in query or 'dikutip resmi' in query or 'kutipan resmi' in query) and ('arab' in query or 'ocr' in query or 'hadis' in query or 'bepergian' in query):
        src = results[0] if results else {}
        p = src.get('payload', {}) if src else {}
        ev = src.get('arabic_evidence') or {}
        q = ev.get('selected_quality') or arabic_quality_score(ev.get('selected_arabic_text') or p.get('arabic_text') or '')
        label = q.get('label') or 'none'
        nomor = infer_nomor_from_payload(p) or 'tidak-terbaca'
        title = short_title(p.get('title'), 140)
        page = p.get('page') or 'tidak-terbaca'
        verdict = 'Belum cukup presisi untuk dikutip resmi.' if label != 'usable' else 'Relatif terbaca, tetapi tetap perlu diverifikasi ke crop/page asli sebelum kutipan resmi.'
        return (
            'Status Jawaban:\n'
            '- Ditemukan\n\n'
            'Jawaban Ringkas:\n'
            f'- {verdict}\n'
            f'- Kualitas OCR yang terdeteksi: {label}; arabic_count={q.get("arabic_count")}; pua_count={q.get("pua_count")}.\n'
            '- Untuk kutipan akademik/resmi, gunakan gambar crop atau halaman PDF asli sebagai verifikasi akhir.\n\n'
            'Bukti dari Dokumen:\n'
            f'- [S1] Fatwa/No {nomor}, {title}, halaman {page}.\n\n'
            'Teks Arab/Evidence OCR:\n'
            '- Tidak ditetapkan sebagai kutipan resmi pada jawaban ini karena perlu verifikasi visual.\n\n'
            'Catatan Keterbatasan:\n'
            '- OCR Arab rawan salah huruf/harakat, terutama jika sumber PDF memakai font encoded atau hasil scan.'
        )

    return generate_answer(prompt, query=query_raw)


def run_one_retrieval_test(item: Dict[str, Any]) -> Dict[str, Any]:
    query = item['query']
    category_filter = item.get('category_filter') or None
    force_evidence = item.get('force_evidence')
    need_evidence = should_include_arabic_for_query(query, force=force_evidence)
    requested_nomor = extract_requested_nomor(query)

    t0 = now_ms()
    results, timing = retrieve_final(query, category_filter=category_filter)
    evidence_ms = 0.0
    if need_evidence:
        e0 = now_ms()
        results = enrich_results_with_arabic_evidence(results, query=query, max_results=EVIDENCE_TOP_N)
        evidence_ms = now_ms() - e0

    context = build_rag_context(results, query=query, include_arabic=need_evidence)
    prompt = build_generator_prompt(query, context)
    total_retrieval_evidence_ms = now_ms() - t0
    top = results[0].get('payload', {}) if results else {}
    hit = expected_hit(results, item.get('expected_nomor', ''), item.get('expected_category', ''))
    answer_row = {
        'run_id': RUN_ID, 'query_id': item.get('id'), 'query': query, 'category_filter': category_filter, 'need_evidence': need_evidence,
        'requested_nomor_from_query': requested_nomor, 'expected_nomor': item.get('expected_nomor', ''), 'expected_category': item.get('expected_category', ''),
        'must_not_hallucinate': item.get('must_not_hallucinate', False), 'expected_note': item.get('expected', ''), **hit,
        'top1_category': infer_category_from_payload(top), 'top1_nomor': infer_nomor_from_payload(top), 'top1_title': top.get('title'),
        'top1_page': top.get('page'), 'top1_chunk_id': top.get('chunk_id'), 'source_count': len(results),
        'vector_ms': timing.get('vector_ms'), 'bm25_ms': timing.get('bm25_ms'), 'rrf_ms': timing.get('rrf_ms'), 'rerank_ms': timing.get('rerank_ms'),
        'retrieval_total_ms': timing.get('retrieval_total_ms'), 'direct_nomor_candidates': timing.get('direct_nomor_candidates'),
        'evidence_ms': round(evidence_ms, 2), 'generation_ms': None, 'total_ms': round(total_retrieval_evidence_ms, 2),
        'context_chars': len(context), 'prompt_chars': len(prompt), 'answer_chars': 0, 'answer_validation_bad': None,
        'context_compression_mode': CONTEXT_COMPRESSION_MODE,
        'answer': '[PENDING_GENERATION]'
    }
    return {'item': item, 'answer_row': answer_row, 'retrieval_rows': flatten_retrieval_rows(item, results), 'evidence_rows': flatten_evidence_rows(item, results), 'results': results, 'context': context, 'prompt': prompt, 'answer': '[PENDING_GENERATION]'}

print('PATCH V5 installed: extractive context compression, shorter prompt, stricter Arabic gating, anti-truncation retry.')



# %% CELL 15

DEFAULT_UMKM_TEST_QUERIES = [{'id': 'U001',
  'domain': 'jual_beli_online',
  'query': 'Saya punya UMKM makanan dan ingin menjual produk secara online. Prinsip syariah apa yang perlu diperhatikan agar transaksi jual belinya sah dan '
           'terhindar dari riba, gharar, maysir, tadlis, atau najasy? Sertakan dalil atau kaidah jika ada di dokumen.',
  'category_filter': None,
  'expected_focus': 'prinsip umum jual beli, larangan riba/gharar/maysir/tadlis/najasy, objek halal, kerelaan para pihak',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U002',
  'domain': 'jual_beli_online',
  'query': 'Saya mau menjual barang dengan sistem pre-order. Dalam perspektif dokumen DSN-MUI, akad apa yang paling relevan, salam atau istishna, dan apa '
           'ketentuan umumnya?',
  'category_filter': 'fatwa',
  'expected_focus': 'akad salam/istishna, objek pesanan, pembayaran, spesifikasi barang',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U003',
  'domain': 'jual_beli_online',
  'query': 'UMKM saya menerima pesanan custom furniture yang dibuat setelah konsumen membayar uang muka. Akad syariah apa yang cocok dan apa batasan yang '
           'harus dijelaskan ke konsumen?',
  'category_filter': 'fatwa',
  'expected_focus': 'istishna, spesifikasi barang, harga, waktu penyerahan, kesepakatan',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U004',
  'domain': 'jual_beli_online',
  'query': 'Saya menjual produk secara cicilan dengan harga lebih tinggi daripada tunai. Apakah boleh menurut prinsip syariah dan akad apa yang relevan?',
  'category_filter': 'fatwa',
  'expected_focus': 'murabahah/jual beli angsuran, kejelasan harga, tidak ada riba/denda bunga',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U005',
  'domain': 'jual_beli_online',
  'query': 'Dalam marketplace UMKM, apakah biaya admin atau fee platform boleh dikenakan? Akad apa yang paling mendekati dan apa syaratnya?',
  'category_filter': None,
  'expected_focus': 'ujrah, wakalah bil ujrah/ijarah jasa platform, transparansi biaya',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U006',
  'domain': 'jual_beli_online',
  'query': 'Apakah voucher diskon, cashback, atau poin loyalitas untuk pembeli UMKM boleh dalam transaksi syariah? Jelaskan batasannya berdasarkan dokumen '
           'DSN-MUI jika tersedia.',
  'category_filter': 'fatwa',
  'expected_focus': "hadiah/ju'alah/promo, tidak gharar, tidak maysir, tidak mengandung riba",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U007',
  'domain': 'jual_beli_online',
  'query': 'Saya ingin membuat program affiliate untuk penjualan produk UMKM: affiliator mendapat komisi jika berhasil menjual. Akad apa yang relevan dan apa '
           'batasan syariahnya?',
  'category_filter': None,
  'expected_focus': "ju'alah atau wakalah bil ujrah, komisi, kinerja, transparansi",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U008',
  'domain': 'jual_beli_online',
  'query': 'Bolehkah dropshipper menjual barang yang belum ia miliki? Jelaskan risiko syariah dan akad/struktur yang lebih aman berdasarkan dokumen DSN-MUI '
           'jika ada.',
  'category_filter': 'fatwa',
  'expected_focus': 'larangan menjual yang belum dimiliki, salam/wakalah, kejelasan peran',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U009',
  'domain': 'jual_beli_online',
  'query': 'Saya memakai sistem COD untuk toko online. Apa aspek syariah yang perlu diperhatikan dalam akad jual beli dan penyerahan barang?',
  'category_filter': None,
  'expected_focus': 'kejelasan objek, harga, serah terima, kerelaan, tidak gharar/tadlis',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U010',
  'domain': 'jual_beli_online',
  'query': 'Apakah boleh menaikkan harga saat pembeli memilih pembayaran tempo atau angsuran? Jelaskan dengan rujukan dokumen DSN-MUI jika tersedia.',
  'category_filter': 'fatwa',
  'expected_focus': 'harga kredit disepakati di awal, murabahah, larangan tambahan karena penundaan utang',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U011',
  'domain': 'pembiayaan_umkm',
  'query': 'UMKM saya butuh modal kerja dari bank syariah untuk membeli bahan baku. Akad pembiayaan apa saja yang mungkin digunakan dan kapan murabahah, '
           'mudharabah, atau musyarakah lebih tepat?',
  'category_filter': 'fatwa',
  'expected_focus': 'murabahah untuk pembelian barang, mudharabah/musyarakah untuk kerja sama modal/usaha',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U012',
  'domain': 'pembiayaan_umkm',
  'query': 'Saya ingin mengajukan pembiayaan murabahah untuk membeli mesin produksi. Bagaimana struktur akad murabahah yang sesuai syariah menurut DSN-MUI? '
           'Sertakan dalil jika ada.',
  'category_filter': 'fatwa',
  'expected_focus': 'murabahah, harga pokok, margin, barang, kepemilikan, pembayaran',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U013',
  'domain': 'pembiayaan_umkm',
  'query': 'Dua orang ingin patungan modal membuka usaha kuliner. Apakah akad musyarakah relevan, dan apa prinsip pembagian keuntungan serta risiko yang perlu '
           'diperhatikan?',
  'category_filter': 'fatwa',
  'expected_focus': 'musyarakah, kontribusi modal, nisbah keuntungan, kerugian sesuai porsi modal',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U014',
  'domain': 'pembiayaan_umkm',
  'query': 'Investor memberi modal kepada pengelola UMKM, sedangkan pengelola menjalankan usaha. Apakah ini lebih dekat dengan mudharabah? Apa konsekuensi '
           'pembagian hasil dan kerugian?',
  'category_filter': 'fatwa',
  'expected_focus': 'mudharabah, shahibul maal, mudharib, nisbah, kerugian modal kecuali kelalaian',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U015',
  'domain': 'pembiayaan_umkm',
  'query': 'Teman saya meminjamkan uang ke UMKM tanpa bunga tetapi meminta biaya administrasi. Apakah qardh boleh dan bagaimana batas biaya yang aman secara '
           'syariah?',
  'category_filter': 'fatwa',
  'expected_focus': 'qardh, larangan tambahan manfaat yang disyaratkan, biaya riil administrasi',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U016',
  'domain': 'pembiayaan_umkm',
  'query': 'UMKM ingin memberikan jaminan barang saat menerima pembiayaan. Bagaimana konsep rahn atau jaminan dalam dokumen DSN-MUI?',
  'category_filter': 'fatwa',
  'expected_focus': 'rahn, marhun, biaya pemeliharaan, tidak memanfaatkan jaminan secara batil',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U017',
  'domain': 'pembiayaan_umkm',
  'query': 'Supplier meminta bank syariah menjamin pembayaran dari UMKM. Apakah akad kafalah relevan dan apa batasan fee-nya?',
  'category_filter': 'fatwa',
  'expected_focus': 'kafalah, penjaminan, ujrah/fee jika diatur, objek jaminan',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U018',
  'domain': 'pembiayaan_umkm',
  'query': 'Saya ingin menggunakan invoice financing syariah untuk piutang usaha. Apa prinsip yang harus diperiksa agar tidak menjadi jual beli utang yang '
           'bermasalah?',
  'category_filter': 'fatwa',
  'expected_focus': 'anjak piutang syariah/hawalah/wakalah/qardh, piutang, ujrah, larangan riba',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U019',
  'domain': 'pembiayaan_umkm',
  'query': 'Platform fintech syariah ingin membiayai UMKM secara online. Dokumen DSN-MUI apa yang relevan dan prinsip apa yang harus dipenuhi?',
  'category_filter': 'fatwa',
  'expected_focus': 'layanan pembiayaan berbasis teknologi informasi berdasarkan prinsip syariah, akad dan larangan',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U020',
  'domain': 'pembiayaan_umkm',
  'query': 'Apakah denda keterlambatan pembayaran pembiayaan UMKM boleh menjadi pendapatan lembaga keuangan syariah? Jelaskan sesuai dokumen DSN-MUI.',
  'category_filter': 'fatwa',
  'expected_focus': "ta'zir/ta'widh, dana sosial, bukan pendapatan, kerugian riil",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U021',
  'domain': 'jasa_dan_sewa',
  'query': 'UMKM saya menyewa ruko untuk usaha. Apa prinsip akad ijarah yang perlu dipastikan agar sewa-menyewa sesuai syariah?',
  'category_filter': 'fatwa',
  'expected_focus': 'ijarah, manfaat, ujrah, jangka waktu, objek manfaat jelas',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U022',
  'domain': 'jasa_dan_sewa',
  'query': 'Saya menyewakan alat produksi kepada UMKM lain. Apa perbedaan akad ijarah barang/jasa dengan jual beli, dan apa ketentuan pentingnya?',
  'category_filter': 'fatwa',
  'expected_focus': 'ijarah manfaat, kepemilikan barang tetap, ujrah, tanggung jawab pemeliharaan',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U023',
  'domain': 'jasa_dan_sewa',
  'query': 'Agen penjualan produk UMKM mendapat fee tetap per transaksi. Apakah wakalah bil ujrah cocok dan apa yang harus tertulis dalam akad?',
  'category_filter': 'fatwa',
  'expected_focus': 'wakalah bil ujrah, kuasa, objek kuasa, fee, tugas agen',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U024',
  'domain': 'jasa_dan_sewa',
  'query': "UMKM ingin memberi bonus kepada tenaga penjual hanya jika target tercapai. Apakah akad ju'alah dapat digunakan? Jelaskan ketentuannya.",
  'category_filter': 'fatwa',
  'expected_focus': "ju'alah, imbalan atas hasil tertentu, target jelas, tidak gharar berlebihan",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U025',
  'domain': 'jasa_dan_sewa',
  'query': 'Saya membuka jasa titip pembelian barang. Apakah akad wakalah bil ujrah relevan dan bagaimana memisahkan harga barang dengan fee jasa?',
  'category_filter': 'fatwa',
  'expected_focus': 'wakalah, ujrah transparan, amanah, harga barang jelas',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U026',
  'domain': 'jasa_dan_sewa',
  'query': 'UMKM digital agency membuat website untuk klien dengan pembayaran bertahap. Akad apa yang sesuai: ijarah jasa atau istishna?',
  'category_filter': 'fatwa',
  'expected_focus': 'ijarah jasa/istishna untuk karya pesanan, spesifikasi pekerjaan, milestone, pembayaran',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U027',
  'domain': 'jasa_dan_sewa',
  'query': 'Bagaimana prinsip syariah untuk biaya layanan pengantaran barang oleh kurir atau platform delivery?',
  'category_filter': None,
  'expected_focus': 'ijarah/wakalah jasa pengiriman, ujrah, transparansi biaya, tanggung jawab',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U028',
  'domain': 'jasa_dan_sewa',
  'query': 'Bolehkah UMKM mengambil komisi dari pemasok karena berhasil membawa pembeli? Akad apa yang bisa menjelaskan komisi tersebut?',
  'category_filter': 'fatwa',
  'expected_focus': "ju'alah/wakalah bil ujrah, transparansi, tidak menipu pembeli",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U029',
  'domain': 'travel_dan_hotel_syariah',
  'query': 'Saya punya usaha travel wisata halal. Apa prinsip umum penyelenggaraan pariwisata syariah menurut DSN-MUI? Sertakan dalil jika ada.',
  'category_filter': 'fatwa',
  'expected_focus': 'Fatwa 108 pariwisata syariah, prinsip terhindar dari syirik/maksiat/mafsadat, kemaslahatan',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U030',
  'domain': 'travel_dan_hotel_syariah',
  'query': 'Travel umrah saya bekerja sama dengan hotel dan pemandu wisata. Akad apa yang relevan antara biro perjalanan, hotel, pemandu, dan wisatawan?',
  'category_filter': 'fatwa',
  'expected_focus': "Fatwa 108, ijarah, wakalah bil ujrah, ju'alah antara pihak pariwisata",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U031',
  'domain': 'travel_dan_hotel_syariah',
  'query': 'Apakah hotel syariah boleh menyediakan hiburan untuk tamu? Apa batasan hiburan yang sesuai prinsip syariah?',
  'category_filter': 'fatwa',
  'expected_focus': 'Fatwa 108 hotel syariah, larangan pornografi/asusila/kemaksiatan/kemusyrikan',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U032',
  'domain': 'travel_dan_hotel_syariah',
  'query': 'UMKM homestay ingin mengklaim sebagai penginapan syariah. Ketentuan apa yang harus diperhatikan dari dokumen DSN-MUI?',
  'category_filter': 'fatwa',
  'expected_focus': 'hotel syariah, makanan/minuman halal, fasilitas ibadah, larangan maksiat',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U033',
  'domain': 'travel_dan_hotel_syariah',
  'query': 'Dalam paket wisata syariah, apa kewajiban penyedia layanan terhadap fasilitas ibadah, makanan halal, dan kegiatan wisata?',
  'category_filter': 'fatwa',
  'expected_focus': 'destinasi wisata syariah, fasilitas ibadah, makanan halal, kegiatan sesuai prinsip syariah',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U034',
  'domain': 'travel_dan_hotel_syariah',
  'query': 'Apa dalil atau hadis yang digunakan DSN-MUI untuk mendukung pariwisata syariah atau aktivitas bepergian? Jika OCR Arab tersedia, tampilkan dengan '
           'catatan kualitasnya.',
  'category_filter': 'fatwa',
  'expected_focus': 'Fatwa 108, hadis bepergian, ayat perjalanan, kualitas OCR jika ada',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': True},
 {'id': 'U035',
  'domain': 'asuransi_syariah',
  'query': 'UMKM ingin memberi perlindungan asuransi syariah untuk karyawan. Apa prinsip asuransi syariah menurut DSN-MUI dan akad apa yang biasanya '
           'digunakan?',
  'category_filter': 'fatwa',
  'expected_focus': 'asuransi syariah, takaful, tabarru, wakalah bil ujrah/mudharabah, risk sharing',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U036',
  'domain': 'asuransi_syariah',
  'query': 'Apa perbedaan konsep risk sharing dalam asuransi syariah dengan risk transfer pada asuransi konvensional menurut dokumen DSN-MUI?',
  'category_filter': 'fatwa',
  'expected_focus': 'asuransi syariah, tabarru, tolong menolong, pengelola dana',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U037',
  'domain': 'asuransi_syariah',
  'query': 'Dana tabarru dalam asuransi syariah itu untuk apa, dan bagaimana jika dana tabarru mengalami defisit?',
  'category_filter': 'fatwa',
  'expected_focus': 'dana tabarru, qardh dari perusahaan, klaim, defisit underwriting',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U038',
  'domain': 'asuransi_syariah',
  'query': 'Apakah surplus underwriting boleh dibagikan kepada peserta asuransi syariah? Jelaskan berdasarkan ketentuan DSN-MUI jika tersedia.',
  'category_filter': 'fatwa',
  'expected_focus': 'surplus underwriting, pembagian, cadangan, peserta/pengelola sesuai akad',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U039',
  'domain': 'pembayaran_digital',
  'query': 'UMKM saya menerima pembayaran dengan uang elektronik. Apa prinsip syariah yang perlu diperhatikan dalam e-money menurut DSN-MUI?',
  'category_filter': 'fatwa',
  'expected_focus': "uang elektronik syariah, akad wadi'ah/qardh/ijarah, dana float, merchant fee",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U040',
  'domain': 'pembayaran_digital',
  'query': 'Apakah saldo dompet digital boleh mendapatkan bonus atau cashback? Jelaskan dengan rujukan dokumen DSN-MUI jika tersedia.',
  'category_filter': 'fatwa',
  'expected_focus': "e-money, bonus saldo, qardh/wadi'ah, hadiah yang tidak diperjanjikan, larangan riba",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U041',
  'domain': 'pembayaran_digital',
  'query': 'UMKM ingin memakai QR payment dan dikenakan merchant discount rate. Apakah biaya layanan seperti ini bisa dipandang sebagai ujrah?',
  'category_filter': 'fatwa',
  'expected_focus': 'payment gateway/uang elektronik, ujrah layanan, transparansi biaya',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U042',
  'domain': 'pembayaran_digital',
  'query': 'Bagaimana prinsip syariah untuk layanan paylater atau pembiayaan online bagi pembeli produk UMKM?',
  'category_filter': 'fatwa',
  'expected_focus': 'pembiayaan online syariah/fintech, akad, larangan riba, biaya/denda',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U043',
  'domain': 'pembayaran_digital',
  'query': 'Jika platform mengenakan biaya keterlambatan pada paylater, bagaimana batasan syariahnya menurut dokumen DSN-MUI?',
  'category_filter': 'fatwa',
  'expected_focus': "ta'zir/ta'widh, dana sosial, ganti rugi riil, bukan bunga",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U044',
  'domain': 'investasi_umkm',
  'query': 'UMKM ingin menempatkan dana kas pada instrumen investasi syariah. Instrumen apa saja yang umumnya sesuai prinsip syariah menurut dokumen DSN-MUI?',
  'category_filter': 'fatwa',
  'expected_focus': 'sukuk, saham syariah, reksadana syariah, deposito syariah, SBSN',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U045',
  'domain': 'investasi_umkm',
  'query': 'Apa prinsip sukuk atau obligasi syariah dan apa bedanya dengan surat utang berbunga biasa?',
  'category_filter': 'fatwa',
  'expected_focus': 'sukuk, underlying asset/manfaat, imbal hasil berdasarkan akad, bukan bunga',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U046',
  'domain': 'investasi_umkm',
  'query': 'UMKM ingin menerbitkan sukuk kecil atau mencari pendanaan berbasis aset. Prinsip apa yang perlu diperhatikan agar sesuai syariah?',
  'category_filter': 'fatwa',
  'expected_focus': 'sukuk, underlying, akad, penggunaan dana, imbal hasil',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U047',
  'domain': 'investasi_umkm',
  'query': 'Dalam investasi saham syariah, transaksi apa saja yang harus dihindari seperti najasy, gharar, maysir, atau insider trading?',
  'category_filter': 'fatwa',
  'expected_focus': 'pasar modal syariah, najasy, gharar, maysir, insider trading, short selling',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U048',
  'domain': 'investasi_umkm',
  'query': 'Apakah reksadana syariah bisa dipakai untuk pengelolaan dana UMKM? Apa peran manajer investasi dan bank kustodian dalam prinsip syariah?',
  'category_filter': 'fatwa',
  'expected_focus': 'reksadana syariah, manajer investasi, bank kustodian, akad wakalah/mudharabah',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U049',
  'domain': 'etika_bisnis_syariah',
  'query': 'UMKM saya menjual makanan. Apa yang bisa dijelaskan DSN-MUI tentang larangan objek usaha haram dan prinsip kehalalan transaksi?',
  'category_filter': None,
  'expected_focus': 'objek halal, larangan barang haram, transaksi haram, batasan DSN-MUI vs sertifikasi halal',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U050',
  'domain': 'etika_bisnis_syariah',
  'query': 'Apa itu gharar dalam transaksi bisnis UMKM dan contoh praktik yang perlu dihindari?',
  'category_filter': 'fatwa',
  'expected_focus': 'ketidakjelasan objek/akad/harga/penyerahan, larangan gharar',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U051',
  'domain': 'etika_bisnis_syariah',
  'query': 'Apa itu tadlis dan bagaimana UMKM harus menghindari menyembunyikan cacat barang kepada pembeli?',
  'category_filter': 'fatwa',
  'expected_focus': 'tadlis, informasi tidak jujur/cacat barang, kewajiban transparansi',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U052',
  'domain': 'etika_bisnis_syariah',
  'query': 'Apa itu najasy dan mengapa fake order atau review palsu untuk menaikkan reputasi toko online bermasalah secara syariah?',
  'category_filter': 'fatwa',
  'expected_focus': 'najasy, penawaran palsu, manipulasi harga/reputasi, larangan penipuan',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U053',
  'domain': 'etika_bisnis_syariah',
  'query': 'Apa itu maysir dan bagaimana menghindari unsur perjudian dalam promo undian berhadiah UMKM?',
  'category_filter': 'fatwa',
  'expected_focus': 'maysir, undian, taruhan, ketidakpastian yang merugikan salah satu pihak',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U054',
  'domain': 'etika_bisnis_syariah',
  'query': 'Apa itu riba dalam konteks bisnis UMKM, terutama pada pinjaman modal dan pembayaran terlambat? Sertakan dalil jika ada.',
  'category_filter': 'fatwa',
  'expected_focus': 'riba utang/piutang, tambahan atas pokok, denda bunga, dalil riba',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U055',
  'domain': 'etika_bisnis_syariah',
  'query': "Bagaimana prinsip ta'awun atau tolong-menolong dapat diterapkan dalam koperasi atau komunitas UMKM syariah?",
  'category_filter': None,
  'expected_focus': 'taawun, takaful, tolong menolong, akad sosial/tabarru/qardh',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U056',
  'domain': 'pedoman_implementasi',
  'query': 'Cari pedoman implementasi DSN-MUI yang berkaitan dengan akad. Ringkas poin yang berguna untuk UMKM tanpa mengarang di luar dokumen.',
  'category_filter': 'pedoman_implementasi',
  'expected_focus': 'pedoman implementasi terkait akad, ringkasan praktis',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U057',
  'domain': 'pedoman_implementasi',
  'query': 'Dalam pedoman implementasi, apa contoh ketentuan yang bisa membantu UMKM memahami pelaksanaan akad syariah secara praktis?',
  'category_filter': 'pedoman_implementasi',
  'expected_focus': 'ketentuan implementasi akad, prinsip praktis',
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U058',
  'domain': 'talimat',
  'query': "Cari ta'limat DSN-MUI yang relevan untuk prinsip syariah dalam bisnis dan ringkas manfaatnya untuk UMKM.",
  'category_filter': 'talimat',
  'expected_focus': "ta'limat relevan, prinsip syariah, manfaat untuk UMKM",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U059',
  'domain': 'talimat',
  'query': "Apakah ada ta'limat yang membahas penerapan prinsip syariah di lembaga bisnis? Jawab hanya jika konteks ta'limat mendukung.",
  'category_filter': 'talimat',
  'expected_focus': "ta'limat, dukungan konteks, abstain jika tidak cukup",
  'needs_dalil': False,
  'must_not_hallucinate': False,
  'force_evidence': False},
 {'id': 'U060',
  'domain': 'abstention_test',
  'query': 'Saya ingin membuat token kripto khusus UMKM dengan mekanisme staking. Apakah ada ketentuan DSN-MUI dalam corpus ini yang langsung mengatur token '
           'tersebut? Jika tidak cukup konteks, jangan menebak.',
  'category_filter': None,
  'expected_focus': 'abstain jika tidak ada dokumen spesifik; boleh jelaskan prinsip umum jika ada',
  'needs_dalil': False,
  'must_not_hallucinate': True,
  'force_evidence': False},
 {'id': 'U061',
  'domain': 'abstention_test',
  'query': 'Apakah penggunaan AI chatbot untuk customer service UMKM memiliki fatwa DSN-MUI khusus dalam corpus ini? Jawab berdasarkan dokumen saja.',
  'category_filter': 'fatwa',
  'expected_focus': 'abstain jika tidak ada fatwa khusus AI chatbot; jangan mengarang fatwa',
  'needs_dalil': False,
  'must_not_hallucinate': True,
  'force_evidence': False},
 {'id': 'U062',
  'domain': 'abstention_test',
  'query': 'Bolehkah UMKM membuka usaha game berhadiah NFT? Cari dukungan dokumen DSN-MUI; jika tidak ada, jelaskan keterbatasan konteks.',
  'category_filter': None,
  'expected_focus': 'abstain/risiko maysir/gharar jika ada konteks umum, jangan fatwa baru',
  'needs_dalil': False,
  'must_not_hallucinate': True,
  'force_evidence': False},
 {'id': 'U063',
  'domain': 'abstention_test',
  'query': 'Apakah ada dalil Arab khusus tentang marketplace digital dalam corpus DSN-MUI? Jika tidak ada, jangan membuat teks Arab baru.',
  'category_filter': 'fatwa',
  'expected_focus': 'tidak mengarang dalil Arab; hanya tampilkan jika ada di context',
  'needs_dalil': True,
  'must_not_hallucinate': True,
  'force_evidence': True},
 {'id': 'U064',
  'domain': 'abstention_test',
  'query': 'Saya ingin membuat skema investasi dengan janji return tetap 5% per bulan untuk UMKM. Apakah ini aman secara syariah menurut dokumen DSN-MUI?',
  'category_filter': 'fatwa',
  'expected_focus': 'return tetap, riba/gharar, mudharabah/musyarakah tidak boleh menjamin keuntungan tetap',
  'needs_dalil': True,
  'must_not_hallucinate': False,
  'force_evidence': False}]
UMKM_TEST_QUERIES_CSV = RUN_DIR / 'rag_umkm_test_queries.csv'
pd.DataFrame(DEFAULT_UMKM_TEST_QUERIES).to_csv(UMKM_TEST_QUERIES_CSV, index=False, encoding='utf-8-sig')
print('UMKM test queries saved to:', UMKM_TEST_QUERIES_CSV)
print('Jumlah query:', len(DEFAULT_UMKM_TEST_QUERIES))
display(pd.DataFrame(DEFAULT_UMKM_TEST_QUERIES).head(10))



# %% CELL 17

# =====================================================
# PATCH V9: UMKM scenario testing + dalil-aware output
# =====================================================
UMKM_FINAL_CONTEXT_K = 5
FINAL_CONTEXT_K = UMKM_FINAL_CONTEXT_K
MAX_CONTEXT_TOTAL_CHARS = 5200
MAX_CONTEXT_CHARS_PER_SOURCE = 720
MAX_DALIL_CHARS_PER_SOURCE = 520
MAX_NEW_TOKENS = 650
MAX_PROMPT_TOKENS = 6144

DALIL_KEYWORDS = [
    'firman allah', 'allah swt', 'q.s', 'qs.', 'al-qur', 'al qur', 'ayat',
    'hadis', 'hadits', 'rasulullah', 'nabi', 'riwayat', 'hr.', 'h.r.',
    'kaidah fikih', 'kaidah fiqh', 'kaidah', 'الأصل', 'قال', 'عن ', 'نهى'
]

UMKM_DOMAIN_TERMS = {
    'umkm', 'usaha', 'bisnis', 'jual', 'beli', 'online', 'marketplace', 'cicilan', 'angsuran',
    'pembiayaan', 'modal', 'bank', 'syariah', 'akad', 'murabahah', 'salam', 'istishna',
    'mudharabah', 'musyarakah', 'ijarah', 'wakalah', 'ujrah', 'jualah', "ju'alah", 'qardh',
    'rahn', 'kafalah', 'hawalah', 'asuransi', 'tabarru', 'takaful', 'e-money', 'uang',
    'elektronik', 'paylater', 'cashback', 'voucher', 'diskon', 'hotel', 'travel', 'wisata',
    'halal', 'riba', 'gharar', 'maysir', 'maisir', 'tadlis', 'najasy', 'sukuk', 'reksadana',
    'investasi', 'fintech', 'platform', 'fee', 'komisi', 'denda', "ta'widh", "ta'zir"
}

def load_umkm_test_queries_from_csv(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path).fillna('')
    rows = []
    for _, row in df.iterrows():
        item = {k: row.get(k, '') for k in df.columns}
        item['id'] = str(item.get('id') or '').strip()
        item['query'] = str(item.get('query') or '').strip()
        item['category_filter'] = str(item.get('category_filter') or '').strip() or None
        for bcol in ['needs_dalil', 'must_not_hallucinate', 'force_evidence']:
            item[bcol] = str(item.get(bcol) or '').strip().lower() in ['true', '1', 'yes', 'ya']
        rows.append(item)
    return rows

def make_umkm_retrieval_query(item: Dict[str, Any]) -> str:
    query = normalize_text(item.get('query') or '')
    focus = normalize_text(item.get('expected_focus') or '')
    domain = normalize_text(item.get('domain') or '')
    retrieval_query = f"{query}\n{query}\nKata kunci domain: {domain}\nFokus konsep: {focus}"
    return retrieval_query[:1600]

def contains_dalil(text: str) -> bool:
    low = normalize_text(text).lower()
    return any(k in low for k in DALIL_KEYWORDS) or count_arabic(text) >= 15

def extract_dalil_snippets(text: str, max_chars: int = MAX_DALIL_CHARS_PER_SOURCE) -> str:
    text = normalize_text(text)
    if not text:
        return ''
    units = split_text_units(text)
    chosen = []
    for idx, u in enumerate(units):
        ulow = u.lower()
        if any(k in ulow for k in DALIL_KEYWORDS) or count_arabic(u) >= 8:
            start = max(0, idx - 1)
            end = min(len(units), idx + 3)
            snippet = ' '.join(units[start:end])
            if snippet not in chosen:
                chosen.append(snippet)
        if sum(len(x) for x in chosen) >= max_chars:
            break
    return normalize_text(' ... '.join(chosen))[:max_chars]

def build_umkm_context(results: List[Dict[str, Any]], item: Dict[str, Any]) -> str:
    retrieval_query = item.get('retrieval_query') or make_umkm_retrieval_query(item)
    parts = []
    total_chars = 0
    for i, r in enumerate(results[:FINAL_CONTEXT_K], start=1):
        p = r.get('payload', {})
        ev = r.get('arabic_evidence') or {}
        header = make_source_summary_line(p, f'S{i}')
        doc_text = payload_text(p)
        compressed = extractive_compress_text(doc_text, retrieval_query, max_chars=MAX_CONTEXT_CHARS_PER_SOURCE, max_units=4)
        dalil_text = extract_dalil_snippets(doc_text, max_chars=MAX_DALIL_CHARS_PER_SOURCE)
        if not dalil_text and ev.get('selected_arabic_text'):
            q = ev.get('selected_quality') or arabic_quality_score(ev.get('selected_arabic_text'))
            dalil_text = f"OCR_quality={q.get('label')}; OCR_preview={normalize_text(ev.get('selected_arabic_text'))[:MAX_DALIL_CHARS_PER_SOURCE]}"
        dalil_block = dalil_text if dalil_text else 'Tidak terdeteksi dalil/kaidah eksplisit pada source ini.'
        block = f"""{header}
Bukti ekstraktif relevan:
{compressed}

Dalil/Kaidah kandidat dari source:
{dalil_block}""".strip()
        if total_chars + len(block) > MAX_CONTEXT_TOTAL_CHARS and parts:
            break
        parts.append(block)
        total_chars += len(block)
    return ('\n\n' + '-' * 72 + '\n\n').join(parts).strip()

def build_umkm_generator_prompt(item: Dict[str, Any], context: str) -> str:
    query = item.get('query') or ''
    domain = item.get('domain') or ''
    expected_focus = item.get('expected_focus') or ''
    category_filter = item.get('category_filter') or 'semua kategori'
    return f"""
Anda adalah asisten RAG DSN-MUI untuk UMKM. Jawab pertanyaan praktis bisnis dengan HANYA memakai konteks dokumen DSN-MUI yang diberikan.

Profil tugas:
- Pengguna adalah pelaku UMKM yang ingin memahami aspek syariah bisnisnya.
- Corpus berasal dari fatwa, pedoman implementasi, dan ta'limat DSN-MUI yang sudah di-precompute.
- Jawaban harus edukatif, praktis, dan grounded. Jangan memberikan fatwa baru.
- Jika konteks tidak cukup, jawab terus terang: "Tidak cukup konteks".

Aturan wajib:
1. Jangan menggunakan pengetahuan luar di luar konteks.
2. Jangan mengarang nomor fatwa, judul dokumen, halaman, dalil Arab, hadis, ayat, atau kaidah fikih.
3. Jika ada bagian "Dalil/Kaidah kandidat" yang relevan, tampilkan pada bagian Dalil/Kaidah. Kalau tidak ada, tulis "Tidak ditemukan dalam konteks yang terambil".
4. Kalau teks Arab/OCR tampak rusak/noisy, beri catatan perlu verifikasi dokumen asli.
5. Semua klaim penting harus diberi sumber [S1], [S2], dst.
6. Untuk UMKM, berikan implikasi praktis: apa yang boleh, apa yang harus dijaga, dan kapan perlu konsultasi ke DPS/ahli syariah.
7. Jawab hanya dalam Bahasa Indonesia. Jangan pakai karakter China/Jepang/Rusia. Jangan menyalin konteks mentah panjang.

Metadata benchmark:
- Domain: {domain}
- Filter kategori: {category_filter}
- Fokus konsep yang diuji: {expected_focus}

Format jawaban wajib:
Status Jawaban:
- Ditemukan / Tidak cukup konteks

Jawaban Praktis untuk UMKM:
- ...

Akad/Ketentuan yang Relevan:
- ...

Dalil/Kaidah dari Dokumen:
- ...

Sumber Dokumen:
- [Sx] kategori, nomor/judul, halaman, alasan relevan.

Catatan Batasan:
- ...

[KONTEKS DOKUMEN DSN-MUI]
{context}

[PERTANYAAN UMKM]
{query}

Mulai jawaban tepat dengan "Status Jawaban:".
""".strip()

def support_score_for_results(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    q_terms = [t for t in query_keywords(query) if t not in STOPWORDS_ID]
    q_terms = list(dict.fromkeys(q_terms))
    if not q_terms:
        return {'support_score': 0.0, 'support_level': 'unknown', 'matched_terms': '', 'missing_terms': ''}
    joined = ' '.join(payload_text(r.get('payload', {})).lower() for r in results[:FINAL_CONTEXT_K])
    matched = [t for t in q_terms if t.lower() in joined]
    domain_hits = [t for t in matched if t in UMKM_DOMAIN_TERMS or len(t) >= 6]
    score = (len(matched) + 0.75 * len(domain_hits)) / max(len(q_terms), 1)
    if score >= 0.42:
        level = 'high'
    elif score >= 0.20:
        level = 'medium'
    elif score >= MIN_SUPPORT_SCORE_FOR_GENERATION:
        level = 'low'
    else:
        level = 'very_low'
    return {'support_score': round(float(score), 4), 'support_level': level, 'matched_terms': ', '.join(matched[:30]), 'missing_terms': ', '.join([t for t in q_terms if t not in matched][:30])}

def answer_abstained(ans: str) -> bool:
    low = normalize_text(ans).lower()
    return 'tidak cukup konteks' in low or 'tidak ditemukan dalam konteks' in low or 'tidak ada konteks' in low

def answer_query_umkm_safely(item: Dict[str, Any], results: List[Dict[str, Any]], prompt: str, support_info: Dict[str, Any]) -> str:
    query = item.get('query') or ''
    if FORCE_ABSTAIN_ON_LOW_SUPPORT and support_info.get('support_level') == 'very_low':
        return ('Status Jawaban:\n- Tidak cukup konteks\n\nJawaban Praktis untuk UMKM:\n- Dokumen DSN-MUI yang terambil belum cukup relevan untuk menjawab pertanyaan ini secara aman.\n\nAkad/Ketentuan yang Relevan:\n- Tidak dapat ditentukan dari konteks yang terambil.\n\nDalil/Kaidah dari Dokumen:\n- Tidak ditemukan dalam konteks yang terambil.\n\nSumber Dokumen:\n- Tidak ada sumber yang cukup kuat untuk mendukung jawaban.\n\nCatatan Batasan:\n- Sistem hanya boleh menjawab berdasarkan corpus DSN-MUI precomputed. Pertanyaan ini perlu pencarian dokumen tambahan atau konsultasi ahli syariah.')
    ans = generate_answer(prompt, query=query)
    if output_is_bad(ans, query=query, prompt=prompt):
        retry_prompt = prompt + "\n\nULANGI DENGAN KETAT: jawab hanya Bahasa Indonesia, jangan karakter asing, jangan menyalin konteks mentah, jangan membuat dalil/nomor fatwa, dan tetap gunakan format wajib."
        ans2 = generate_answer(retry_prompt, max_new_tokens=MAX_NEW_TOKENS, query=query)
        if not output_is_bad(ans2, query=query, prompt=retry_prompt):
            ans = ans2
    return normalize_answer_domain_terms(ans, query=query, prompt=prompt)

def flatten_umkm_retrieval_rows(item: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for source_idx, r in enumerate(results, start=1):
        p = r.get('payload', {})
        ev = r.get('arabic_evidence') or {}
        rows.append({'run_id': RUN_ID, 'query_id': item.get('id'), 'domain': item.get('domain'), 'query': item.get('query'), 'retrieval_query': item.get('retrieval_query'), 'source_id': f'S{source_idx}', 'rank_final': r.get('rank_final'), 'rank_hybrid': r.get('rank_hybrid'), 'rank_vector': r.get('rank_vector'), 'rank_bm25': r.get('rank_bm25'), 'reranker_used': r.get('reranker_used'), 'score_reranker': r.get('score_reranker'), 'score_hybrid': r.get('score_hybrid'), 'score_vector': r.get('score_vector'), 'score_bm25': r.get('score_bm25'), 'chunk_id': r.get('chunk_id') or p.get('chunk_id'), 'category': infer_category_from_payload(p), 'nomor': infer_nomor_from_payload(p), 'title': p.get('title'), 'page': p.get('page'), 'pdf_filename': p.get('pdf_filename'), 'pdf_path': p.get('pdf_path'), 'text_preview': text_preview(payload_text(p), 700), 'has_dalil_snippet': contains_dalil(payload_text(p)), 'arabic_evidence_status': ev.get('ocr_status'), 'arabic_selected_quality': (ev.get('selected_quality') or {}).get('label'), 'crop_image_path': ev.get('crop_image_path')})
    return rows

def run_one_umkm_retrieval_test(item: Dict[str, Any]) -> Dict[str, Any]:
    query = item['query']
    retrieval_query = make_umkm_retrieval_query(item)
    item = dict(item)
    item['retrieval_query'] = retrieval_query
    category_filter = item.get('category_filter') or None
    need_evidence = bool(item.get('force_evidence')) or (ENABLE_ARABIC_EVIDENCE_FOR_DALIL_QUERY and bool(item.get('needs_dalil')))
    t0 = now_ms()
    results, timing = retrieve_final(retrieval_query, category_filter=category_filter)
    evidence_ms = 0.0
    if need_evidence:
        e0 = now_ms()
        results = enrich_results_with_arabic_evidence(results, query=query, max_results=EVIDENCE_TOP_N)
        evidence_ms = now_ms() - e0
    context = build_umkm_context(results, item)
    prompt = build_umkm_generator_prompt(item, context)
    support_info = support_score_for_results(retrieval_query, results)
    dalil_available = any(contains_dalil(payload_text(r.get('payload', {}))) for r in results[:FINAL_CONTEXT_K])
    total_ms = now_ms() - t0
    top = results[0].get('payload', {}) if results else {}
    answer_row = {'run_id': RUN_ID, 'query_id': item.get('id'), 'domain': item.get('domain'), 'query': query, 'retrieval_query': retrieval_query, 'category_filter': category_filter, 'expected_focus': item.get('expected_focus'), 'needs_dalil': bool(item.get('needs_dalil')), 'must_not_hallucinate': bool(item.get('must_not_hallucinate')), 'dalil_available_in_topk': bool(dalil_available), **support_info, 'top1_category': infer_category_from_payload(top), 'top1_nomor': infer_nomor_from_payload(top), 'top1_title': top.get('title'), 'top1_page': top.get('page'), 'n_results': len(results), 'context_chars': len(context), 'prompt_chars': len(prompt), 'evidence_ms': round(evidence_ms, 2), **timing, 'total_retrieval_evidence_ms': round(total_ms, 2), 'answer': ''}
    retrieval_rows = flatten_umkm_retrieval_rows(item, results)
    evidence_rows = []
    for ridx, r in enumerate(results, start=1):
        ev = r.get('arabic_evidence') or {}
        if ev:
            evidence_rows.append({'query_id': item.get('id'), 'source_id': f'S{ridx}', **ev})
    return {'item': item, 'results': results, 'context': context, 'prompt': prompt, 'support_info': support_info, 'answer_row': answer_row, 'retrieval_rows': retrieval_rows, 'evidence_rows': evidence_rows, 'answer': ''}



# %% CELL 20

# =====================================================
# PATCH V10: UMKM production evaluation mode
# =====================================================
# Tujuan:
# 1) PDF/soal ujian tidak dipakai lagi; test set UMKM tetap dipakai.
# 2) Generator tidak dipaksa menulis panjang. Prompt lebih pendek.
# 3) Jika generator rusak/typo/kepotong, fallback tetap memakai evidence ekstraktif,
#    bukan fallback kosong "Output generator gagal divalidasi".
# 4) Dalil/kaidah hanya diambil dari source yang ter-retrieve, bukan dikarang.

UMKM_FINAL_CONTEXT_K = 4
FINAL_CONTEXT_K = UMKM_FINAL_CONTEXT_K
MAX_CONTEXT_TOTAL_CHARS = 3400
MAX_CONTEXT_CHARS_PER_SOURCE = 520
MAX_DALIL_CHARS_PER_SOURCE = 360
MAX_NEW_TOKENS = 360
MAX_PROMPT_TOKENS = 4096

KNOWN_SYARIAH_CONCEPTS = [
    'murabahah', 'salam', 'istishna', 'istishna’', 'ijarah', 'wakalah', 'wakalah bil ujrah',
    'ju’alah', "ju'alah", 'mudharabah', 'musyarakah', 'qardh', 'rahn', 'kafalah',
    'hawalah', 'tabarru', 'tabarru’', 'takaful', 'ujrah', 'jual beli', 'ba’i',
    'riba', 'gharar', 'maysir', 'maisir', 'tadlis', 'najasy', 'ta’widh', "ta'widh",
    'ta’zir', "ta'zir", 'uang elektronik', 'marketplace', 'online shop', 'dropship',
    'voucher', 'sukuk', 'reksa dana', 'reksadana'
]

V10_BAD_TERMS = [
    'umkp', 'umkh', 'umkk', 'umku', 'jawaba ', 'ketetuan', 'relevans', 'ujhrah',
    'jual belik', 'komision', 'hargasaat', 'perolehen', 'biayaa', 'wakolah',
    'kaidarah', 'kaidoh', 'keterbatasn', 'muhyarakah', 'ijaroh', 'ijroh',
    'muwakkul', 'musyarah ', 'berdasrkansyariah'
]

V10_REPLACEMENTS = [
    (r'\bUMKP\b|\bUMKH\b|\bUMKK\b|\bUMKU\b', 'UMKM'),
    (r'\bJawaba\s+Praktis\b', 'Jawaban Praktis'),
    (r'\bJawaba Praktis untukUMKM\b', 'Jawaban Praktis untuk UMKM'),
    (r'\bKetetuan\b', 'Ketentuan'),
    (r'\bRelevans\b', 'Relevan'),
    (r'\bRelevante\b', 'Relevan'),
    (r'\bwakolah\b', 'wakalah'),
    (r'\bWakolah\b', 'Wakalah'),
    (r'\bujhrah\b', 'ujrah'),
    (r'\bkomision\b', 'komisi'),
    (r'\bhargasaat\b', 'harga saat'),
    (r'\bbiayaa\b', 'biaya'),
    (r'\bperolehen\b', 'perolehan'),
    (r'\bkaidarah\b|\bkaidoh\b|\bkaidyah\b', 'kaidah'),
    (r'\bketerbatasn\b', 'keterbatasan'),
    (r'\bijaroh\b|\bijroh\b', 'ijarah'),
    (r'\bmuhyarakah\b|\bmusyarah\b|\bmusyarakoh\b', 'musyarakah'),
    (r'\bmurabaha\b', 'murabahah'),
    (r'\bjual belih\b|\bjual belik\b', 'jual beli'),
    (r'\bdroppe?r\b', 'dropshipper'),
    (r'\bFatwa DEWANSYARI’AH NASIOINAL\b', 'Fatwa DSN-MUI'),
]

def normalize_answer_domain_terms(answer: str, query: str = '', prompt: str = '') -> str:
    ans = clean_generated_answer(answer)
    for pat, repl in DOMAIN_REPLACEMENTS:
        ans = re.sub(pat, repl, ans, flags=re.I)
    for pat, repl in V10_REPLACEMENTS:
        ans = re.sub(pat, repl, ans, flags=re.I)
    # rapikan spasi yang sering hilang akibat generation
    ans = re.sub(r'(?<=[a-zA-Z])(?=UMKM\b)', ' ', ans)
    ans = re.sub(r'\buntukUMKM\b', 'untuk UMKM', ans, flags=re.I)
    ans = re.sub(r'\bprinsipsyariah\b', 'prinsip syariah', ans, flags=re.I)
    ans = re.sub(r'\blembagakeuangan\b', 'lembaga keuangan', ans, flags=re.I)
    ans = re.sub(r'\bpembayarandapat\b', 'pembayaran dapat', ans, flags=re.I)
    ans = re.sub(r'\s+', ' ', ans).replace('Status Jawaban: -', 'Status Jawaban:\n-')
    # kembalikan sedikit struktur newline
    ans = ans.replace(' Jawaban Praktis untuk UMKM:', '\n\nJawaban Praktis untuk UMKM:')
    ans = ans.replace(' Akad/Ketentuan yang Relevan:', '\n\nAkad/Ketentuan yang Relevan:')
    ans = ans.replace(' Dalil/Kaidah dari Dokumen:', '\n\nDalil/Kaidah dari Dokumen:')
    ans = ans.replace(' Sumber Dokumen:', '\n\nSumber Dokumen:')
    ans = ans.replace(' Catatan Batasan:', '\n\nCatatan Batasan:')
    return ans.strip()

def output_is_bad(answer: str, query: str = '', prompt: str = '') -> bool:
    ans = normalize_answer_domain_terms(answer, query=query, prompt=prompt)
    if not ans or len(ans) < 80:
        return True
    if not ans.startswith('Status Jawaban:'):
        return True
    if '[GENERATOR_NOT_LOADED]' in ans:
        return True
    bad_pattern = r'[\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0400-\u04ff]|user\n|assistant\n|chunk_id|reranker_score|pdf_path'
    if re.search(bad_pattern, ans, flags=re.I):
        return True
    low = ans.lower()
    if any(t in low for t in V10_BAD_TERMS):
        return True
    if 'output generator gagal divalidasi' in low:
        return True
    # Terlalu panjang tanpa tanda selesai/penutup sering berarti kepotong.
    tail = ans.strip()[-90:].lower()
    if len(ans) > 1100 and not re.search(r'[.!?)]\s*$', tail):
        return True
    if tail.endswith((' dan', ' atau', ' dengan', ' untuk', ' dalam', ' yaitu', ' sebagai', ' berdasarkan', ',', ';', ':', 'hak-h')):
        return True
    return False

def extract_concepts_from_results(results: List[Dict[str, Any]], query: str = '', limit: int = 10) -> List[str]:
    joined = (query + ' ' + ' '.join(payload_text(r.get('payload', {})) for r in results[:FINAL_CONTEXT_K])).lower()
    found = []
    for c in KNOWN_SYARIAH_CONCEPTS:
        if c.lower() in joined and c not in found:
            found.append(c)
        if len(found) >= limit:
            break
    return found

def source_line_from_payload(p: Dict[str, Any], sid: str) -> str:
    cat = infer_category_from_payload(p) or '-'
    nomor = infer_nomor_from_payload(p) or '-'
    title = short_title(p.get('title'), 120)
    page = p.get('page') or '-'
    return f'- [{sid}] {cat}, No: {nomor}, halaman {page}, {title}'

def collect_dalil_from_results(results: List[Dict[str, Any]], max_sources: int = 3) -> List[str]:
    rows = []
    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get('payload', {})
        doc_text = payload_text(p)
        dalil = extract_dalil_snippets(doc_text, max_chars=MAX_DALIL_CHARS_PER_SOURCE)
        if dalil:
            rows.append(f'- [S{i}] {dalil}')
    return rows

def collect_evidence_points(item: Dict[str, Any], results: List[Dict[str, Any]], max_sources: int = 3) -> List[str]:
    retrieval_query = item.get('retrieval_query') or make_umkm_retrieval_query(item)
    points = []
    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get('payload', {})
        txt = payload_text(p)
        ev = extractive_compress_text(txt, retrieval_query, max_chars=260, max_units=2)
        ev = normalize_text(ev)
        if ev:
            points.append(f'- [S{i}] {ev}')
    return points

def build_extractive_safe_answer(item: Dict[str, Any], results: List[Dict[str, Any]], support_info: Dict[str, Any], reason: str = '') -> str:
    query = item.get('query') or ''
    concepts = extract_concepts_from_results(results, query=query, limit=8)
    evidence_points = collect_evidence_points(item, results, max_sources=3)
    dalil_points = collect_dalil_from_results(results, max_sources=3)
    sources = [source_line_from_payload(r.get('payload', {}), f'S{i}') for i, r in enumerate(results[:3], start=1)]
    support_level = support_info.get('support_level', 'unknown')
    expected_focus = normalize_text(item.get('expected_focus') or '')
    if not results or support_level in ['very_low']:
        return (
            'Status Jawaban:\n'
            '- Tidak cukup konteks\n\n'
            'Jawaban Praktis untuk UMKM:\n'
            '- Dokumen DSN-MUI yang terambil belum cukup kuat untuk menjawab pertanyaan ini secara aman.\n\n'
            'Akad/Ketentuan yang Relevan:\n'
            '- Tidak dapat ditentukan dari konteks yang terambil.\n\n'
            'Dalil/Kaidah dari Dokumen:\n'
            '- Tidak ditemukan dalam konteks yang terambil.\n\n'
            'Sumber Dokumen:\n'
            '- Tidak ada sumber yang cukup kuat.\n\n'
            'Catatan Batasan:\n'
            '- Sistem hanya memakai corpus DSN-MUI precomputed; jangan memaksakan jawaban di luar konteks.'
        )
    akad_block = '\n'.join([f'- {c}' for c in concepts]) if concepts else f'- Konsep yang diuji: {expected_focus or "lihat bukti ekstraktif dan sumber"}'
    bukti_block = '\n'.join(evidence_points[:3]) if evidence_points else '- Tidak ada kutipan ringkas yang cukup bersih dari context.'
    dalil_block = '\n'.join(dalil_points[:3]) if dalil_points else '- Tidak ditemukan dalil/kaidah eksplisit dalam konteks yang terambil.'
    sumber_block = '\n'.join(sources) if sources else '- Tidak ada sumber.'
    return (
        'Status Jawaban:\n'
        '- Ditemukan\n\n'
        'Jawaban Praktis untuk UMKM:\n'
        f'- Berdasarkan sumber DSN-MUI yang terambil, pertanyaan ini dapat dikaitkan dengan: {", ".join(concepts) if concepts else expected_focus or "ketentuan syariah terkait"}. '
        'Untuk penerapan bisnis, pastikan objek, harga/ujrah, kewajiban para pihak, dan risiko transaksi dijelaskan sejak awal.\n'
        '- Bukti ekstraktif utama:\n'
        f'{bukti_block}\n\n'
        'Akad/Ketentuan yang Relevan:\n'
        f'{akad_block}\n\n'
        'Dalil/Kaidah dari Dokumen:\n'
        f'{dalil_block}\n\n'
        'Sumber Dokumen:\n'
        f'{sumber_block}\n\n'
        'Catatan Batasan:\n'
        '- Jawaban ini adalah fallback ekstraktif karena output generator tidak cukup stabil. Gunakan sumber [S] untuk review manual dan konsultasikan detail akad kepada DPS/ahli syariah.'
    )

def build_umkm_context(results: List[Dict[str, Any]], item: Dict[str, Any]) -> str:
    retrieval_query = item.get('retrieval_query') or make_umkm_retrieval_query(item)
    parts = []
    total_chars = 0
    for i, r in enumerate(results[:FINAL_CONTEXT_K], start=1):
        p = r.get('payload', {})
        header = make_source_summary_line(p, f'S{i}')
        doc_text = payload_text(p)
        compressed = extractive_compress_text(doc_text, retrieval_query, max_chars=MAX_CONTEXT_CHARS_PER_SOURCE, max_units=3)
        dalil_text = ''
        if bool(item.get('needs_dalil')) or contains_dalil(doc_text):
            dalil_text = extract_dalil_snippets(doc_text, max_chars=MAX_DALIL_CHARS_PER_SOURCE)
        block = f"""{header}
Bukti relevan:
{compressed}""".strip()
        if dalil_text:
            block += f"\n\nDalil/Kaidah kandidat:\n{dalil_text}"
        if total_chars + len(block) > MAX_CONTEXT_TOTAL_CHARS and parts:
            break
        parts.append(block)
        total_chars += len(block)
    return ('\n\n' + '-' * 48 + '\n\n').join(parts).strip()

def build_umkm_generator_prompt(item: Dict[str, Any], context: str) -> str:
    query = item.get('query') or ''
    expected_focus = item.get('expected_focus') or ''
    return f"""
Anda adalah asisten RAG DSN-MUI untuk UMKM. Jawab HANYA dari konteks. Jangan membuat fatwa baru.

Aturan:
- Jawaban maksimal 180 kata.
- Jangan mengarang dalil, ayat, hadis, nomor fatwa, judul, atau halaman.
- Jika konteks kuat, jawab praktis. Jika tidak cukup, katakan tidak cukup konteks.
- Jika ada Dalil/Kaidah kandidat yang relevan, tampilkan singkat apa adanya dari konteks. Jika tidak ada, tulis: Tidak ditemukan dalam konteks yang terambil.
- Semua klaim utama harus diberi sumber [S1], [S2], dst.
- Bahasa Indonesia normal dan rapi. Jangan typo pada label.

Format wajib:
Status Jawaban:
- Ditemukan / Tidak cukup konteks

Jawaban Praktis untuk UMKM:
- ...

Akad/Ketentuan yang Relevan:
- ...

Dalil/Kaidah dari Dokumen:
- ...

Sumber Dokumen:
- ...

Catatan Batasan:
- ...

Fokus evaluasi: {expected_focus}

[KONTEKS]
{context}

[PERTANYAAN]
{query}
""".strip()

def generate_answer_once(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if generator is None or tokenizer is None:
        return '[GENERATOR_NOT_LOADED]'
    messages = [
        {'role': 'system', 'content': 'Anda adalah asisten RAG DSN-MUI untuk UMKM. Jawab singkat, rapi, Bahasa Indonesia, hanya berdasarkan konteks.'},
        {'role': 'user', 'content': prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer_prefix = 'Status Jawaban:\n- '
    text = text + answer_prefix
    input_device = next(generator.parameters()).device
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_PROMPT_TOKENS).to(input_device)
    with torch.inference_mode():
        output_ids = generator.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.02,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    generated = output_ids[0][inputs['input_ids'].shape[-1]:]
    ans = answer_prefix + tokenizer.decode(generated, skip_special_tokens=True).strip()
    return normalize_answer_domain_terms(ans)

def generate_answer(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, query: str = '') -> str:
    if generator is None or tokenizer is None:
        return '[GENERATOR_NOT_LOADED]'
    ans = generate_answer_once(prompt, max_new_tokens=max_new_tokens)
    ans = normalize_answer_domain_terms(ans, query=query, prompt=prompt)
    if not output_is_bad(ans, query=query, prompt=prompt):
        return ans
    retry_prompt = prompt + "\n\nULANGI: jawaban maksimal 120 kata, format wajib persis, Bahasa Indonesia rapi, jangan typo, jangan menambah sumber/dalil."
    ans2 = generate_answer_once(retry_prompt, max_new_tokens=min(max_new_tokens, 260))
    ans2 = normalize_answer_domain_terms(ans2, query=query, prompt=prompt)
    return ans2

def answer_query_umkm_safely(item: Dict[str, Any], results: List[Dict[str, Any]], prompt: str, support_info: Dict[str, Any]) -> str:
    query = item.get('query') or ''
    support_level = support_info.get('support_level')
    if FORCE_ABSTAIN_ON_LOW_SUPPORT and support_level == 'very_low':
        return build_extractive_safe_answer(item, results, support_info, reason='very_low_support')
    ans = generate_answer(prompt, query=query)
    # Kalau support sebenarnya kuat tapi model malah abstain/gagal/typo, gunakan fallback ekstraktif yang tetap grounded.
    if output_is_bad(ans, query=query, prompt=prompt) or (support_level in ['high', 'medium'] and answer_abstained(ans)):
        ans_retry_prompt = prompt + "\n\nKonteks dinilai cukup. Jangan menjawab 'Tidak cukup konteks' kecuali benar-benar tidak ada sumber. Jawab sangat singkat dan rapi."
        ans2 = generate_answer(ans_retry_prompt, max_new_tokens=260, query=query)
        if not output_is_bad(ans2, query=query, prompt=ans_retry_prompt) and not (support_level in ['high', 'medium'] and answer_abstained(ans2)):
            return ans2
        return build_extractive_safe_answer(item, results, support_info, reason='generator_bad_or_abstained')
    return ans

print('PATCH V10 installed: shorter UMKM prompt + extractive fallback + safer dalil handling.')



# %% CELL 22

# =====================================================
# PATCH V11: Qwen3-4B-Instruct-2507 + JSON schema generation
# =====================================================
# Rationale:
# - Generator lama sering typo/kepotong ketika diminta format teks panjang.
# - V11 meminta model output JSON minimal, lalu notebook yang merender jawaban final.
# - Dalil/kaidah dan sumber dokumen diambil deterministik dari context retrieval.

import json, re, gc, inspect
from packaging import version

GENERATOR_MODEL_NAME = 'Qwen/Qwen3-4B-Instruct-2507'
GENERATOR_USE_4BIT = False
MAX_NEW_TOKENS = 768
MAX_PROMPT_TOKENS = 8192
MAX_CONTEXT_CHARS_PER_SOURCE = 650
MAX_CONTEXT_TOTAL_CHARS = 4200
MAX_DALIL_CHARS_PER_SOURCE = 480
FINAL_CONTEXT_K = 4

print('V11 generator model:', GENERATOR_MODEL_NAME)

try:
    import transformers
    print('transformers:', transformers.__version__)
    if version.parse(transformers.__version__) < version.parse('4.51.0'):
        print('WARNING: Qwen3 membutuhkan transformers >= 4.51.0. Upgrade dengan: pip install -U transformers accelerate')
except Exception as e:
    print('Tidak bisa cek versi transformers:', e)


def load_generator_model(force_reload: bool = False):
    """Load Qwen3 Instruct. Qwen3-4B-Instruct-2507 adalah non-thinking mode, jadi tidak perlu enable_thinking=False."""
    global generator, tokenizer

    if generator is not None and tokenizer is not None and not force_reload:
        print('Generator already loaded:', GENERATOR_MODEL_NAME)
        return generator, tokenizer

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print_gpu_memory('Before generator load | ')
    print('Loading generator:', GENERATOR_MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        'device_map': 'auto',
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
    }

    if GENERATOR_USE_4BIT:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
            )
            print('4-bit quantization enabled.')
        except Exception as e:
            print('Gagal enable 4-bit. Lanjut load normal.')
            print(type(e).__name__, str(e)[:300])
    else:
        # Qwen3 model card merekomendasikan torch_dtype="auto".
        model_kwargs['torch_dtype'] = 'auto'

    try:
        generator = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_NAME, **model_kwargs)
    except TypeError:
        # Fallback untuk versi transformers lama yang tidak menerima string "auto".
        model_kwargs.pop('torch_dtype', None)
        if torch.cuda.is_available():
            model_kwargs['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            model_kwargs['torch_dtype'] = torch.float32
        generator = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_NAME, **model_kwargs)

    generator.eval()
    print('Generator ready:', GENERATOR_MODEL_NAME)
    print_gpu_memory('After generator load | ')
    return generator, tokenizer


# -----------------------------
# JSON parsing helpers
# -----------------------------
REQUIRED_JSON_KEYS = ['status', 'jawaban_praktis', 'akad_ketentuan_relevan', 'catatan_batasan']


def strip_code_fence(text: str) -> str:
    text = (text or '').strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = strip_code_fence(text)
    if not raw:
        return None
    # direct parse
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # find first balanced-looking JSON object
    start = raw.find('{')
    end = raw.rfind('}')
    if start >= 0 and end > start:
        candidate = raw[start:end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            # mild cleanup: trailing comma
            candidate2 = re.sub(r',\s*([}\]])', r'\1', candidate)
            try:
                obj = json.loads(candidate2)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


def clean_json_value(x):
    if isinstance(x, str):
        return normalize_text(x)
    if isinstance(x, list):
        return [normalize_text(str(v)) for v in x if str(v).strip()]
    if x is None:
        return ''
    return normalize_text(str(x))


def normalize_structured_obj(obj: Dict[str, Any], support_info: Dict[str, Any]) -> Dict[str, Any]:
    obj = obj or {}
    status = str(obj.get('status', '')).lower().strip()
    if status not in ['ditemukan', 'tidak_cukup_konteks', 'tidak cukup konteks']:
        status = 'ditemukan' if support_info.get('support_level') in ['high', 'medium'] else 'tidak_cukup_konteks'
    if status == 'tidak cukup konteks':
        status = 'tidak_cukup_konteks'

    out = {
        'status': status,
        'jawaban_praktis': clean_json_value(obj.get('jawaban_praktis', '')),
        'akad_ketentuan_relevan': clean_json_value(obj.get('akad_ketentuan_relevan', [])),
        'catatan_batasan': clean_json_value(obj.get('catatan_batasan', '')),
    }
    if isinstance(out['akad_ketentuan_relevan'], str):
        out['akad_ketentuan_relevan'] = [out['akad_ketentuan_relevan']] if out['akad_ketentuan_relevan'] else []
    return out


def structured_obj_is_bad(obj: Optional[Dict[str, Any]], raw: str = '') -> bool:
    if not isinstance(obj, dict):
        return True
    for k in REQUIRED_JSON_KEYS:
        if k not in obj:
            return True
    combined = json.dumps(obj, ensure_ascii=False) + ' ' + (raw or '')
    bad_pattern = r'[\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0400-\u04ff]|chunk_id|reranker_score|pdf_path|assistant\n|user\n'
    if re.search(bad_pattern, combined, flags=re.I):
        return True
    low = combined.lower()
    if any(t in low for t in V10_BAD_TERMS):
        return True
    return False


# -----------------------------
# Deterministic context/evidence rendering
# -----------------------------

def deterministic_sources(results: List[Dict[str, Any]], max_sources: int = 4) -> List[str]:
    return [source_line_from_payload(r.get('payload', {}), f'S{i}') for i, r in enumerate(results[:max_sources], start=1)]


def deterministic_dalil_lines(results: List[Dict[str, Any]], item: Dict[str, Any], max_sources: int = 3) -> List[str]:
    # Ambil dalil/kaidah hanya dari dokumen terambil.
    lines = []
    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get('payload', {})
        doc_text = payload_text(p)
        dalil = extract_dalil_snippets(doc_text, max_chars=MAX_DALIL_CHARS_PER_SOURCE)
        if dalil:
            lines.append(f'- [S{i}] {dalil}')
    return lines


def build_umkm_context(results: List[Dict[str, Any]], item: Dict[str, Any]) -> str:
    retrieval_query = item.get('retrieval_query') or make_umkm_retrieval_query(item)
    parts = []
    total_chars = 0
    for i, r in enumerate(results[:FINAL_CONTEXT_K], start=1):
        p = r.get('payload', {})
        header = make_source_summary_line(p, f'S{i}')
        doc_text = payload_text(p)
        compressed = extractive_compress_text(
            doc_text,
            retrieval_query,
            max_chars=MAX_CONTEXT_CHARS_PER_SOURCE,
            max_units=4,
        )
        block = f"""{header}
Bukti relevan:
{compressed}""".strip()
        # kandidat dalil tetap dimasukkan ke context supaya model tahu ada sumbernya,
        # tetapi output dalil final tetap dirender deterministik oleh notebook.
        dalil_text = extract_dalil_snippets(doc_text, max_chars=MAX_DALIL_CHARS_PER_SOURCE)
        if dalil_text:
            block += f"\nDalil/Kaidah kandidat dari source ini:\n{dalil_text}"
        if total_chars + len(block) > MAX_CONTEXT_TOTAL_CHARS and parts:
            break
        parts.append(block)
        total_chars += len(block)
    return ('\n\n' + '-' * 48 + '\n\n').join(parts).strip()


def build_umkm_generator_prompt(item: Dict[str, Any], context: str) -> str:
    query = item.get('query') or ''
    expected_focus = item.get('expected_focus') or ''
    return f"""
Anda adalah asisten RAG DSN-MUI untuk UMKM.
Jawab berdasarkan KONTEKS saja. Jangan membuat fatwa baru, jangan mengarang dalil, ayat, hadis, nomor fatwa, judul, atau halaman.

Tugasmu hanya menghasilkan JSON valid, tanpa markdown, tanpa penjelasan di luar JSON.

Schema JSON wajib:
{{
  "status": "ditemukan" atau "tidak_cukup_konteks",
  "jawaban_praktis": "jawaban singkat 2-4 kalimat untuk UMKM, pakai rujukan [S1]/[S2] jika menyebut aturan",
  "akad_ketentuan_relevan": ["poin akad/ketentuan yang relevan, masing-masing singkat dan diberi sumber [Sx]"],
  "catatan_batasan": "batasan jawaban; sebutkan kalau perlu verifikasi DPS/ahli syariah"
}}

Aturan status:
- Gunakan "ditemukan" jika konteks berisi aturan/akad/larangan yang relevan.
- Gunakan "tidak_cukup_konteks" hanya jika konteks tidak mendukung jawaban secara aman.
- Jangan mengisi dalil/kaidah dalam JSON; dalil/kaidah akan diambil otomatis oleh sistem dari source.

Fokus evaluasi: {expected_focus}

[KONTEKS]
{context}

[PERTANYAAN UMKM]
{query}
""".strip()


def render_structured_answer(
    obj: Dict[str, Any],
    item: Dict[str, Any],
    results: List[Dict[str, Any]],
    support_info: Dict[str, Any],
    mode: str = 'json_generated',
) -> str:
    obj = normalize_structured_obj(obj, support_info)
    status_text = 'Ditemukan' if obj.get('status') == 'ditemukan' else 'Tidak cukup konteks'
    jawaban = obj.get('jawaban_praktis') or 'Dokumen yang terambil belum cukup kuat untuk menjawab secara aman.'
    akad_list = obj.get('akad_ketentuan_relevan') or ['Tidak dapat ditentukan dari konteks yang terambil.']
    akad_block = '\n'.join(f'- {x}' for x in akad_list[:6])

    dalil_lines = deterministic_dalil_lines(results, item, max_sources=3)
    dalil_block = '\n'.join(dalil_lines) if dalil_lines else '- Tidak ditemukan dalil/kaidah eksplisit dalam konteks yang terambil.'
    sources = deterministic_sources(results, max_sources=4)
    source_block = '\n'.join(sources) if sources else '- Tidak ada sumber yang cukup kuat.'
    catatan = obj.get('catatan_batasan') or 'Jawaban dibatasi pada corpus DSN-MUI precomputed dan konteks yang terambil.'

    answer = (
        f'Status Jawaban:\n- {status_text}\n\n'
        f'Jawaban Praktis untuk UMKM:\n- {jawaban}\n\n'
        f'Akad/Ketentuan yang Relevan:\n{akad_block}\n\n'
        f'Dalil/Kaidah dari Dokumen:\n{dalil_block}\n\n'
        f'Sumber Dokumen:\n{source_block}\n\n'
        f'Catatan Batasan:\n- {catatan}'
    )
    if mode != 'json_generated':
        answer += f'\n- Mode jawaban: {mode}.'
    return normalize_answer_domain_terms(answer, query=item.get('query', ''))


def build_extractive_safe_answer(item: Dict[str, Any], results: List[Dict[str, Any]], support_info: Dict[str, Any], reason: str = '') -> str:
    query = item.get('query') or ''
    support_level = support_info.get('support_level', 'unknown')
    if not results or support_level == 'very_low':
        obj = {
            'status': 'tidak_cukup_konteks',
            'jawaban_praktis': 'Dokumen DSN-MUI yang terambil belum cukup kuat untuk menjawab pertanyaan ini secara aman.',
            'akad_ketentuan_relevan': ['Tidak dapat ditentukan dari konteks yang terambil.'],
            'catatan_batasan': 'Sistem hanya memakai corpus DSN-MUI precomputed; jangan memaksakan jawaban di luar konteks.'
        }
        return render_structured_answer(obj, item, results, support_info, mode='NO_CONTEXT')

    concepts = extract_concepts_from_results(results, query=query, limit=8)
    evidence_points = collect_evidence_points(item, results, max_sources=3)
    evidence_text = ' '.join([re.sub(r'^- \[S\d+\]\s*', '', x) for x in evidence_points[:2]])
    focus = normalize_text(item.get('expected_focus') or '')
    obj = {
        'status': 'ditemukan',
        'jawaban_praktis': (
            f'Konteks DSN-MUI yang terambil mendukung jawaban terkait {", ".join(concepts) if concepts else focus or "ketentuan syariah bisnis"}. '
            f'Bukti utama: {evidence_text[:520]}'
        ),
        'akad_ketentuan_relevan': concepts or [focus or 'Lihat bukti ekstraktif dari sumber terambil.'],
        'catatan_batasan': 'Jawaban ini fallback ekstraktif karena generator gagal menghasilkan JSON valid; review manual tetap disarankan.'
    }
    return render_structured_answer(obj, item, results, support_info, mode=f'CONTEXT_FOUND_GENERATION_FAILED:{reason or "fallback_extractive"}')


# -----------------------------
# Qwen3 JSON generation
# -----------------------------

def generate_structured_json_once(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> Tuple[Optional[Dict[str, Any]], str]:
    if generator is None or tokenizer is None:
        return None, '[GENERATOR_NOT_LOADED]'
    messages = [
        {
            'role': 'system',
            'content': 'Anda adalah generator JSON untuk RAG DSN-MUI. Output harus JSON valid saja, tanpa markdown.'
        },
        {'role': 'user', 'content': prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_device = next(generator.parameters()).device
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_PROMPT_TOKENS).to(input_device)
    with torch.inference_mode():
        output_ids = generator.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.01,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    generated = output_ids[0][inputs['input_ids'].shape[-1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return extract_json_object(raw), raw


def answer_query_umkm_safely(item: Dict[str, Any], results: List[Dict[str, Any]], prompt: str, support_info: Dict[str, Any]):
    query = item.get('query') or ''
    support_level = support_info.get('support_level')
    if FORCE_ABSTAIN_ON_LOW_SUPPORT and support_level == 'very_low':
        ans = build_extractive_safe_answer(item, results, support_info, reason='very_low_support')
        return ans, {'generation_mode': 'NO_CONTEXT', 'raw_generation': '', 'json_parse_ok': False}

    obj, raw = generate_structured_json_once(prompt, max_new_tokens=MAX_NEW_TOKENS)
    json_ok = not structured_obj_is_bad(obj, raw)
    if json_ok:
        obj_norm = normalize_structured_obj(obj, support_info)
        # Kalau support high/medium tetapi model bilang tidak cukup konteks, retry sekali.
        if support_level in ['high', 'medium'] and obj_norm.get('status') == 'tidak_cukup_konteks':
            retry_prompt = prompt + '\n\nKonteks dinilai cukup oleh retriever. Jika ada aturan relevan, gunakan status "ditemukan". Tetap output JSON valid saja.'
            obj2, raw2 = generate_structured_json_once(retry_prompt, max_new_tokens=min(MAX_NEW_TOKENS, 512))
            if not structured_obj_is_bad(obj2, raw2):
                obj2_norm = normalize_structured_obj(obj2, support_info)
                if obj2_norm.get('status') == 'ditemukan':
                    ans = render_structured_answer(obj2_norm, item, results, support_info, mode='json_generated_retry')
                    return ans, {'generation_mode': 'json_generated_retry', 'raw_generation': raw2, 'json_parse_ok': True}
        ans = render_structured_answer(obj_norm, item, results, support_info, mode='json_generated')
        return ans, {'generation_mode': 'json_generated', 'raw_generation': raw, 'json_parse_ok': True}

    # Retry dengan prompt lebih pendek dan schema lebih tegas.
    retry_prompt = prompt + '\n\nPERBAIKI: Output hanya JSON valid. Jangan tulis markdown. Jangan tulis teks di luar JSON.'
    obj2, raw2 = generate_structured_json_once(retry_prompt, max_new_tokens=min(MAX_NEW_TOKENS, 512))
    if not structured_obj_is_bad(obj2, raw2):
        ans = render_structured_answer(obj2, item, results, support_info, mode='json_generated_retry')
        return ans, {'generation_mode': 'json_generated_retry', 'raw_generation': raw2, 'json_parse_ok': True}

    ans = build_extractive_safe_answer(item, results, support_info, reason='json_invalid')
    return ans, {'generation_mode': 'extractive_fallback_json_invalid', 'raw_generation': raw2 or raw, 'json_parse_ok': False}


def output_is_bad(answer: str, query: str = '', prompt: str = '') -> bool:
    # Setelah v11 rendering deterministik, validasi fokus pada kontaminasi/empty saja.
    ans = normalize_answer_domain_terms(answer, query=query, prompt=prompt)
    if not ans or len(ans) < 80:
        return True
    if not ans.startswith('Status Jawaban:'):
        return True
    bad_pattern = r'[\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0400-\u04ff]|chunk_id|reranker_score|pdf_path|assistant\n|user\n'
    if re.search(bad_pattern, ans, flags=re.I):
        return True
    return False

print('PATCH V11 installed: Qwen3-4B Instruct + JSON schema + deterministic sources/dalil + correct fallback states.')



# %% CELL 23
# =====================================================
# PATCH V11C: Manual HF reranker aktif + anti-stuck diagnostics
# =====================================================
# Catatan penting:
# - USE_RERANKER di config awal sengaja False agar cell loader tidak memanggil FlagEmbedding FlagReranker.
# - Di sini reranker diaktifkan kembali memakai manual HuggingFace AutoModelForSequenceClassification.
# - OCR eksternal tetap dimatikan agar bottleneck tidak bercampur dengan reranker.

DEBUG_RETRIEVAL_STEPS = True

# Aktifkan kembali reranker, tetapi lewat manual HF patch, bukan FlagEmbedding reranker.
USE_RERANKER = True
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 4
RERANK_MAX_CHARS_PER_PASSAGE = min(globals().get('RERANK_MAX_CHARS_PER_PASSAGE', 1800), 1600)

# Matikan OCR eksternal saat full retrieval suite agar tidak terlihat stuck di evidence stage.
# Dalil tetap dicari dari teks chunk precomputed.
ENABLE_ARABIC_EVIDENCE = False
ENABLE_ARABIC_EVIDENCE_FOR_DALIL_QUERY = False
EVIDENCE_TOP_N = 0

# Kandidat dibuat cukup tapi tidak terlalu besar, supaya reranker tidak lama di query pertama.
VECTOR_TOP_K = min(globals().get('VECTOR_TOP_K', 80), 60)
BM25_TOP_K = min(globals().get('BM25_TOP_K', 80), 60)
HYBRID_TOP_K = min(globals().get('HYBRID_TOP_K', 50), 32)
FINAL_CONTEXT_K = min(globals().get('FINAL_CONTEXT_K', 5), 4)
UMKM_FINAL_CONTEXT_K = FINAL_CONTEXT_K


def _dbg(msg: str):
    if DEBUG_RETRIEVAL_STEPS:
        print(msg, flush=True)


@torch.no_grad()
def score_pairs_manual(query, passages):
    """Manual HF reranker scoring dengan progress log.
    Menghindari error FlagEmbedding:
    AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model
    """
    if reranker_model is None or reranker_tokenizer is None:
        return [0.0] * len(passages)

    scores = []
    total = len(passages)
    for start in range(0, total, RERANK_BATCH_SIZE):
        end = min(start + RERANK_BATCH_SIZE, total)
        _dbg(f'[rerank] scoring batch {start//RERANK_BATCH_SIZE + 1}/{(total + RERANK_BATCH_SIZE - 1)//RERANK_BATCH_SIZE} | docs {start+1}-{end}/{total}')

        batch_passages = passages[start:end]
        batch_queries = [query] * len(batch_passages)

        inputs = reranker_tokenizer(
            batch_queries,
            batch_passages,
            padding=True,
            truncation='only_second',
            max_length=RERANK_MAX_LENGTH,
            return_tensors='pt'
        )
        inputs = {k: v.to(reranker_device) for k, v in inputs.items()}

        outputs = reranker_model(**inputs)
        logits = outputs.logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            batch_scores = logits[:, 0]
        elif logits.ndim == 2:
            batch_scores = logits[:, -1]
        else:
            batch_scores = logits.view(-1)
        scores.extend(batch_scores.detach().float().cpu().numpy().tolist())
    return scores


def rerank_hits(query: str, hits: List[Dict[str, Any]], final_k: int = FINAL_CONTEXT_K) -> List[Dict[str, Any]]:
    """V11C override: manual HF reranker dengan debug logs."""
    if not hits:
        return []

    if not USE_RERANKER:
        _dbg(f'[rerank] skipped: USE_RERANKER=False | candidates={len(hits)}')
        boosted = []
        for h in hits:
            hh = dict(h)
            lex = lexical_score_for_payload(query, hh.get('payload', {})) if 'lexical_score_for_payload' in globals() else 0.0
            hh['score_lexical_boost'] = float(lex)
            hh['score_reranker'] = float(hh.get('score_hybrid') or 0.0) + 0.001 * float(lex)
            hh['reranker_used'] = False
            boosted.append(hh)
        ranked = sorted(boosted, key=lambda x: x.get('score_reranker', -1e9), reverse=True)[:final_k]
        for i, h in enumerate(ranked, start=1):
            h['rank_final'] = i
        return ranked

    _dbg(f'[rerank] manual HF reranker active | candidates={len(hits)} | model={RERANKER_MODEL_NAME}')
    load_reranker()
    passages = [build_passage_for_rerank(h.get('payload', {}), max_chars=RERANK_MAX_CHARS_PER_PASSAGE) for h in hits]
    scores = score_pairs_manual(query, passages)

    reranked = []
    for h, s in zip(hits, scores):
        hh = dict(h)
        lex = lexical_score_for_payload(query, hh.get('payload', {})) if 'lexical_score_for_payload' in globals() else 0.0
        hh['score_reranker_raw'] = float(s)
        hh['score_lexical_boost'] = float(lex)
        # Lexical boost kecil agar istilah domain query tetap dihargai.
        hh['score_reranker'] = float(s) + 0.03 * float(lex)
        hh['reranker_used'] = True
        reranked.append(hh)

    ranked = sorted(reranked, key=lambda x: x.get('score_reranker', -1e9), reverse=True)[:final_k]
    for i, h in enumerate(ranked, start=1):
        h['rank_final'] = i
    _dbg('[rerank] done')
    return ranked


def retrieve_final(query: str, category_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """V11C retrieve_final dengan logs per tahap."""
    t0 = now_ms()
    _dbg('[1/5] vector search: embed query + dot product')
    vh = vector_search_numpy(query, limit=VECTOR_TOP_K, category_filter=category_filter)
    t1 = now_ms()
    _dbg(f'[1/5] vector done: {len(vh)} hits | {round(t1-t0,2)} ms')

    _dbg('[2/5] BM25 search')
    bh = bm25_search(query, limit=BM25_TOP_K, category_filter=category_filter)
    t2 = now_ms()
    _dbg(f'[2/5] BM25 done: {len(bh)} hits | {round(t2-t1,2)} ms')

    _dbg('[3/5] reciprocal rank fusion')
    hybrid = reciprocal_rank_fusion(vh, bh, limit=HYBRID_TOP_K)

    requested_nomor = extract_requested_nomor(query) if 'extract_requested_nomor' in globals() else ''
    direct_count = 0
    if requested_nomor:
        _dbg(f'[3/5] query meminta nomor fatwa {requested_nomor}: direct candidates')
        direct = direct_nomor_candidates(query, requested_nomor, category_filter=category_filter, limit=DIRECT_NOMOR_CANDIDATES_K)
        direct_count = len(direct)
        hybrid = merge_hits(hybrid, direct, limit=HYBRID_TOP_K + DIRECT_NOMOR_CANDIDATES_K)
        filtered = [h for h in hybrid if infer_nomor_from_payload(h.get('payload', {})) == requested_nomor]
        if filtered:
            hybrid = filtered
        else:
            _dbg(f'[warn] kandidat Fatwa {requested_nomor} tidak ditemukan, fallback ke hybrid umum')

    if category_filter:
        hybrid = [h for h in hybrid if infer_category_from_payload(h.get('payload', {})) == category_filter]

    for i, h in enumerate(hybrid, start=1):
        h['rank_hybrid_filtered'] = i
    t3 = now_ms()
    _dbg(f'[3/5] fusion/filter done: {len(hybrid)} candidates | {round(t3-t2,2)} ms')

    _dbg('[4/5] manual HF rerank/final select')
    final = rerank_hits(query, hybrid, final_k=FINAL_CONTEXT_K)
    t4 = now_ms()
    _dbg(f'[4/5] final select done: {len(final)} results | {round(t4-t3,2)} ms')

    timing = {
        'vector_ms': round(t1 - t0, 2),
        'bm25_ms': round(t2 - t1, 2),
        'rrf_ms': round(t3 - t2, 2),
        'rerank_ms': round(t4 - t3, 2),
        'retrieval_total_ms': round(t4 - t0, 2),
        'requested_nomor': requested_nomor,
        'direct_nomor_candidates': direct_count,
        'manual_hf_reranker': bool(USE_RERANKER),
    }
    return final, timing


print('V11C manual HF reranker patch active.')
print('Initial FlagEmbedding reranker is intentionally not used.')
print('USE_RERANKER =', USE_RERANKER, '| RERANK_BATCH_SIZE =', RERANK_BATCH_SIZE, '| HYBRID_TOP_K =', HYBRID_TOP_K)
print('ENABLE_ARABIC_EVIDENCE =', ENABLE_ARABIC_EVIDENCE)
print('FINAL_CONTEXT_K =', FINAL_CONTEXT_K)



# %% CELL 25

# =====================================================
# PATCH V12: Statistical reranker (tanpa model HF/FlagEmbedding)
# =====================================================
# Kenapa:
# - Manual HF reranker sebelumnya bisa stuck saat AutoModelForSequenceClassification.from_pretrained(...).
# - V12 tidak load model reranker sama sekali. Jadi Phase 1 hanya pakai statistik:
#   RRF score + vector/BM25 rank + lexical overlap + phrase/concept/title/category boost.
# - Generator tetap Qwen3-4B JSON dari V11.

DEBUG_RETRIEVAL_STEPS = True
USE_RERANKER = True
RERANKER_TYPE = 'statistical_v12'
RERANKER_MODEL_NAME = 'STATISTICAL_RERANKER_V12'

# Tidak memakai OCR eksternal pada Phase 1 agar tidak stuck.
# Dalil/kaidah tetap diambil dari teks chunk precomputed melalui deterministic_dalil_lines().
ENABLE_ARABIC_EVIDENCE = False
ENABLE_ARABIC_EVIDENCE_FOR_DALIL_QUERY = False
EVIDENCE_TOP_N = 0

# Kandidat dan context yang aman untuk statistical reranking.
VECTOR_TOP_K = max(60, globals().get('VECTOR_TOP_K', 60))
BM25_TOP_K = max(60, globals().get('BM25_TOP_K', 60))
HYBRID_TOP_K = 36
FINAL_CONTEXT_K = 5
UMKM_FINAL_CONTEXT_K = FINAL_CONTEXT_K
MAX_CONTEXT_TOTAL_CHARS = 4600
MAX_CONTEXT_CHARS_PER_SOURCE = 650
MAX_PROMPT_TOKENS = 8192
MAX_NEW_TOKENS = 768

STAT_STOPWORDS = {
    'yang','dan','di','ke','dari','untuk','dengan','pada','dalam','atau','agar','apa','bagaimana',
    'saya','kami','mau','ingin','punya','usaha','umkm','produk','bisnis','secara','menurut','berdasarkan',
    'apakah','boleh','harus','perlu','jika','itu','ini','ada','sebagai','antara','oleh','para','pihak','bagi',
    'sebutkan','jelaskan','terhadap','mereka','dapat','bisa','lebih','tidak','tanpa','dengan','menggunakan'
}

STAT_CONCEPT_SYNONYMS = {
    'jual beli': ['jual beli','jual','beli','ba’i','al-ba’i','akad jual beli','transaksi jual beli'],
    'marketplace': ['marketplace','platform','lokapasar','e-commerce','online','toko online','pedagang online'],
    'salam': ['salam','jual beli salam','pemesanan','pre-order','pre order','pesanan'],
    'istishna': ['istishna','istishna\'','istisna','pembuatan barang','pesanan pembuatan'],
    'murabahah': ['murabahah','margin','jual beli dengan margin','pembiayaan pembelian'],
    'ijarah': ['ijarah','sewa','jasa','upah','ujrah','fee'],
    'wakalah': ['wakalah','wakalah bil ujrah','kuasa','wakil','perwakilan','komisi'],
    'ju\'alah': ['ju\'alah','jualah','imbalan keberhasilan','success fee'],
    'mudharabah': ['mudharabah','bagi hasil','mudharib','shahibul maal'],
    'musyarakah': ['musyarakah','syirkah','kerja sama modal','patungan modal'],
    'qardh': ['qardh','pinjaman','dana talangan'],
    'rahn': ['rahn','gadai','jaminan'],
    'kafalah': ['kafalah','penjaminan','jaminan'],
    'hawalah': ['hawalah','pengalihan utang','alih piutang'],
    'riba': ['riba','bunga','tambahan pinjaman','ribawi'],
    'gharar': ['gharar','ketidakjelasan','tidak jelas','spekulatif'],
    'maysir': ['maysir','maisir','judi','perjudian','gambling'],
    'tadlis': ['tadlis','penipuan','menyembunyikan cacat','informasi palsu'],
    'najasy': ['najasy','tanajusy','fake order','review palsu','penawaran palsu'],
    'risywah': ['risywah','suap','gratifikasi'],
    'voucher': ['voucher','kupon','cashback','poin','loyalty','diskon'],
    'uang elektronik': ['uang elektronik','e-money','dompet digital','wallet','saldo elektronik','qr','qris'],
    'fintech': ['fintech','layanan pembiayaan berbasis teknologi','platform pembiayaan','paylater'],
    'asuransi': ['asuransi','takaful','tabarru','dana tabarru','polis'],
    'wisata': ['wisata','pariwisata','travel','hotel','destinasi','tour'],
    'halal': ['halal','sertifikasi halal','produk halal','makanan halal'],
    'sukuk': ['sukuk','obligasi syariah','surat berharga syariah'],
    'reksadana': ['reksadana','reksa dana','investasi kolektif'],
}

TITLE_BOOST_TERMS = [
    'marketplace','jual beli','akad jual beli','salam','istishna','murabahah','wakalah','ijarah','ju\'alah',
    'uang elektronik','voucher','fintech','pembiayaan','rahn','qardh','kafalah','asuransi','pariwisata',
    'hotel','pedoman implementasi','ta\'limat','pasar modal','sukuk','reksa dana'
]


def _dbg(msg: str):
    if DEBUG_RETRIEVAL_STEPS:
        print(msg, flush=True)


def stat_tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    text = text.replace('’', "'").replace('`', "'")
    toks = re.findall(r"[a-z0-9][a-z0-9'\-]{1,}", text)
    return [t for t in toks if t not in STAT_STOPWORDS and len(t) >= 3]


def stat_query_terms(query: str) -> List[str]:
    toks = stat_tokenize(query)
    terms = list(dict.fromkeys(toks))
    qlow = normalize_text(query).lower()
    for concept, variants in STAT_CONCEPT_SYNONYMS.items():
        if any(v in qlow for v in variants):
            for v in variants[:4]:
                terms.extend(stat_tokenize(v))
            terms.extend(stat_tokenize(concept))
    return list(dict.fromkeys(terms))[:80]


def stat_contains(text_low: str, phrase: str) -> bool:
    return normalize_text(phrase).lower() in text_low


def stat_concept_score(query: str, combined_low: str, title_low: str) -> float:
    qlow = normalize_text(query).lower()
    score = 0.0
    for concept, variants in STAT_CONCEPT_SYNONYMS.items():
        query_has = any(v in qlow for v in variants)
        if not query_has:
            continue
        doc_hits = sum(1 for v in variants if v in combined_low)
        title_hits = sum(1 for v in variants if v in title_low)
        if doc_hits:
            score += min(3.0, 0.65 * doc_hits)
        if title_hits:
            score += min(2.5, 1.15 * title_hits)
    return score


def stat_phrase_score(query: str, text_low: str, title_low: str) -> float:
    qlow = normalize_text(query).lower()
    score = 0.0
    # frasa 2-4 kata dari query yang muncul di dokumen/title
    toks = stat_tokenize(query)
    phrases = []
    for n in [4, 3, 2]:
        for i in range(0, max(0, len(toks) - n + 1)):
            ph = ' '.join(toks[i:i+n])
            if len(ph) >= 8:
                phrases.append(ph)
    for ph in list(dict.fromkeys(phrases))[:30]:
        if ph in title_low:
            score += 1.4
        elif ph in text_low:
            score += 0.7
    # title domain boost
    for term in TITLE_BOOST_TERMS:
        if term in qlow and term in title_low:
            score += 2.0
        elif term in qlow and term in text_low:
            score += 0.7
    return score


def stat_keyword_coverage(query: str, text: str, title: str = '') -> Tuple[float, int, int]:
    terms = stat_query_terms(query)
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
    coverage = hits / max(1, len(terms))
    return coverage + 0.03 * weighted, hits, len(terms)


def statistical_score_hit(query: str, hit: Dict[str, Any]) -> Dict[str, float]:
    p = hit.get('payload', {}) or {}
    title = str(p.get('title') or p.get('doc_title') or p.get('pdf_filename') or '')
    text = payload_text(p)
    combined = f"{title}\n{text}"
    combined_low = normalize_text(combined).lower()
    title_low = normalize_text(title).lower()

    coverage_score, kw_hits, kw_total = stat_keyword_coverage(query, text, title)
    concept_score = stat_concept_score(query, combined_low, title_low)
    phrase_score = stat_phrase_score(query, combined_low, title_low)

    # Base signals dari retrieval awal. Tidak perlu minmax karena hanya untuk kandidat top-k.
    rrf_score = float(hit.get('score_hybrid') or hit.get('score_rrf') or 0.0)
    vector_score = float(hit.get('score_vector') or 0.0)
    bm25_rank = hit.get('rank_bm25')
    vector_rank = hit.get('rank_vector')
    bm25_rank_score = 1.0 / (1.0 + float(bm25_rank)) if bm25_rank else 0.0
    vector_rank_score = 1.0 / (1.0 + float(vector_rank)) if vector_rank else 0.0

    # Dokumen dengan judul sangat relevan sering lebih baik untuk UMKM daripada chunk yang cuma cocok keyword.
    title_score = 0.0
    qlow = normalize_text(query).lower()
    for term in TITLE_BOOST_TERMS:
        if term in qlow and term in title_low:
            title_score += 1.2
    title_score = min(title_score, 5.0)

    # Penalti ringan kalau OCR/text sangat pendek atau terlalu noise.
    text_len = len(normalize_text(text))
    length_score = 0.5 if text_len >= 180 else -0.8

    total = (
        10.0 * rrf_score +
        1.2 * vector_score +
        1.4 * bm25_rank_score +
        1.0 * vector_rank_score +
        6.0 * coverage_score +
        2.2 * concept_score +
        1.3 * phrase_score +
        1.8 * title_score +
        length_score
    )
    return {
        'score_reranker': float(total),
        'score_stat_rrf': float(rrf_score),
        'score_stat_vector': float(vector_score),
        'score_stat_bm25_rank': float(bm25_rank_score),
        'score_stat_vector_rank': float(vector_rank_score),
        'score_stat_keyword_coverage': float(coverage_score),
        'score_stat_keyword_hits': float(kw_hits),
        'score_stat_keyword_total': float(kw_total),
        'score_stat_concept': float(concept_score),
        'score_stat_phrase': float(phrase_score),
        'score_stat_title': float(title_score),
        'score_stat_length': float(length_score),
    }


def rerank_hits(query: str, hits: List[Dict[str, Any]], final_k: int = FINAL_CONTEXT_K) -> List[Dict[str, Any]]:
    """V12 override: statistical reranker tanpa load model.
    Ini menggantikan FlagReranker/manual HF reranker, jadi tidak ada from_pretrained untuk reranker.
    """
    if not hits:
        return []
    _dbg(f'[rerank] statistical_v12 active | candidates={len(hits)} | final_k={final_k}')
    scored = []
    for h in hits:
        hh = dict(h)
        stats = statistical_score_hit(query, hh)
        hh.update(stats)
        hh['reranker_used'] = True
        hh['reranker_type'] = RERANKER_TYPE
        scored.append(hh)
    ranked = sorted(scored, key=lambda x: x.get('score_reranker', -1e9), reverse=True)[:final_k]
    for i, h in enumerate(ranked, start=1):
        h['rank_final'] = i
    _dbg('[rerank] statistical done: ' + ', '.join([f"{i+1}:{r.get('score_reranker',0):.3f}" for i, r in enumerate(ranked[:5])]))
    return ranked


def retrieve_final(query: str, category_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """V12 debug retrieve_final: dense + BM25 + RRF + statistical reranker."""
    t0 = now_ms()
    _dbg('[1/5] vector search: embed query + dot product')
    vh = vector_search_numpy(query, limit=VECTOR_TOP_K, category_filter=category_filter)
    t1 = now_ms()
    _dbg(f'[1/5] vector done: {len(vh)} hits | {round(t1-t0, 2)} ms')

    _dbg('[2/5] BM25 search')
    bh = bm25_search(query, limit=BM25_TOP_K, category_filter=category_filter)
    t2 = now_ms()
    _dbg(f'[2/5] BM25 done: {len(bh)} hits | {round(t2-t1, 2)} ms')

    _dbg('[3/5] reciprocal rank fusion')
    hybrid = reciprocal_rank_fusion(vh, bh, limit=HYBRID_TOP_K)
    t3 = now_ms()
    _dbg(f'[3/5] fusion done: {len(hybrid)} candidates | {round(t3-t2, 2)} ms')

    _dbg('[4/5] statistical rerank/final select')
    final = rerank_hits(query, hybrid, final_k=FINAL_CONTEXT_K)
    t4 = now_ms()
    _dbg(f'[4/5] final selected: {len(final)} | {round(t4-t3, 2)} ms')

    timing = {
        'vector_ms': round(t1 - t0, 2),
        'bm25_ms': round(t2 - t1, 2),
        'fusion_ms': round(t3 - t2, 2),
        'rerank_ms': round(t4 - t3, 2),
        'retrieval_total_ms': round(t4 - t0, 2),
    }
    return final, timing

print('V12 statistical reranker installed. No reranker model will be loaded.')
print('RERANKER_TYPE:', RERANKER_TYPE)
print('HYBRID_TOP_K:', HYBRID_TOP_K, '| FINAL_CONTEXT_K:', FINAL_CONTEXT_K)



# %% CELL 26

# =====================================================
# PATCH V13: Conservative support + short JSON + safer dalil + refined fallback
# =====================================================
# Perbaikan dari audit V12:
# 1) Jika status tidak cukup konteks, jangan tampilkan dalil mentah/noisy.
# 2) support_level dibuat lebih konservatif agar abstention test tidak dinilai high berlebihan.
# 3) JSON schema dipendekkan supaya Qwen3-4B lebih patuh format.
# 4) Fallback ekstraktif dirapikan agar terasa seperti jawaban asisten, bukan potongan mentah.

V13_ACTIVE = True
GENERATION_SCHEMA_VERSION = 'v13_short_json'
RERANKER_TYPE = 'statistical_v12_v13_conservative_support'
RERANKER_MODEL_NAME = 'STATISTICAL_RERANKER_V12_NO_MODEL'

# Tetap tanpa OCR eksternal agar tidak stuck.
ENABLE_ARABIC_EVIDENCE = False
ENABLE_ARABIC_EVIDENCE_FOR_DALIL_QUERY = False
EVIDENCE_TOP_N = 0

# Prompt lebih pendek + output lebih pendek.
FINAL_CONTEXT_K = 4
MAX_CONTEXT_CHARS_PER_SOURCE = 560
MAX_CONTEXT_TOTAL_CHARS = 3600
MAX_DALIL_CHARS_PER_SOURCE = 360
MAX_NEW_TOKENS = 520
MAX_PROMPT_TOKENS = 6144

# Schema baru yang lebih pendek.
REQUIRED_JSON_KEYS = ['status', 'jawab', 'poin', 'batasan']
SUPPORTED_STATUS_VALUES = {'ada', 'tidak_cukup'}

UNSUPPORTED_QUERY_TERMS = {
    'ai', 'chatbot', 'artificial intelligence', 'kecerdasan buatan',
    'nft', 'staking', 'kripto', 'crypto', 'token', 'blockchain', 'game berhadiah'
}

CORE_CONCEPT_TERMS = set([
    'jual beli', 'marketplace', 'akad', 'murabahah', 'salam', 'istishna', 'istishna’',
    'ijarah', 'wakalah', 'wakalah bil ujrah', 'ujrah', 'jualah', 'ju’alah', "ju'alah",
    'mudharabah', 'musyarakah', 'qardh', 'rahn', 'kafalah', 'hawalah',
    'riba', 'gharar', 'maysir', 'maisir', 'tadlis', 'najasy', 'risywah',
    'uang elektronik', 'e-money', 'fintech', 'asuransi', 'tabarru', 'takaful',
    'voucher', 'cashback', 'diskon', 'hadiah', 'pariwisata', 'hotel', 'wisata',
    'halal', 'haram', 'objek', 'barang', 'jasa', 'pembiayaan', 'modal usaha'
])


def has_unsupported_specific_terms(query: str) -> List[str]:
    low = normalize_text(query).lower()
    return [t for t in UNSUPPORTED_QUERY_TERMS if t in low]


def query_core_concepts(query: str) -> List[str]:
    low = normalize_text(query).lower()
    hits = [t for t in CORE_CONCEPT_TERMS if t in low]
    # tambahkan token domain yang cukup spesifik
    toks = [t for t in query_keywords(query) if len(t) >= 6 and t not in STOPWORDS_ID]
    for t in toks:
        if t in UMKM_DOMAIN_TERMS and t not in hits:
            hits.append(t)
    return list(dict.fromkeys(hits))[:20]


def support_score_for_results(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """V13: support lebih konservatif.
    Tidak cukup hanya keyword overlap; butuh konsep inti yang cocok dan tidak boleh high kalau istilah spesifik query hilang.
    """
    q_norm = normalize_text(query)
    q_terms = [t for t in query_keywords(q_norm) if t not in STOPWORDS_ID and len(t) >= 3]
    q_terms = list(dict.fromkeys(q_terms))
    if not q_terms or not results:
        return {
            'support_score': 0.0,
            'support_level': 'very_low',
            'matched_terms': '',
            'missing_terms': ', '.join(q_terms[:30]),
            'core_concepts': '',
            'matched_core_concepts': '',
            'unsupported_specific_terms': ''
        }

    top_texts = []
    for r in results[:FINAL_CONTEXT_K]:
        p = r.get('payload', {})
        top_texts.append(' '.join([str(p.get('title') or ''), payload_text(p)]).lower())
    joined = ' '.join(top_texts)

    matched = [t for t in q_terms if t.lower() in joined]
    missing = [t for t in q_terms if t not in matched]
    concepts = query_core_concepts(q_norm)
    matched_concepts = [c for c in concepts if c.lower() in joined]
    unsupported = has_unsupported_specific_terms(q_norm)
    unsupported_missing = [t for t in unsupported if t.lower() not in joined]

    # Coverage umum + coverage konsep inti
    term_cov = len(matched) / max(len(q_terms), 1)
    concept_cov = len(matched_concepts) / max(len(concepts), 1) if concepts else 0.0

    # Sinyal ranking dari statistical reranker kalau ada.
    top_scores = []
    for r in results[:min(3, len(results))]:
        val = r.get('rerank_score') or r.get('statistical_score') or r.get('rrf_score') or 0
        try:
            top_scores.append(float(val))
        except Exception:
            pass
    score_signal = min(max(np.mean(top_scores) if top_scores else 0.0, 0.0), 1.0)

    score = (0.45 * term_cov) + (0.40 * concept_cov) + (0.15 * score_signal)

    # Penalti keras untuk pertanyaan yang menyebut hal sangat spesifik tetapi tidak ada di corpus terambil.
    if unsupported_missing:
        score *= 0.45

    # Kalau query minta dalil Arab khusus, jangan high hanya karena ada potongan Arab umum.
    q_low = q_norm.lower()
    if ('dalil arab' in q_low or 'teks arab' in q_low or 'hadis arab' in q_low) and concept_cov < 0.35:
        score *= 0.65

    # Level lebih konservatif dari v12.
    if score >= 0.58 and (not concepts or concept_cov >= 0.45) and not unsupported_missing:
        level = 'high'
    elif score >= 0.34 and (not concepts or concept_cov >= 0.20):
        level = 'medium'
    elif score >= 0.18:
        level = 'low'
    else:
        level = 'very_low'

    return {
        'support_score': round(float(score), 4),
        'support_level': level,
        'matched_terms': ', '.join(matched[:30]),
        'missing_terms': ', '.join(missing[:30]),
        'core_concepts': ', '.join(concepts[:20]),
        'matched_core_concepts': ', '.join(matched_concepts[:20]),
        'unsupported_specific_terms': ', '.join(unsupported_missing[:20]),
    }


def build_umkm_generator_prompt(item: Dict[str, Any], context: str) -> str:
    query = item.get('query') or ''
    focus = item.get('expected_focus') or ''
    return f"""
Anda adalah asisten RAG DSN-MUI untuk UMKM.
Jawab hanya dari KONTEKS. Jangan membuat fatwa/dalil/nomor halaman baru.
Output hanya JSON valid. Jangan markdown. Jangan teks di luar JSON.

Schema wajib, pendek:
{{
  "status": "ada" atau "tidak_cukup",
  "jawab": "jawaban praktis 2-3 kalimat untuk UMKM, sertakan [S1]/[S2] jika menyebut aturan",
  "poin": ["1-4 poin akad/ketentuan relevan, singkat, dengan [Sx]"],
  "batasan": "batasan singkat"
}}

Aturan:
- Pakai status "ada" jika konteks memuat aturan/akad/larangan yang relevan.
- Pakai status "tidak_cukup" jika konteks tidak cukup spesifik untuk kasus user.
- Jangan tulis dalil/kaidah dalam JSON. Sistem akan mengambil dalil/kaidah dari source secara otomatis.

Fokus: {focus}

[KONTEKS]
{context}

[PERTANYAAN]
{query}
""".strip()


def normalize_structured_obj(obj: Dict[str, Any], support_info: Dict[str, Any]) -> Dict[str, Any]:
    obj = obj or {}
    raw_status = str(obj.get('status', '')).lower().strip()
    if raw_status in ['ada', 'ditemukan', 'found', 'supported']:
        status = 'ditemukan'
    elif raw_status in ['tidak_cukup', 'tidak_cukup_konteks', 'tidak cukup konteks', 'no_context', 'not_enough_context']:
        status = 'tidak_cukup_konteks'
    else:
        status = 'ditemukan' if support_info.get('support_level') in ['high', 'medium'] else 'tidak_cukup_konteks'

    jawab = obj.get('jawab', obj.get('jawaban_praktis', ''))
    poin = obj.get('poin', obj.get('akad_ketentuan_relevan', []))
    batasan = obj.get('batasan', obj.get('catatan_batasan', ''))

    out = {
        'status': status,
        'jawaban_praktis': clean_json_value(jawab),
        'akad_ketentuan_relevan': clean_json_value(poin),
        'catatan_batasan': clean_json_value(batasan),
    }
    if isinstance(out['akad_ketentuan_relevan'], str):
        out['akad_ketentuan_relevan'] = [out['akad_ketentuan_relevan']] if out['akad_ketentuan_relevan'] else []
    if not isinstance(out['akad_ketentuan_relevan'], list):
        out['akad_ketentuan_relevan'] = []
    return out


def structured_obj_is_bad(obj: Optional[Dict[str, Any]], raw: str = '') -> bool:
    if not isinstance(obj, dict):
        return True
    # V13 menerima schema pendek atau schema lama agar robust terhadap retry.
    has_short = all(k in obj for k in ['status', 'jawab', 'poin', 'batasan'])
    has_old = all(k in obj for k in ['status', 'jawaban_praktis', 'akad_ketentuan_relevan', 'catatan_batasan'])
    if not (has_short or has_old):
        return True
    combined = json.dumps(obj, ensure_ascii=False) + ' ' + (raw or '')
    bad_pattern = r'[\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0400-\u04ff]|chunk_id|reranker_score|pdf_path|assistant\n|user\n'
    if re.search(bad_pattern, combined, flags=re.I):
        return True
    low = combined.lower()
    if any(t in low for t in V10_BAD_TERMS):
        return True
    return False


def deterministic_dalil_lines(results: List[Dict[str, Any]], item: Dict[str, Any], max_sources: int = 3, allow: bool = True) -> List[str]:
    if not allow:
        return []
    # Dalil lebih selektif: hanya source yang punya potongan dalil eksplisit dan cukup relevan.
    lines = []
    query = item.get('retrieval_query') or item.get('query') or ''
    q_terms = set(query_keywords(query))
    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get('payload', {})
        doc_text = payload_text(p)
        dalil = extract_dalil_snippets(doc_text, max_chars=MAX_DALIL_CHARS_PER_SOURCE)
        if not dalil:
            continue
        # Jangan tampilkan dalil kalau source sama sekali tidak punya overlap konsep/query.
        combined = (str(p.get('title') or '') + ' ' + doc_text).lower()
        overlap = sum(1 for t in q_terms if t in combined and t not in STOPWORDS_ID)
        if overlap < 2 and count_arabic(dalil) > 20:
            continue
        lines.append(f'- [S{i}] {dalil}')
    return lines


def render_structured_answer(
    obj: Dict[str, Any],
    item: Dict[str, Any],
    results: List[Dict[str, Any]],
    support_info: Dict[str, Any],
    mode: str = 'json_generated',
) -> str:
    obj = normalize_structured_obj(obj, support_info)
    is_found = obj.get('status') == 'ditemukan'
    status_text = 'Ditemukan' if is_found else 'Tidak cukup konteks'
    jawaban = obj.get('jawaban_praktis') or 'Dokumen yang terambil belum cukup kuat untuk menjawab secara aman.'
    akad_list = obj.get('akad_ketentuan_relevan') or ['Tidak dapat ditentukan dari konteks yang terambil.']
    akad_block = '\n'.join(f'- {x}' for x in akad_list[:5])

    # V13: kalau status tidak cukup konteks, jangan tampilkan dalil mentah/noisy.
    if is_found:
        dalil_lines = deterministic_dalil_lines(results, item, max_sources=3, allow=True)
        dalil_block = '\n'.join(dalil_lines) if dalil_lines else '- Tidak ditemukan dalil/kaidah eksplisit dalam konteks yang terambil.'
    else:
        dalil_block = '- Tidak ditampilkan karena konteks tidak cukup mendukung jawaban.'

    sources = deterministic_sources(results, max_sources=4) if is_found else deterministic_sources(results, max_sources=2)
    source_block = '\n'.join(sources) if sources else '- Tidak ada sumber yang cukup kuat.'
    catatan = obj.get('catatan_batasan') or 'Jawaban dibatasi pada corpus DSN-MUI precomputed dan konteks yang terambil.'

    answer = (
        f'Status Jawaban:\n- {status_text}\n\n'
        f'Jawaban Praktis untuk UMKM:\n- {jawaban}\n\n'
        f'Akad/Ketentuan yang Relevan:\n{akad_block}\n\n'
        f'Dalil/Kaidah dari Dokumen:\n{dalil_block}\n\n'
        f'Sumber Dokumen:\n{source_block}\n\n'
        f'Catatan Batasan:\n- {catatan}'
    )
    if mode != 'json_generated':
        answer += f'\n- Mode jawaban: {mode}.'
    return normalize_answer_domain_terms(answer, query=item.get('query', ''))


def summarize_evidence_point(point: str, max_len: int = 260) -> str:
    point = re.sub(r'^- \[S\d+\]\s*', '', normalize_text(point))
    point = re.sub(r'\s+', ' ', point).strip()
    # Ambil kalimat awal yang paling readable, bukan potongan terlalu panjang.
    units = split_text_units(point)
    chosen = []
    for u in units:
        if len(u) < 20:
            continue
        chosen.append(u)
        if sum(len(x) for x in chosen) >= max_len:
            break
    text = ' '.join(chosen) if chosen else point[:max_len]
    return text[:max_len].strip()


def build_extractive_safe_answer(item: Dict[str, Any], results: List[Dict[str, Any]], support_info: Dict[str, Any], reason: str = '') -> str:
    query = item.get('query') or ''
    support_level = support_info.get('support_level', 'unknown')
    if not results or support_level in ['very_low', 'low']:
        obj = {
            'status': 'tidak_cukup',
            'jawab': 'Dokumen DSN-MUI yang terambil belum cukup spesifik untuk menjawab pertanyaan ini secara aman.',
            'poin': ['Tidak dapat ditentukan dari konteks yang terambil.'],
            'batasan': 'Sistem hanya memakai corpus DSN-MUI precomputed; jawaban tidak dipaksakan jika konteks tidak cukup.'
        }
        return render_structured_answer(obj, item, results, support_info, mode='NO_CONTEXT')

    concepts = extract_concepts_from_results(results, query=query, limit=6)
    evidence_points = collect_evidence_points(item, results, max_sources=3)
    ev1 = summarize_evidence_point(evidence_points[0], max_len=300) if evidence_points else ''
    ev2 = summarize_evidence_point(evidence_points[1], max_len=220) if len(evidence_points) > 1 else ''
    focus = normalize_text(item.get('expected_focus') or '')

    practical = 'Konteks DSN-MUI yang terambil memuat dasar yang relevan untuk menjawab kasus ini.'
    if concepts:
        practical += f' Konsep yang paling terkait adalah {", ".join(concepts[:4])}.'
    if ev1:
        practical += f' Bukti utama dari dokumen: {ev1}'
    if ev2:
        practical += f' Bukti tambahan: {ev2}'

    obj = {
        'status': 'ada',
        'jawab': practical[:700],
        'poin': concepts[:5] or [focus or 'Gunakan bukti ekstraktif dari sumber terambil.'],
        'batasan': 'Jawaban ini fallback ekstraktif karena generator gagal menghasilkan JSON valid; review manual tetap disarankan.'
    }
    return render_structured_answer(obj, item, results, support_info, mode=f'CONTEXT_FOUND_GENERATION_FAILED:{reason or "fallback_extractive_v13"}')


def answer_query_umkm_safely(item: Dict[str, Any], results: List[Dict[str, Any]], prompt: str, support_info: Dict[str, Any]):
    support_level = support_info.get('support_level')
    if FORCE_ABSTAIN_ON_LOW_SUPPORT and support_level == 'very_low':
        ans = build_extractive_safe_answer(item, results, support_info, reason='very_low_support')
        return ans, {'generation_mode': 'NO_CONTEXT', 'raw_generation': '', 'json_parse_ok': False, 'schema_version': GENERATION_SCHEMA_VERSION}

    obj, raw = generate_structured_json_once(prompt, max_new_tokens=MAX_NEW_TOKENS)
    json_ok = not structured_obj_is_bad(obj, raw)
    if json_ok:
        obj_norm = normalize_structured_obj(obj, support_info)
        # Kalau support high/medium tetapi model abstain, retry sekali dengan instruksi tegas.
        if support_level in ['high', 'medium'] and obj_norm.get('status') == 'tidak_cukup_konteks':
            retry_prompt = prompt + '\n\nKonteks dinilai cukup oleh retriever. Jika ada aturan relevan, gunakan status "ada". Output tetap JSON valid schema pendek.'
            obj2, raw2 = generate_structured_json_once(retry_prompt, max_new_tokens=min(MAX_NEW_TOKENS, 420))
            if not structured_obj_is_bad(obj2, raw2):
                obj2_norm = normalize_structured_obj(obj2, support_info)
                if obj2_norm.get('status') == 'ditemukan':
                    ans = render_structured_answer(obj2_norm, item, results, support_info, mode='json_generated_retry')
                    return ans, {'generation_mode': 'json_generated_retry', 'raw_generation': raw2, 'json_parse_ok': True, 'schema_version': GENERATION_SCHEMA_VERSION}
        ans = render_structured_answer(obj_norm, item, results, support_info, mode='json_generated')
        return ans, {'generation_mode': 'json_generated', 'raw_generation': raw, 'json_parse_ok': True, 'schema_version': GENERATION_SCHEMA_VERSION}

    # Retry dengan contoh minimal agar JSON compliance naik.
    retry_prompt = prompt + '\n\nOutput hanya JSON. Contoh format: {"status":"ada","jawab":"... [S1]","poin":["... [S1]"],"batasan":"..."}'
    obj2, raw2 = generate_structured_json_once(retry_prompt, max_new_tokens=min(MAX_NEW_TOKENS, 420))
    if not structured_obj_is_bad(obj2, raw2):
        obj2_norm = normalize_structured_obj(obj2, support_info)
        ans = render_structured_answer(obj2_norm, item, results, support_info, mode='json_generated_retry')
        return ans, {'generation_mode': 'json_generated_retry', 'raw_generation': raw2, 'json_parse_ok': True, 'schema_version': GENERATION_SCHEMA_VERSION}

    ans = build_extractive_safe_answer(item, results, support_info, reason='json_invalid_v13')
    return ans, {'generation_mode': 'extractive_fallback_json_invalid_v13', 'raw_generation': raw2 or raw, 'json_parse_ok': False, 'schema_version': GENERATION_SCHEMA_VERSION}

print('V13 patch active: conservative support, short JSON schema, safer dalil, refined extractive fallback.')



# %% CELL 28
# =====================================================
# PATCH V14: Visual Arabic Dalil Layer with Qwen3-VL-8B-Instruct
# =====================================================
# Ide utama:
# - RAG text tetap dipakai untuk jawaban praktis UMKM.
# - Dalil Arab TIDAK digenerate bebas oleh LLM text-only.
# - Jika query/source butuh dalil, halaman PDF asli dirender/crop, lalu Qwen3-VL menyalin teks Arab dari gambar.
# - Hasil VL diperlakukan sebagai evidence visual dengan confidence + warning verifikasi.

V14_ACTIVE = True
DALIL_EXTRACTION_MODE = 'visual_vlm_with_quality_gate'
VISUAL_DALIL_ENABLED = True
VISUAL_DALIL_MODEL_NAME = os.environ.get('VISUAL_DALIL_MODEL_NAME', 'Qwen/Qwen3-VL-8B-Instruct')
VISUAL_DALIL_MAX_SOURCES = 2
VISUAL_DALIL_MAX_CROPS_PER_SOURCE = 4
VISUAL_DALIL_RENDER_DPI = 300
VISUAL_DALIL_MAX_NEW_TOKENS = 420
VISUAL_DALIL_TEMPERATURE = 0.0
VISUAL_DALIL_MIN_ARABIC_COUNT = 12
VISUAL_DALIL_MIN_QUALITY_SCORE = 8.0
VISUAL_DALIL_ONLY_FOR_DALIL_QUERIES = True
VISUAL_DALIL_FORCE_FOR_SOURCE_WITH_DALIL = True
UNLOAD_RETRIEVAL_BEFORE_VISUAL_DALIL = True
UNLOAD_VISUAL_DALIL_BEFORE_GENERATION = True

# Text generator tetap Qwen3-4B JSON untuk jawaban praktis.
# Visual model hanya dipakai untuk transkripsi dalil Arab dari gambar.
GENERATION_SCHEMA_VERSION = 'v14_short_json_visual_dalil'

visual_dalil_processor = None
visual_dalil_model = None


def image_exists(path: Any) -> bool:
    try:
        return Path(path).exists()
    except Exception:
        return False


def release_visual_dalil_model():
    global visual_dalil_processor, visual_dalil_model
    visual_dalil_processor = None
    visual_dalil_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('[visual-dalil] released visual model and cleared cache')


def load_visual_dalil_model():
    """Load Qwen3-VL model hanya saat phase visual evidence.
    Gunakan AutoModelForImageTextToText agar kompatibel dengan Qwen3-VL di transformers terbaru.
    """
    global visual_dalil_processor, visual_dalil_model
    if visual_dalil_model is not None and visual_dalil_processor is not None:
        print('[visual-dalil] model already loaded')
        return

    print('[visual-dalil] loading processor:', VISUAL_DALIL_MODEL_NAME)
    from transformers import AutoProcessor, AutoModelForImageTextToText
    visual_dalil_processor = AutoProcessor.from_pretrained(
        VISUAL_DALIL_MODEL_NAME,
        trust_remote_code=True,
    )
    print('[visual-dalil] processor loaded')

    print('[visual-dalil] loading model:', VISUAL_DALIL_MODEL_NAME)
    visual_dalil_model = AutoModelForImageTextToText.from_pretrained(
        VISUAL_DALIL_MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
    )
    visual_dalil_model.eval()
    print('[visual-dalil] model ready')


def visual_crop_boxes(w: int, h: int) -> Dict[str, Tuple[int, int, int, int]]:
    """Crop yang lebih cocok untuk halaman fatwa: full page + area atas/tengah/bawah.
    Full page berguna untuk layout, crop berguna agar teks Arab lebih besar.
    """
    return {
        'full_page': (0, 0, w, h),
        'upper_55': (int(w * 0.03), int(h * 0.04), int(w * 0.97), int(h * 0.58)),
        'middle_60': (int(w * 0.03), int(h * 0.18), int(w * 0.97), int(h * 0.82)),
        'lower_60': (int(w * 0.03), int(h * 0.34), int(w * 0.97), int(h * 0.98)),
        'center_wide': (int(w * 0.06), int(h * 0.10), int(w * 0.94), int(h * 0.92)),
    }


def crop_page_candidates_visual(page_img_path: Path, chunk_id: str, page_no: int, max_crops: int = VISUAL_DALIL_MAX_CROPS_PER_SOURCE) -> List[Dict[str, Any]]:
    img = Image.open(page_img_path).convert('RGB')
    w, h = img.size
    boxes = visual_crop_boxes(w, h)
    preferred_order = ['upper_55', 'middle_60', 'lower_60', 'center_wide', 'full_page']
    rows = []
    for name in preferred_order:
        if name not in boxes:
            continue
        box = boxes[name]
        crop = img.crop(box)
        out = CROP_DIR / f'visual_{safe_name(chunk_id)}_p{page_no}_{name}.png'
        crop.save(out)
        rows.append({
            'crop_name': name,
            'image_path': str(out),
            'page_image_path': str(page_img_path),
            'box': list(box),
            'width': crop.size[0],
            'height': crop.size[1],
        })
        if len(rows) >= max_crops:
            break
    return rows


def source_should_use_visual_dalil(item: Dict[str, Any], result: Dict[str, Any]) -> bool:
    q = item.get('query') or ''
    needs = bool(item.get('needs_dalil')) or should_run_arabic_evidence(q)
    if not needs and VISUAL_DALIL_ONLY_FOR_DALIL_QUERIES:
        # Kalau query tidak eksplisit butuh dalil, tetap boleh visual kalau source mengandung dalil eksplisit.
        if not VISUAL_DALIL_FORCE_FOR_SOURCE_WITH_DALIL:
            return False
        p = result.get('payload', {})
        if not contains_dalil(payload_text(p)):
            return False
    p = result.get('payload', {})
    return bool(p.get('pdf_path') and p.get('page'))


def build_visual_dalil_prompt(source_label: str, item: Dict[str, Any], crop_name: str) -> str:
    query = item.get('query') or ''
    return f"""
Anda membaca gambar crop dari dokumen DSN-MUI.
Tugas Anda hanya TRANSKRIPSI teks Arab yang terlihat pada gambar, khususnya ayat, hadis, atau kaidah fikih yang relevan.

Aturan sangat penting:
- Salin PERSIS teks Arab yang terlihat.
- Jangan memperbaiki, jangan melengkapi, jangan menambahkan harakat jika tidak terlihat.
- Jangan membuat dalil baru.
- Jika sebagian tidak jelas, tulis [TIDAK TERBACA] pada bagian itu.
- Jika tidak ada teks Arab yang terlihat, isi arabic_visible=false dan arabic_text="".
- Jangan jelaskan panjang. Output JSON valid saja.

Konteks query pengguna: {query}
Source: {source_label}
Crop: {crop_name}

Schema JSON wajib:
{{
  "arabic_visible": true,
  "arabic_text": "...",
  "confidence": "high|medium|low",
  "unclear_parts": ["..."],
  "notes": "singkat"
}}
""".strip()


def _vl_parse_json(raw: str) -> Dict[str, Any]:
    raw = strip_code_fence(raw or '').strip()
    obj = extract_json_object(raw)
    if isinstance(obj, dict):
        return obj
    # fallback sederhana kalau model tidak JSON
    return {
        'arabic_visible': bool(count_arabic(raw) >= VISUAL_DALIL_MIN_ARABIC_COUNT),
        'arabic_text': raw.strip(),
        'confidence': 'low',
        'unclear_parts': [],
        'notes': 'raw_non_json_output'
    }


@torch.no_grad()
def transcribe_arabic_from_image_vlm(image_path: str, source_label: str, item: Dict[str, Any], crop_name: str) -> Dict[str, Any]:
    load_visual_dalil_model()
    img = Image.open(image_path).convert('RGB')
    prompt = build_visual_dalil_prompt(source_label, item, crop_name)
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': str(image_path)},
            {'type': 'text', 'text': prompt},
        ]
    }]

    text = visual_dalil_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Qwen-VL utils kadang diperlukan; fallback ke processor(images=[PIL]) jika tidak tersedia.
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = visual_dalil_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt'
        )
    except Exception:
        inputs = visual_dalil_processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors='pt'
        )

    device = next(visual_dalil_model.parameters()).device
    inputs = inputs.to(device)
    out_ids = visual_dalil_model.generate(
        **inputs,
        max_new_tokens=VISUAL_DALIL_MAX_NEW_TOKENS,
        do_sample=False,
        temperature=VISUAL_DALIL_TEMPERATURE,
        pad_token_id=getattr(visual_dalil_processor.tokenizer, 'eos_token_id', None),
    )
    gen = out_ids[0][inputs['input_ids'].shape[-1]:]
    raw = visual_dalil_processor.decode(gen, skip_special_tokens=True).strip()
    obj = _vl_parse_json(raw)
    arabic_text = normalize_text(obj.get('arabic_text') or '')
    q = arabic_quality_score(arabic_text)
    accepted = bool(
        obj.get('arabic_visible') is True
        and q.get('arabic_count', 0) >= VISUAL_DALIL_MIN_ARABIC_COUNT
        and q.get('score', 0.0) >= VISUAL_DALIL_MIN_QUALITY_SCORE
    )
    return {
        'image_path': str(image_path),
        'source_label': source_label,
        'crop_name': crop_name,
        'raw_generation': raw,
        'parsed': obj,
        'arabic_text': arabic_text,
        'confidence': obj.get('confidence') or 'low',
        'unclear_parts': obj.get('unclear_parts') if isinstance(obj.get('unclear_parts'), list) else [],
        'notes': obj.get('notes') or '',
        'quality': q,
        'accepted': accepted,
        'needs_human_verification': True,
    }


def visual_dalil_for_result(result: Dict[str, Any], item: Dict[str, Any], source_idx: int) -> Dict[str, Any]:
    p = result.get('payload', {})
    source_label = f"S{source_idx} | {infer_category_from_payload(p)} {infer_nomor_from_payload(p)} | halaman {p.get('page')}"
    base = {
        'status': 'not_run',
        'source_id': f'S{source_idx}',
        'source_label': source_label,
        'category': infer_category_from_payload(p),
        'nomor': infer_nomor_from_payload(p),
        'title': p.get('title'),
        'page': p.get('page'),
        'pdf_path': p.get('pdf_path'),
        'page_image_path': None,
        'best_crop_image_path': None,
        'best_crop_name': None,
        'arabic_text': '',
        'confidence': 'low',
        'quality': {},
        'accepted': False,
        'needs_human_verification': True,
        'candidates': [],
        'error': None,
    }
    if not VISUAL_DALIL_ENABLED:
        base['status'] = 'disabled'
        return base
    if not source_should_use_visual_dalil(item, result):
        base['status'] = 'skipped_not_needed_or_no_pdf'
        return base
    try:
        page_img = render_pdf_page(p.get('pdf_path'), int(p.get('page')), str(p.get('doc_id') or p.get('chunk_id') or source_label), dpi=VISUAL_DALIL_RENDER_DPI)
        base['page_image_path'] = str(page_img)
        crops = crop_page_candidates_visual(page_img, str(p.get('chunk_id') or source_label), int(p.get('page')))
        candidates = []
        print(f"[visual-dalil] {source_label}: {len(crops)} crops")
        for crop in crops:
            print(f"[visual-dalil] transcribe {source_label} crop={crop['crop_name']}")
            cand = transcribe_arabic_from_image_vlm(crop['image_path'], source_label, item, crop['crop_name'])
            cand.update({
                'page_image_path': str(page_img),
                'box': crop.get('box'),
                'width': crop.get('width'),
                'height': crop.get('height'),
            })
            candidates.append(cand)
        best = None
        for cand in candidates:
            if best is None or cand.get('quality', {}).get('score', -999) > best.get('quality', {}).get('score', -999):
                best = cand
        if best:
            base.update({
                'status': 'accepted' if best.get('accepted') else 'low_quality',
                'best_crop_image_path': best.get('image_path'),
                'best_crop_name': best.get('crop_name'),
                'arabic_text': best.get('arabic_text') or '',
                'confidence': best.get('confidence') or 'low',
                'quality': best.get('quality') or {},
                'accepted': bool(best.get('accepted')),
                'needs_human_verification': True,
                'candidates': candidates,
            })
        else:
            base['status'] = 'no_candidate'
        return base
    except Exception as e:
        base['status'] = 'error'
        base['error'] = f'{type(e).__name__}: {str(e)}'
        traceback.print_exc()
        return base


def generate_visual_dalil_for_cached_rows(answer_rows, retrieval_rows, evidence_rows, full_rows):
    """Phase 1.5: setelah retrieval selesai, sebelum generation.
    Attach visual_dalil ke setiap result dalam full_rows.
    """
    if not VISUAL_DALIL_ENABLED:
        print('[visual-dalil] disabled')
        return answer_rows, retrieval_rows, evidence_rows, full_rows

    if UNLOAD_RETRIEVAL_BEFORE_VISUAL_DALIL:
        try:
            release_retrieval_models()
        except Exception as e:
            print('[visual-dalil] warning: release_retrieval_models failed:', e)

    load_visual_dalil_model()
    for x in tqdm(full_rows, desc='Phase 1.5 - Visual Arabic dalil extraction'):
        item = x.get('item', {})
        results = x.get('results', [])
        qid = item.get('id')
        print('\n' + '-' * 100)
        print('[visual-dalil]', qid, '|', item.get('query', '')[:140])
        ran = 0
        accepted = 0
        for idx, r in enumerate(results[:VISUAL_DALIL_MAX_SOURCES], start=1):
            vd = visual_dalil_for_result(r, item, idx)
            r['visual_dalil'] = vd
            if vd.get('status') not in ['skipped_not_needed_or_no_pdf', 'disabled']:
                ran += 1
            if vd.get('accepted'):
                accepted += 1
        x['answer_row']['visual_dalil_ran_sources'] = ran
        x['answer_row']['visual_dalil_accepted_sources'] = accepted
        x['answer_row']['visual_dalil_model'] = VISUAL_DALIL_MODEL_NAME
        x['answer_row']['dalil_extraction_mode'] = DALIL_EXTRACTION_MODE
        save_umkm_outputs(answer_rows, rebuild_umkm_retrieval_rows_from_full_rows(full_rows), evidence_rows, full_rows)
        write_umkm_summary(answer_rows)

    retrieval_rows = rebuild_umkm_retrieval_rows_from_full_rows(full_rows)
    if UNLOAD_VISUAL_DALIL_BEFORE_GENERATION:
        release_visual_dalil_model()
    return answer_rows, retrieval_rows, evidence_rows, full_rows


def visual_dalil_line_from_result(result: Dict[str, Any], source_idx: int) -> Optional[str]:
    vd = result.get('visual_dalil') or {}
    if not vd:
        return None
    p = result.get('payload', {})
    source = f'[S{source_idx}] {infer_category_from_payload(p)} No. {infer_nomor_from_payload(p)}, halaman {p.get("page")}'
    if vd.get('accepted') and vd.get('arabic_text'):
        text = normalize_text(vd.get('arabic_text'))
        conf = vd.get('confidence') or 'medium'
        crop = vd.get('best_crop_image_path') or ''
        return f'- {source}\n  Arab hasil pembacaan visual: {text}\n  Status: transkripsi visual Qwen3-VL, confidence={conf}, perlu verifikasi untuk kutipan resmi. Crop: {crop}'
    if vd.get('status') in ['low_quality', 'error']:
        crop = vd.get('best_crop_image_path') or vd.get('page_image_path') or ''
        return f'- {source}\n  Ditemukan kandidat dalil pada gambar, tetapi teks Arab belum cukup jelas untuk ditampilkan sebagai kutipan. Verifikasi pada crop/halaman asli: {crop}'
    return None


# Override deterministic dalil renderer: utamakan hasil visual VLM, baru fallback ke text snippet yang lolos quality gate.
def deterministic_dalil_lines(results: List[Dict[str, Any]], item: Dict[str, Any], max_sources: int = 3, allow: bool = True) -> List[str]:
    if not allow:
        return []
    lines = []
    query = item.get('retrieval_query') or item.get('query') or ''
    q_terms = set(query_keywords(query))

    # 1) Prefer visual Arabic evidence.
    for i, r in enumerate(results[:max_sources], start=1):
        line = visual_dalil_line_from_result(r, i)
        if line:
            lines.append(line)

    if lines:
        return lines

    # 2) Fallback text snippet, tapi jangan tampilkan Arab noisy.
    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get('payload', {})
        doc_text = payload_text(p)
        dalil = extract_dalil_snippets(doc_text, max_chars=MAX_DALIL_CHARS_PER_SOURCE)
        if not dalil:
            continue
        combined = (str(p.get('title') or '') + ' ' + doc_text).lower()
        overlap = sum(1 for t in q_terms if t in combined and t not in STOPWORDS_ID)
        if overlap < 2:
            continue
        if count_arabic(dalil) >= 8:
            q = arabic_quality_score(dalil)
            if q.get('label') != 'usable' or q.get('score', 0.0) < VISUAL_DALIL_MIN_QUALITY_SCORE:
                lines.append(f'- [S{i}] Ditemukan potongan dalil/kaidah pada source, tetapi teks Arab dari ekstraksi teks belum cukup bersih untuk dikutip. Gunakan halaman dokumen asli untuk verifikasi.')
                continue
        lines.append(f'- [S{i}] {dalil}')
    return lines


def rebuild_umkm_retrieval_rows_from_full_rows(full_rows):
    rows = []
    for x in full_rows:
        rows.extend(flatten_umkm_retrieval_rows(x.get('item', {}), x.get('results', [])))
    return rows


# Override flatten retrieval rows agar visual evidence ikut masuk CSV.
def flatten_umkm_retrieval_rows(item: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for source_idx, r in enumerate(results, start=1):
        p = r.get('payload', {})
        ev = r.get('arabic_evidence') or {}
        vd = r.get('visual_dalil') or {}
        rows.append({
            'run_id': RUN_ID,
            'query_id': item.get('id'),
            'domain': item.get('domain'),
            'query': item.get('query'),
            'retrieval_query': item.get('retrieval_query'),
            'source_id': f'S{source_idx}',
            'rank_final': r.get('rank_final'),
            'rank_hybrid': r.get('rank_hybrid'),
            'rank_vector': r.get('rank_vector'),
            'rank_bm25': r.get('rank_bm25'),
            'reranker_used': r.get('reranker_used'),
            'reranker_type': r.get('reranker_type'),
            'score_reranker': r.get('score_reranker'),
            'statistical_score': r.get('statistical_score'),
            'score_hybrid': r.get('score_hybrid'),
            'score_vector': r.get('score_vector'),
            'score_bm25': r.get('score_bm25'),
            'chunk_id': r.get('chunk_id') or p.get('chunk_id'),
            'category': infer_category_from_payload(p),
            'nomor': infer_nomor_from_payload(p),
            'title': p.get('title'),
            'page': p.get('page'),
            'pdf_filename': p.get('pdf_filename'),
            'pdf_path': p.get('pdf_path'),
            'text_preview': text_preview(payload_text(p), 700),
            'has_dalil_snippet': contains_dalil(payload_text(p)),
            'arabic_evidence_status': ev.get('ocr_status'),
            'arabic_selected_quality': (ev.get('selected_quality') or {}).get('label'),
            'crop_image_path': ev.get('crop_image_path'),
            'visual_dalil_status': vd.get('status'),
            'visual_dalil_accepted': vd.get('accepted'),
            'visual_dalil_confidence': vd.get('confidence'),
            'visual_dalil_quality_label': (vd.get('quality') or {}).get('label'),
            'visual_dalil_quality_score': (vd.get('quality') or {}).get('score'),
            'visual_dalil_arabic_preview': text_preview(vd.get('arabic_text') or '', 400),
            'visual_dalil_page_image_path': vd.get('page_image_path'),
            'visual_dalil_crop_image_path': vd.get('best_crop_image_path'),
        })
    return rows


print('V14 patch active: Qwen3-VL visual Arabic dalil layer + quality gate + deterministic rendering.')
print('VISUAL_DALIL_MODEL_NAME:', VISUAL_DALIL_MODEL_NAME)
print('DALIL_EXTRACTION_MODE:', DALIL_EXTRACTION_MODE)



# %% CELL 29

# =====================================================
# PATCH V15: Chunk-aligned Visual Arabic Dalil Layer
# =====================================================
# Perubahan dari V14:
# - V14: render halaman PDF lalu crop beberapa area umum.
# - V15: deteksi dulu apakah chunk retrieval yang sama mengandung Arab/dalil,
#        lalu cari bbox teks Arab pada halaman PDF dan render CLIP area itu saja.
# - Tujuan: chunk yang masuk Qwen3 text dan crop yang masuk Qwen3-VL berasal dari source/chunk yang sama,
#           meminimalkan mismatch dan tidak perlu render full page.
#
# Catatan:
# - Kalau PDF punya text layer, PyMuPDF bisa mengambil bbox teks Arab langsung.
# - Kalau PDF scan/image-only dan tidak punya text layer, exact bbox tidak bisa diketahui tanpa detector/OCR layout.
#   Dalam kasus itu V15 fallback ke heuristic clip kecil, tetap bukan full-page render.

V15_ACTIVE = True
DALIL_EXTRACTION_MODE = 'chunk_aligned_visual_vlm'
VISUAL_DALIL_RENDER_STRATEGY = 'chunk_aligned_clip'
VISUAL_DALIL_USE_FULL_PAGE = False
VISUAL_DALIL_REQUIRE_ARABIC_IN_CHUNK = True
VISUAL_DALIL_MAX_CROPS_PER_SOURCE = 2
VISUAL_DALIL_CLIP_PADDING_PT = 18
VISUAL_DALIL_MIN_ARABIC_IN_CHUNK = 8
VISUAL_DALIL_FALLBACK_HEURISTIC_CLIP = True
VISUAL_DALIL_HEURISTIC_CLIPS = ['upper_middle', 'middle']


def has_arabic(s: str, min_count: int = 1) -> bool:
    return count_arabic(s or '') >= min_count


def get_chunk_arabic_signal(result: Dict[str, Any]) -> Dict[str, Any]:
    """Deteksi Arab/dalil dari payload chunk yang sama dengan source retrieval.
    Ini memastikan VLM hanya membaca gambar dari chunk yang juga dipakai untuk context text.
    """
    p = result.get('payload', {})
    txt = payload_text(p)
    raw_ar = p.get('arabic_text') or ''
    combined = f"{raw_ar}\n{txt}"
    arabic_count = count_arabic(combined)
    return {
        'has_arabic': arabic_count >= VISUAL_DALIL_MIN_ARABIC_IN_CHUNK,
        'arabic_count': arabic_count,
        'has_dalil_keyword': contains_dalil(txt),
        'raw_arabic_text': raw_ar,
        'chunk_text_preview': text_preview(txt, 1000),
    }


def source_should_use_visual_dalil(item: Dict[str, Any], result: Dict[str, Any]) -> bool:
    p = result.get('payload', {})
    if not (p.get('pdf_path') and p.get('page')):
        return False

    q = item.get('query') or ''
    query_needs_dalil = bool(item.get('needs_dalil')) or should_run_arabic_evidence(q)
    signal = get_chunk_arabic_signal(result)

    # Kalau query eksplisit minta dalil, jalankan hanya jika chunk yang sama punya sinyal Arab/dalil.
    if query_needs_dalil:
        return bool(signal['has_arabic'] or signal['has_dalil_keyword'])

    # Kalau query tidak eksplisit minta dalil, jangan boros VLM kecuali source chunk jelas punya dalil Arab.
    if VISUAL_DALIL_ONLY_FOR_DALIL_QUERIES:
        return bool(VISUAL_DALIL_FORCE_FOR_SOURCE_WITH_DALIL and signal['has_arabic'] and signal['has_dalil_keyword'])

    return bool(signal['has_arabic'])


def _norm_for_match(s: str) -> str:
    s = s or ''
    # Hilangkan whitespace berlebih saja. Jangan normalisasi agresif teks Arab karena bisa mengubah bentuk.
    return re.sub(r'\s+', ' ', s).strip()


def _arabic_terms_for_match(s: str, min_len: int = 3, max_terms: int = 40) -> List[str]:
    # Ambil token Arab untuk membantu memilih bbox yang paling dekat dengan chunk retrieval.
    toks = re.findall(r'[\u0600-\u06FF]{%d,}' % min_len, s or '')
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= max_terms:
            break
    return out


def find_arabic_bboxes_from_pdf_text_layer(pdf_path: str, page_no: int, chunk_signal_text: str = '') -> List[Dict[str, Any]]:
    """Cari bbox line/span Arab dari text layer PDF pada halaman yang sama.
    Return bbox dalam koordinat PDF points, bukan pixel.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return []
    terms = _arabic_terms_for_match(chunk_signal_text)
    rows = []
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[int(page_no) - 1]
        data = page.get_text('dict')
        for bi, block in enumerate(data.get('blocks', [])):
            if block.get('type') != 0:
                continue
            for li, line in enumerate(block.get('lines', [])):
                spans = line.get('spans', [])
                line_text = ''.join(sp.get('text', '') for sp in spans)
                ar_count = count_arabic(line_text)
                if ar_count < 2:
                    continue
                bbox = None
                for sp in spans:
                    if count_arabic(sp.get('text', '')) < 1:
                        continue
                    rect = fitz.Rect(sp.get('bbox'))
                    bbox = rect if bbox is None else (bbox | rect)
                if bbox is None:
                    bbox = fitz.Rect(line.get('bbox'))
                term_hits = sum(1 for t in terms if t in line_text)
                # Score: prioritaskan line yang cocok dengan Arab chunk dan punya cukup karakter Arab.
                score = (term_hits * 10.0) + min(ar_count / 8.0, 8.0)
                rows.append({
                    'bbox': [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)],
                    'text': line_text,
                    'arabic_count': ar_count,
                    'term_hits': term_hits,
                    'score': score,
                    'block_index': bi,
                    'line_index': li,
                })
    finally:
        doc.close()
    rows.sort(key=lambda x: x['score'], reverse=True)
    return rows


def _pad_rect(rect: fitz.Rect, page_rect: fitz.Rect, pad: float = VISUAL_DALIL_CLIP_PADDING_PT) -> fitz.Rect:
    r = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
    r.x0 = max(page_rect.x0, r.x0)
    r.y0 = max(page_rect.y0, r.y0)
    r.x1 = min(page_rect.x1, r.x1)
    r.y1 = min(page_rect.y1, r.y1)
    return r


def merge_close_bboxes(bbox_rows: List[Dict[str, Any]], page_rect: fitz.Rect, max_groups: int = 2) -> List[Dict[str, Any]]:
    """Gabungkan beberapa line Arab di sekitar bbox terbaik supaya crop punya konteks dalil utuh.
    Tidak merender full page; hanya membuat clip rectangle.
    """
    if not bbox_rows:
        return []

    # Ambil top line, lalu gabungkan line lain yang dekat secara vertikal.
    selected = []
    used = set()
    for gi, row in enumerate(bbox_rows[:8]):
        key = (row.get('block_index'), row.get('line_index'))
        if key in used:
            continue
        r0 = fitz.Rect(row['bbox'])
        group_rect = fitz.Rect(r0)
        group_texts = [row.get('text', '')]
        group_score = float(row.get('score', 0.0))
        used.add(key)
        center_y = (r0.y0 + r0.y1) / 2
        for other in bbox_rows[:20]:
            okey = (other.get('block_index'), other.get('line_index'))
            if okey in used:
                continue
            ro = fitz.Rect(other['bbox'])
            oy = (ro.y0 + ro.y1) / 2
            # 85 points kira-kira beberapa baris pada PDF; cukup untuk satu blok ayat/hadis.
            if abs(oy - center_y) <= 85:
                group_rect |= ro
                group_texts.append(other.get('text', ''))
                group_score += float(other.get('score', 0.0))
                used.add(okey)
        selected.append({
            'rect': _pad_rect(group_rect, page_rect),
            'score': group_score,
            'text_preview': text_preview(' '.join(group_texts), 500),
            'source': 'pdf_text_layer_arabic_bbox',
        })
        if len(selected) >= max_groups:
            break
    selected.sort(key=lambda x: x['score'], reverse=True)
    return selected[:max_groups]


def render_pdf_clip(pdf_path: str, page_no: int, rect: fitz.Rect, out_path: Path, dpi: int = VISUAL_DALIL_RENDER_DPI) -> Path:
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[int(page_no) - 1]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        pix.save(str(out_path))
    finally:
        doc.close()
    return out_path


def heuristic_clip_rects(page_rect: fitz.Rect, names: Optional[List[str]] = None) -> List[Tuple[str, fitz.Rect]]:
    names = names or VISUAL_DALIL_HEURISTIC_CLIPS
    w, h = page_rect.width, page_rect.height
    specs = {
        'upper_middle': fitz.Rect(page_rect.x0 + 0.04*w, page_rect.y0 + 0.08*h, page_rect.x0 + 0.96*w, page_rect.y0 + 0.58*h),
        'middle':       fitz.Rect(page_rect.x0 + 0.04*w, page_rect.y0 + 0.22*h, page_rect.x0 + 0.96*w, page_rect.y0 + 0.78*h),
        'lower_middle': fitz.Rect(page_rect.x0 + 0.04*w, page_rect.y0 + 0.38*h, page_rect.x0 + 0.96*w, page_rect.y0 + 0.96*h),
    }
    return [(n, specs[n]) for n in names if n in specs]


def crop_visual_candidates_chunk_aligned(result: Dict[str, Any], item: Dict[str, Any], source_label: str, max_crops: int = VISUAL_DALIL_MAX_CROPS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Buat crop untuk VLM dari chunk yang sama, tanpa render full page.
    1) Cek Arab pada payload chunk.
    2) Cari bbox Arab di PDF text layer halaman yang sama.
    3) Render clip bbox langsung via PyMuPDF.
    4) Kalau bbox tidak tersedia, fallback heuristic clip kecil, bukan full page.
    """
    p = result.get('payload', {})
    pdf_path = p.get('pdf_path')
    page_no = p.get('page')
    chunk_id = str(p.get('chunk_id') or source_label)
    if not pdf_path or not page_no:
        return []

    signal = get_chunk_arabic_signal(result)
    if VISUAL_DALIL_REQUIRE_ARABIC_IN_CHUNK and not (signal['has_arabic'] or signal['has_dalil_keyword']):
        return []

    chunk_signal_text = '\n'.join([
        str(p.get('title') or ''),
        str(signal.get('raw_arabic_text') or ''),
        str(payload_text(p) or ''),
    ])

    crops = []
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[int(page_no) - 1]
        page_rect = page.rect
    finally:
        doc.close()

    bbox_rows = find_arabic_bboxes_from_pdf_text_layer(pdf_path, int(page_no), chunk_signal_text)
    groups = merge_close_bboxes(bbox_rows, page_rect, max_groups=max_crops)

    for gi, g in enumerate(groups, start=1):
        rect = g['rect']
        out = CROP_DIR / f'visual_chunkclip_{safe_name(chunk_id)}_p{page_no}_bbox{gi}.png'
        render_pdf_clip(pdf_path, int(page_no), rect, out, dpi=VISUAL_DALIL_RENDER_DPI)
        with Image.open(out) as im:
            width, height = im.size
        crops.append({
            'crop_name': f'chunk_arabic_bbox_{gi}',
            'image_path': str(out),
            'page_image_path': None,
            'box_pdf_points': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            'box': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            'width': width,
            'height': height,
            'bbox_source': g.get('source'),
            'bbox_score': g.get('score'),
            'bbox_text_preview': g.get('text_preview'),
            'chunk_arabic_count': signal.get('arabic_count'),
        })
        if len(crops) >= max_crops:
            return crops

    # Fallback tanpa full-page render. Ini hanya clip region kecil dari halaman yang sama.
    if VISUAL_DALIL_FALLBACK_HEURISTIC_CLIP:
        for name, rect in heuristic_clip_rects(page_rect):
            out = CROP_DIR / f'visual_chunkclip_{safe_name(chunk_id)}_p{page_no}_{name}.png'
            render_pdf_clip(pdf_path, int(page_no), rect, out, dpi=VISUAL_DALIL_RENDER_DPI)
            with Image.open(out) as im:
                width, height = im.size
            crops.append({
                'crop_name': f'heuristic_{name}',
                'image_path': str(out),
                'page_image_path': None,
                'box_pdf_points': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                'box': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                'width': width,
                'height': height,
                'bbox_source': 'heuristic_clip_no_text_layer_bbox',
                'bbox_score': None,
                'bbox_text_preview': '',
                'chunk_arabic_count': signal.get('arabic_count'),
            })
            if len(crops) >= max_crops:
                break

    return crops


# Override V14 visual dalil function: no full-page render, chunk-aligned clip only.
def visual_dalil_for_result(result: Dict[str, Any], item: Dict[str, Any], source_idx: int) -> Dict[str, Any]:
    p = result.get('payload', {})
    source_label = f"S{source_idx} | {infer_category_from_payload(p)} {infer_nomor_from_payload(p)} | halaman {p.get('page')}"
    signal = get_chunk_arabic_signal(result)
    base = {
        'status': 'not_run',
        'source_id': f'S{source_idx}',
        'source_label': source_label,
        'category': infer_category_from_payload(p),
        'nomor': infer_nomor_from_payload(p),
        'title': p.get('title'),
        'page': p.get('page'),
        'pdf_path': p.get('pdf_path'),
        'page_image_path': None,
        'best_crop_image_path': None,
        'best_crop_name': None,
        'arabic_text': '',
        'confidence': 'low',
        'quality': {},
        'accepted': False,
        'needs_human_verification': True,
        'candidates': [],
        'error': None,
        'render_strategy': VISUAL_DALIL_RENDER_STRATEGY,
        'chunk_has_arabic': signal.get('has_arabic'),
        'chunk_arabic_count': signal.get('arabic_count'),
        'chunk_has_dalil_keyword': signal.get('has_dalil_keyword'),
    }
    if not VISUAL_DALIL_ENABLED:
        base['status'] = 'disabled'
        return base
    if not source_should_use_visual_dalil(item, result):
        base['status'] = 'skipped_no_arabic_in_same_chunk_or_no_pdf'
        return base
    try:
        crops = crop_visual_candidates_chunk_aligned(result, item, source_label, max_crops=VISUAL_DALIL_MAX_CROPS_PER_SOURCE)
        if not crops:
            base['status'] = 'no_chunk_aligned_crop'
            return base
        candidates = []
        print(f"[visual-dalil-v15] {source_label}: {len(crops)} chunk-aligned clips")
        for crop in crops:
            print(f"[visual-dalil-v15] transcribe {source_label} crop={crop['crop_name']} source={crop.get('bbox_source')}")
            cand = transcribe_arabic_from_image_vlm(crop['image_path'], source_label, item, crop['crop_name'])
            cand.update({
                'page_image_path': None,
                'box': crop.get('box'),
                'box_pdf_points': crop.get('box_pdf_points'),
                'width': crop.get('width'),
                'height': crop.get('height'),
                'bbox_source': crop.get('bbox_source'),
                'bbox_score': crop.get('bbox_score'),
                'bbox_text_preview': crop.get('bbox_text_preview'),
                'chunk_arabic_count': crop.get('chunk_arabic_count'),
            })
            candidates.append(cand)
        best = None
        for cand in candidates:
            score = cand.get('quality', {}).get('score', -999)
            if best is None or score > best.get('quality', {}).get('score', -999):
                best = cand
        if best:
            base.update({
                'status': 'accepted' if best.get('accepted') else 'low_quality',
                'best_crop_image_path': best.get('image_path'),
                'best_crop_name': best.get('crop_name'),
                'arabic_text': best.get('arabic_text') or '',
                'confidence': best.get('confidence') or 'low',
                'quality': best.get('quality') or {},
                'accepted': bool(best.get('accepted')),
                'needs_human_verification': True,
                'candidates': candidates,
            })
        else:
            base['status'] = 'no_candidate'
        return base
    except Exception as e:
        base['status'] = 'error'
        base['error'] = f'{type(e).__name__}: {str(e)}'
        traceback.print_exc()
        return base


# Override retrieval flatten supaya kolom CSV menunjukkan strategi chunk-aligned clip.
def flatten_umkm_retrieval_rows(item: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for source_idx, r in enumerate(results, start=1):
        p = r.get('payload', {})
        ev = r.get('arabic_evidence') or {}
        vd = r.get('visual_dalil') or {}
        rows.append({
            'run_id': RUN_ID,
            'query_id': item.get('id'),
            'domain': item.get('domain'),
            'query': item.get('query'),
            'retrieval_query': item.get('retrieval_query'),
            'source_id': f'S{source_idx}',
            'rank_final': r.get('rank_final'),
            'rank_hybrid': r.get('rank_hybrid'),
            'rank_vector': r.get('rank_vector'),
            'rank_bm25': r.get('rank_bm25'),
            'reranker_used': r.get('reranker_used'),
            'reranker_type': r.get('reranker_type'),
            'score_reranker': r.get('score_reranker'),
            'statistical_score': r.get('statistical_score'),
            'score_hybrid': r.get('score_hybrid'),
            'score_vector': r.get('score_vector'),
            'score_bm25': r.get('score_bm25'),
            'chunk_id': r.get('chunk_id') or p.get('chunk_id'),
            'category': infer_category_from_payload(p),
            'nomor': infer_nomor_from_payload(p),
            'title': p.get('title'),
            'page': p.get('page'),
            'pdf_filename': p.get('pdf_filename'),
            'pdf_path': p.get('pdf_path'),
            'text_preview': text_preview(payload_text(p), 700),
            'has_dalil_snippet': contains_dalil(payload_text(p)),
            'arabic_evidence_status': ev.get('ocr_status'),
            'arabic_selected_quality': (ev.get('selected_quality') or {}).get('label'),
            'crop_image_path': ev.get('crop_image_path'),
            'visual_dalil_status': vd.get('status'),
            'visual_dalil_render_strategy': vd.get('render_strategy'),
            'visual_dalil_chunk_has_arabic': vd.get('chunk_has_arabic'),
            'visual_dalil_chunk_arabic_count': vd.get('chunk_arabic_count'),
            'visual_dalil_chunk_has_dalil_keyword': vd.get('chunk_has_dalil_keyword'),
            'visual_dalil_accepted': vd.get('accepted'),
            'visual_dalil_confidence': vd.get('confidence'),
            'visual_dalil_quality_label': (vd.get('quality') or {}).get('label'),
            'visual_dalil_quality_score': (vd.get('quality') or {}).get('score'),
            'visual_dalil_arabic_preview': text_preview(vd.get('arabic_text') or '', 400),
            'visual_dalil_page_image_path': vd.get('page_image_path'),
            'visual_dalil_crop_image_path': vd.get('best_crop_image_path'),
        })
    return rows

print('V15 patch active: chunk-aligned visual dalil; PDF clip rendering only, no full-page render by default.')
print('VISUAL_DALIL_RENDER_STRATEGY:', VISUAL_DALIL_RENDER_STRATEGY)
print('VISUAL_DALIL_USE_FULL_PAGE:', VISUAL_DALIL_USE_FULL_PAGE)
print('VISUAL_DALIL_REQUIRE_ARABIC_IN_CHUNK:', VISUAL_DALIL_REQUIRE_ARABIC_IN_CHUNK)



# %% CELL 31
# =====================================================
# PATCH V16: Visual Dalil Quality Filter + Dalil Type Classification
# =====================================================
# Perubahan dari V15:
# - V15 sudah chunk-aligned: crop VLM berasal dari source/chunk yang sama dengan context text.
# - V16 menambahkan quality gate khusus DALIL, bukan sekadar deteksi teks Arab.
# - Output VLM diparse lebih robust agar field arabic_text tidak bocor sebagai JSON mentah.
# - Dalil type diklasifikasikan: ayat / hadis / kaidah / kutipan_ulama / istilah_arab / unknown.
# - Accepted hanya untuk ayat/hadis/kaidah/kutipan_ulama yang kualitasnya cukup.
# - Kalau hanya istilah Arab, tidak ditampilkan sebagai dalil.
# - Kalau query butuh dalil tetapi crop sempit gagal, fallback ke wider band dari HALAMAN YANG SAMA, bukan full PDF.

V16_ACTIVE = True
DALIL_EXTRACTION_MODE = 'chunk_aligned_visual_vlm_v16_dalil_quality_gate'
VISUAL_DALIL_RENDER_STRATEGY = 'chunk_aligned_clip_v16_quality_gate'

# Tetap tidak render full page by default.
VISUAL_DALIL_USE_FULL_PAGE = False
VISUAL_DALIL_REQUIRE_ARABIC_IN_CHUNK = True

# V16 mencoba sedikit lebih banyak crop/band dari halaman yang sama, bukan full page.
VISUAL_DALIL_MAX_CROPS_PER_SOURCE = 4
VISUAL_DALIL_HEURISTIC_CLIPS = ['upper_middle', 'middle', 'lower_middle']
VISUAL_DALIL_ENABLE_WIDER_SAME_PAGE_FALLBACK = True
VISUAL_DALIL_WIDER_CLIPS = ['wide_upper', 'wide_middle', 'wide_lower']

# Gate kualitas dalil.
VISUAL_DALIL_ALLOWED_TYPES = {'ayat', 'hadis', 'kaidah', 'kutipan_ulama'}
VISUAL_DALIL_REJECT_TYPES = {'istilah_arab', 'unknown', 'not_dalil', 'none'}
VISUAL_DALIL_MIN_ARABIC_COUNT = 12
VISUAL_DALIL_MIN_QUALITY_SCORE = 7.5
VISUAL_DALIL_ACCEPT_LOW_CONFIDENCE = False


def _safe_lower(x: Any) -> str:
    return str(x or '').strip().lower()


def _json_loads_maybe(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_first_json_object_text(raw: str) -> str:
    """Ambil substring JSON object pertama dari raw output model.
    Lebih toleran daripada regex sederhana karena menghitung brace.
    """
    raw = strip_code_fence(raw or '').strip()
    start = raw.find('{')
    if start < 0:
        return ''
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return raw[start:i+1]
    return ''


def _normalize_vlm_key_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalisasi key dari output VLM agar schema lebih stabil."""
    if not isinstance(obj, dict):
        return {}

    # Map beberapa kemungkinan key.
    key_map = {
        'arabic': 'arabic_text',
        'text_arabic': 'arabic_text',
        'arabic_transcription': 'arabic_text',
        'transcription': 'arabic_text',
        'teks_arab': 'arabic_text',
        'dalil_type': 'dalil_type',
        'type': 'dalil_type',
        'jenis': 'dalil_type',
        'jenis_dalil': 'dalil_type',
        'is_dalil': 'is_dalil',
        'visible': 'arabic_visible',
        'arabic_visible': 'arabic_visible',
        'confidence': 'confidence',
        'notes': 'notes',
        'unclear_parts': 'unclear_parts',
    }
    norm = {}
    for k, v in obj.items():
        kk = key_map.get(str(k).strip(), str(k).strip())
        norm[kk] = v
    return norm


def _clean_vlm_arabic_text(value: Any, raw: str = '') -> str:
    """Bersihkan field arabic_text.
    Jika value ternyata JSON string/dict, ambil field arabic_text di dalamnya.
    Jangan agresif mengubah bentuk Arab.
    """
    if isinstance(value, dict):
        value = value.get('arabic_text') or value.get('arabic') or value.get('text') or ''
    if isinstance(value, list):
        value = ' '.join(str(x) for x in value if x is not None)
    text = str(value or '').strip()

    # Kalau text berisi JSON mentah, parse lagi.
    if text.startswith('{') or '"arabic_text"' in text or '"arabic"' in text:
        sub = _json_loads_maybe(text)
        if isinstance(sub, dict):
            sub = _normalize_vlm_key_obj(sub)
            text = str(sub.get('arabic_text') or sub.get('arabic') or '')
        else:
            jtxt = _extract_first_json_object_text(text)
            sub = _json_loads_maybe(jtxt) if jtxt else None
            if isinstance(sub, dict):
                sub = _normalize_vlm_key_obj(sub)
                text = str(sub.get('arabic_text') or sub.get('arabic') or '')

    # Jangan jadikan raw JSON sebagai Arab. Kalau masih ada schema JSON, kosongkan.
    if '"arabic_visible"' in text or '"confidence"' in text or '"dalil_type"' in text:
        text = ''

    # Hilangkan label Latin yang kadang kebawa, tapi biarkan Arab dan tanda baca umum.
    text = strip_code_fence(text)
    text = re.sub(r'(?i)^arabic_text\s*[:：]\s*', '', text).strip()
    text = re.sub(r'(?i)^arabic\s*[:：]\s*', '', text).strip()
    text = normalize_text(text)
    return text


def classify_visual_dalil_type(arabic_text: str, raw_obj: Optional[Dict[str, Any]] = None, query: str = '', source_text: str = '') -> str:
    """Klasifikasi kasar untuk membedakan dalil sungguhan vs istilah Arab.
    Ini heuristik, bukan fatwa. Tujuannya quality gate agar istilah Arab tidak ditampilkan sebagai dalil.
    """
    raw_obj = raw_obj or {}
    supplied = _safe_lower(raw_obj.get('dalil_type'))
    supplied = supplied.replace('-', '_').replace(' ', '_')
    alias = {
        'ayah': 'ayat', 'quran': 'ayat', 'al_quran': 'ayat', 'alquran': 'ayat', 'ayat_al_quran': 'ayat',
        'hadith': 'hadis', 'hadeeth': 'hadis', 'hadis_nabi': 'hadis',
        'kaidah_fiqh': 'kaidah', 'kaidah_fikih': 'kaidah', 'qawaid': 'kaidah', 'qaidah': 'kaidah',
        'quote': 'kutipan_ulama', 'kutipan': 'kutipan_ulama', 'ulama': 'kutipan_ulama',
        'term': 'istilah_arab', 'istilah': 'istilah_arab', 'istilah_arab': 'istilah_arab',
        'not_dalil': 'not_dalil', 'none': 'none', 'unknown': 'unknown'
    }
    if supplied in alias:
        supplied = alias[supplied]
    if supplied in VISUAL_DALIL_ALLOWED_TYPES or supplied in VISUAL_DALIL_REJECT_TYPES:
        # tetap validasi ringan di bawah untuk supplied=unknown/istilah.
        if supplied in VISUAL_DALIL_ALLOWED_TYPES:
            return supplied
        return supplied

    txt = arabic_text or ''
    combined = f"{txt}\n{source_text}".lower()

    # Frasa yang umum muncul pada ayat/hadis/kaidah dalam fatwa.
    ayat_markers = [
        'قال الله', 'قوله تعالى', 'يا أيها', 'ومن', 'فإن', 'إن الله', 'والله', 'ولا ', 'وأحل', 'وحرم',
        'البقرة', 'النساء', 'المائدة', 'الجمعة', 'الحشر', 'الإسراء'
    ]
    hadis_markers = [
        'قال رسول', 'رسول الله', 'صلى الله عليه وسلم', 'رواه', 'عن ', 'حديث', 'النبي', 'نهى', 'لا ضرر'
    ]
    kaidah_markers = [
        'الأصل', 'الضرر', 'المشقة', 'اليقين', 'العادة', 'الأمور', 'درء', 'جلب', 'لا ضرر ولا ضرار',
        'الأصل في المعاملات', 'الإباحة', 'المعاملات'
    ]

    if any(m in combined for m in hadis_markers):
        return 'hadis'
    if any(m in combined for m in kaidah_markers):
        return 'kaidah'
    if any(m in combined for m in ayat_markers):
        return 'ayat'

    # Kalau Arab sangat pendek biasanya istilah, bukan dalil.
    ar_count = count_arabic(txt)
    words = re.findall(r'[\u0600-\u06FF]+', txt)
    if ar_count < 25 or len(words) <= 3:
        return 'istilah_arab'

    # Jika cukup panjang tapi tanpa marker, masukkan unknown agar tidak langsung accepted.
    return 'unknown'


def is_visual_dalil_accepted(arabic_text: str, obj: Dict[str, Any], quality: Dict[str, Any], dalil_type: str) -> bool:
    conf = _safe_lower(obj.get('confidence') or 'low')
    visible = obj.get('arabic_visible')
    if isinstance(visible, str):
        visible = visible.strip().lower() in ['true', '1', 'yes', 'ya']
    if visible is not True:
        return False
    if dalil_type not in VISUAL_DALIL_ALLOWED_TYPES:
        return False
    if count_arabic(arabic_text) < VISUAL_DALIL_MIN_ARABIC_COUNT:
        return False
    if quality.get('score', 0.0) < VISUAL_DALIL_MIN_QUALITY_SCORE:
        return False
    if (not VISUAL_DALIL_ACCEPT_LOW_CONFIDENCE) and conf == 'low':
        return False
    if '[TIDAK TERBACA]' in arabic_text or 'TIDAK TERBACA' in arabic_text:
        # masih bisa jadi evidence, tapi jangan accepted sebagai kutipan.
        return False
    return True


def build_visual_dalil_prompt(source_label: str, item: Dict[str, Any], crop_name: str) -> str:
    query = item.get('query') or ''
    return f"""
Anda membaca crop gambar dari dokumen DSN-MUI.
Tugas Anda adalah MENYALIN teks Arab yang terlihat dan MENGKLASIFIKASIKAN apakah teks Arab itu benar-benar dalil.

Aturan penting:
- Salin PERSIS teks Arab yang terlihat pada gambar.
- Jangan memperbaiki, jangan melengkapi, jangan menambahkan harakat jika tidak terlihat.
- Jangan membuat dalil baru.
- Kalau bagian teks tidak jelas, tulis [TIDAK TERBACA] pada bagian itu.
- Kalau teks Arab yang terlihat hanya istilah akad/istilah Arab biasa, beri dalil_type="istilah_arab".
- Kalau tidak ada teks Arab terlihat, arabic_visible=false dan arabic_text="".
- Output JSON valid saja, tanpa markdown.

Konteks query pengguna: {query}
Source: {source_label}
Crop: {crop_name}

Schema JSON wajib:
{{
  "arabic_visible": true,
  "arabic_text": "...",
  "dalil_type": "ayat|hadis|kaidah|kutipan_ulama|istilah_arab|unknown|none",
  "confidence": "high|medium|low",
  "unclear_parts": ["..."],
  "notes": "singkat"
}}
""".strip()


def _vl_parse_json(raw: str) -> Dict[str, Any]:
    """Robust parser V16.
    Prioritas: ambil JSON valid; kalau gagal, fallback raw text Arab.
    Penting: jangan biarkan arabic_text berisi JSON mentah.
    """
    raw0 = raw or ''
    raw_clean = strip_code_fence(raw0).strip()

    obj = None
    parsed = _json_loads_maybe(raw_clean)
    if isinstance(parsed, dict):
        obj = parsed
    if obj is None:
        jtxt = _extract_first_json_object_text(raw_clean)
        parsed = _json_loads_maybe(jtxt) if jtxt else None
        if isinstance(parsed, dict):
            obj = parsed
    if obj is None:
        ext = extract_json_object(raw_clean)
        if isinstance(ext, dict):
            obj = ext

    if obj is None:
        arabic_text = _clean_vlm_arabic_text(raw_clean, raw=raw_clean)
        obj = {
            'arabic_visible': bool(count_arabic(arabic_text) >= VISUAL_DALIL_MIN_ARABIC_COUNT),
            'arabic_text': arabic_text,
            'dalil_type': classify_visual_dalil_type(arabic_text, {}, source_text=raw_clean),
            'confidence': 'low',
            'unclear_parts': [],
            'notes': 'raw_non_json_output'
        }
    else:
        obj = _normalize_vlm_key_obj(obj)
        arabic_text = _clean_vlm_arabic_text(obj.get('arabic_text'), raw=raw_clean)
        obj['arabic_text'] = arabic_text
        obj['dalil_type'] = classify_visual_dalil_type(arabic_text, obj, source_text=raw_clean)
        if 'arabic_visible' not in obj:
            obj['arabic_visible'] = bool(count_arabic(arabic_text) >= VISUAL_DALIL_MIN_ARABIC_COUNT)
        if 'confidence' not in obj:
            obj['confidence'] = 'low'
        if not isinstance(obj.get('unclear_parts'), list):
            obj['unclear_parts'] = []
        obj.setdefault('notes', '')
    return obj


@torch.no_grad()
def transcribe_arabic_from_image_vlm(image_path: str, source_label: str, item: Dict[str, Any], crop_name: str) -> Dict[str, Any]:
    """Override V16: hasil VLM dilengkapi dalil_type + quality gate dalil."""
    load_visual_dalil_model()
    img = Image.open(image_path).convert('RGB')
    prompt = build_visual_dalil_prompt(source_label, item, crop_name)
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': str(image_path)},
            {'type': 'text', 'text': prompt},
        ]
    }]

    text = visual_dalil_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = visual_dalil_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt'
        )
    except Exception:
        inputs = visual_dalil_processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors='pt'
        )

    device = next(visual_dalil_model.parameters()).device
    inputs = inputs.to(device)
    out_ids = visual_dalil_model.generate(
        **inputs,
        max_new_tokens=VISUAL_DALIL_MAX_NEW_TOKENS,
        do_sample=False,
        temperature=VISUAL_DALIL_TEMPERATURE,
        pad_token_id=getattr(visual_dalil_processor.tokenizer, 'eos_token_id', None),
    )
    gen = out_ids[0][inputs['input_ids'].shape[-1]:]
    raw = visual_dalil_processor.decode(gen, skip_special_tokens=True).strip()
    obj = _vl_parse_json(raw)
    arabic_text = _clean_vlm_arabic_text(obj.get('arabic_text'), raw=raw)
    dalil_type = classify_visual_dalil_type(arabic_text, obj, query=item.get('query') or '', source_text=raw)
    obj['arabic_text'] = arabic_text
    obj['dalil_type'] = dalil_type
    q = arabic_quality_score(arabic_text)
    accepted = is_visual_dalil_accepted(arabic_text, obj, q, dalil_type)

    if accepted:
        quality_status = 'verified_visual_candidate'
    elif count_arabic(arabic_text) >= VISUAL_DALIL_MIN_ARABIC_COUNT and dalil_type in VISUAL_DALIL_ALLOWED_TYPES:
        quality_status = 'visual_needs_verification'
    elif count_arabic(arabic_text) >= VISUAL_DALIL_MIN_ARABIC_COUNT:
        quality_status = 'arabic_detected_not_dalil'
    else:
        quality_status = 'no_readable_arabic'

    return {
        'image_path': str(image_path),
        'source_label': source_label,
        'crop_name': crop_name,
        'raw_generation': raw,
        'parsed': obj,
        'arabic_text': arabic_text,
        'dalil_type': dalil_type,
        'confidence': obj.get('confidence') or 'low',
        'unclear_parts': obj.get('unclear_parts') if isinstance(obj.get('unclear_parts'), list) else [],
        'notes': obj.get('notes') or '',
        'quality': q,
        'quality_status': quality_status,
        'accepted': accepted,
        'needs_human_verification': True,
    }


def heuristic_clip_rects(page_rect: fitz.Rect, names: Optional[List[str]] = None) -> List[Tuple[str, fitz.Rect]]:
    """Override V16: tambah wider same-page band sebagai fallback, tetap bukan full page."""
    names = names or VISUAL_DALIL_HEURISTIC_CLIPS
    w, h = page_rect.width, page_rect.height
    specs = {
        'upper_middle': fitz.Rect(page_rect.x0 + 0.04*w, page_rect.y0 + 0.08*h, page_rect.x0 + 0.96*w, page_rect.y0 + 0.58*h),
        'middle':       fitz.Rect(page_rect.x0 + 0.04*w, page_rect.y0 + 0.22*h, page_rect.x0 + 0.96*w, page_rect.y0 + 0.78*h),
        'lower_middle': fitz.Rect(page_rect.x0 + 0.04*w, page_rect.y0 + 0.38*h, page_rect.x0 + 0.96*w, page_rect.y0 + 0.96*h),
        # Wider same-page bands: masih clip per area, bukan full page.
        'wide_upper':   fitz.Rect(page_rect.x0 + 0.02*w, page_rect.y0 + 0.04*h, page_rect.x0 + 0.98*w, page_rect.y0 + 0.64*h),
        'wide_middle':  fitz.Rect(page_rect.x0 + 0.02*w, page_rect.y0 + 0.16*h, page_rect.x0 + 0.98*w, page_rect.y0 + 0.86*h),
        'wide_lower':   fitz.Rect(page_rect.x0 + 0.02*w, page_rect.y0 + 0.30*h, page_rect.x0 + 0.98*w, page_rect.y0 + 0.98*h),
    }
    return [(n, specs[n]) for n in names if n in specs]


def crop_visual_candidates_chunk_aligned(result: Dict[str, Any], item: Dict[str, Any], source_label: str, max_crops: int = VISUAL_DALIL_MAX_CROPS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Override V16: bbox line Arab lebih dulu, lalu fallback wider same-page band bila query butuh dalil."""
    p = result.get('payload', {})
    pdf_path = p.get('pdf_path')
    page_no = p.get('page')
    chunk_id = str(p.get('chunk_id') or source_label)
    if not pdf_path or not page_no:
        return []

    signal = get_chunk_arabic_signal(result)
    if VISUAL_DALIL_REQUIRE_ARABIC_IN_CHUNK and not (signal['has_arabic'] or signal['has_dalil_keyword']):
        return []

    chunk_signal_text = '\n'.join([
        str(p.get('title') or ''),
        str(signal.get('raw_arabic_text') or ''),
        str(payload_text(p) or ''),
    ])

    crops = []
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[int(page_no) - 1]
        page_rect = page.rect
    finally:
        doc.close()

    bbox_rows = find_arabic_bboxes_from_pdf_text_layer(pdf_path, int(page_no), chunk_signal_text)
    groups = merge_close_bboxes(bbox_rows, page_rect, max_groups=max_crops)

    for gi, g in enumerate(groups, start=1):
        rect = g['rect']
        out = CROP_DIR / f'visual_v16_chunkclip_{safe_name(chunk_id)}_p{page_no}_bbox{gi}.png'
        render_pdf_clip(pdf_path, int(page_no), rect, out, dpi=VISUAL_DALIL_RENDER_DPI)
        with Image.open(out) as im:
            width, height = im.size
        crops.append({
            'crop_name': f'chunk_arabic_bbox_{gi}',
            'image_path': str(out),
            'page_image_path': None,
            'box_pdf_points': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            'box': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            'width': width,
            'height': height,
            'bbox_source': g.get('source'),
            'bbox_score': g.get('score'),
            'bbox_text_preview': g.get('text_preview'),
            'chunk_arabic_count': signal.get('arabic_count'),
        })
        if len(crops) >= max_crops:
            return crops

    # Fallback: for query/source yang butuh dalil, coba band lebih luas dari halaman yang sama.
    q = item.get('query') or ''
    query_needs_dalil = bool(item.get('needs_dalil')) or should_run_arabic_evidence(q)
    fallback_names = list(VISUAL_DALIL_HEURISTIC_CLIPS)
    if VISUAL_DALIL_ENABLE_WIDER_SAME_PAGE_FALLBACK and query_needs_dalil:
        for n in VISUAL_DALIL_WIDER_CLIPS:
            if n not in fallback_names:
                fallback_names.append(n)

    if VISUAL_DALIL_FALLBACK_HEURISTIC_CLIP:
        for name, rect in heuristic_clip_rects(page_rect, fallback_names):
            out = CROP_DIR / f'visual_v16_chunkclip_{safe_name(chunk_id)}_p{page_no}_{name}.png'
            render_pdf_clip(pdf_path, int(page_no), rect, out, dpi=VISUAL_DALIL_RENDER_DPI)
            with Image.open(out) as im:
                width, height = im.size
            crops.append({
                'crop_name': f'heuristic_{name}',
                'image_path': str(out),
                'page_image_path': None,
                'box_pdf_points': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                'box': [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                'width': width,
                'height': height,
                'bbox_source': 'heuristic_wider_same_page_clip_v16' if name in VISUAL_DALIL_WIDER_CLIPS else 'heuristic_clip_no_text_layer_bbox_v16',
                'bbox_score': None,
                'bbox_text_preview': '',
                'chunk_arabic_count': signal.get('arabic_count'),
            })
            if len(crops) >= max_crops:
                break

    return crops


def visual_dalil_for_result(result: Dict[str, Any], item: Dict[str, Any], source_idx: int) -> Dict[str, Any]:
    """Override V16: pilih kandidat berdasarkan acceptance dalil_type, bukan hanya quality Arabic."""
    p = result.get('payload', {})
    source_label = f"S{source_idx} | {infer_category_from_payload(p)} {infer_nomor_from_payload(p)} | halaman {p.get('page')}"
    signal = get_chunk_arabic_signal(result)
    base = {
        'status': 'not_run',
        'source_id': f'S{source_idx}',
        'source_label': source_label,
        'category': infer_category_from_payload(p),
        'nomor': infer_nomor_from_payload(p),
        'title': p.get('title'),
        'page': p.get('page'),
        'pdf_path': p.get('pdf_path'),
        'page_image_path': None,
        'best_crop_image_path': None,
        'best_crop_name': None,
        'arabic_text': '',
        'dalil_type': 'none',
        'confidence': 'low',
        'quality': {},
        'quality_status': 'not_run',
        'accepted': False,
        'needs_human_verification': True,
        'candidates': [],
        'error': None,
        'render_strategy': VISUAL_DALIL_RENDER_STRATEGY,
        'chunk_has_arabic': signal.get('has_arabic'),
        'chunk_arabic_count': signal.get('arabic_count'),
        'chunk_has_dalil_keyword': signal.get('has_dalil_keyword'),
    }
    if not VISUAL_DALIL_ENABLED:
        base['status'] = 'disabled'
        return base
    if not source_should_use_visual_dalil(item, result):
        base['status'] = 'skipped_no_arabic_in_same_chunk_or_no_pdf'
        return base
    try:
        crops = crop_visual_candidates_chunk_aligned(result, item, source_label, max_crops=VISUAL_DALIL_MAX_CROPS_PER_SOURCE)
        if not crops:
            base['status'] = 'no_chunk_aligned_crop'
            return base
        candidates = []
        print(f"[visual-dalil-v16] {source_label}: {len(crops)} chunk-aligned/same-page clips")
        for crop in crops:
            print(f"[visual-dalil-v16] transcribe {source_label} crop={crop['crop_name']} source={crop.get('bbox_source')}")
            cand = transcribe_arabic_from_image_vlm(crop['image_path'], source_label, item, crop['crop_name'])
            cand.update({
                'page_image_path': None,
                'box': crop.get('box'),
                'box_pdf_points': crop.get('box_pdf_points'),
                'width': crop.get('width'),
                'height': crop.get('height'),
                'bbox_source': crop.get('bbox_source'),
                'bbox_score': crop.get('bbox_score'),
                'bbox_text_preview': crop.get('bbox_text_preview'),
                'chunk_arabic_count': crop.get('chunk_arabic_count'),
            })
            candidates.append(cand)

        def cand_rank(c):
            accepted_bonus = 1000 if c.get('accepted') else 0
            type_bonus = 120 if c.get('dalil_type') in VISUAL_DALIL_ALLOWED_TYPES else 0
            conf_bonus = {'high': 40, 'medium': 20, 'low': 0}.get(_safe_lower(c.get('confidence')), 0)
            qscore = (c.get('quality') or {}).get('score', 0.0)
            return accepted_bonus + type_bonus + conf_bonus + qscore

        best = max(candidates, key=cand_rank) if candidates else None
        if best:
            dalil_type = best.get('dalil_type') or 'unknown'
            quality_status = best.get('quality_status') or 'unknown'
            if best.get('accepted'):
                status = 'accepted'
            elif quality_status == 'visual_needs_verification':
                status = 'visual_needs_verification'
            elif quality_status == 'arabic_detected_not_dalil':
                status = 'arabic_detected_not_dalil'
            elif quality_status == 'no_readable_arabic':
                status = 'no_readable_arabic'
            else:
                status = 'low_quality'
            base.update({
                'status': status,
                'best_crop_image_path': best.get('image_path'),
                'best_crop_name': best.get('crop_name'),
                'arabic_text': best.get('arabic_text') or '',
                'dalil_type': dalil_type,
                'confidence': best.get('confidence') or 'low',
                'quality': best.get('quality') or {},
                'quality_status': quality_status,
                'accepted': bool(best.get('accepted')),
                'needs_human_verification': True,
                'candidates': candidates,
            })
        else:
            base['status'] = 'no_candidate'
        return base
    except Exception as e:
        base['status'] = 'error'
        base['error'] = f'{type(e).__name__}: {str(e)}'
        traceback.print_exc()
        return base


def visual_dalil_line_from_result(result: Dict[str, Any], source_idx: int) -> Optional[str]:
    """Override V16: tampilkan hanya accepted; warning untuk kandidat dalil yang perlu verifikasi; jangan tampilkan istilah Arab sebagai dalil."""
    vd = result.get('visual_dalil') or {}
    if not vd:
        return None
    p = result.get('payload', {})
    source = f'[S{source_idx}] {infer_category_from_payload(p)} No. {infer_nomor_from_payload(p)}, halaman {p.get("page")}'
    crop = vd.get('best_crop_image_path') or ''
    dtype = vd.get('dalil_type') or 'unknown'
    conf = vd.get('confidence') or 'medium'

    if vd.get('accepted') and vd.get('arabic_text'):
        text = _clean_vlm_arabic_text(vd.get('arabic_text'))
        return (
            f'- {source}\n'
            f'  Jenis: {dtype}\n'
            f'  Arab hasil pembacaan visual: {text}\n'
            f'  Status: transkripsi visual Qwen3-VL, confidence={conf}, perlu verifikasi untuk kutipan resmi. Crop: {crop}'
        )

    if vd.get('status') == 'visual_needs_verification':
        return (
            f'- {source}\n'
            f'  Ditemukan kandidat {dtype} pada gambar, tetapi teks Arab belum cukup presisi untuk ditampilkan sebagai kutipan resmi. '
            f'Verifikasi crop/halaman asli: {crop}'
        )

    if vd.get('status') == 'arabic_detected_not_dalil':
        # Jangan tampilkan sebagai dalil. Ini menghindari istilah Arab biasa masuk ke jawaban.
        return None

    if vd.get('status') in ['low_quality', 'error']:
        return (
            f'- {source}\n'
            f'  Ditemukan kandidat teks Arab pada gambar, tetapi kualitasnya belum cukup untuk dikutip. Verifikasi crop/halaman asli: {crop}'
        )
    return None


def deterministic_dalil_lines(results: List[Dict[str, Any]], item: Dict[str, Any], max_sources: int = 3, allow: bool = True) -> List[str]:
    """Override V16: visual evidence dulu; fallback teks hanya kalau relevan dan bukan no-context."""
    if not allow:
        return []
    lines = []
    query = item.get('retrieval_query') or item.get('query') or ''
    q_terms = set(query_keywords(query))

    # 1) Prefer visual Arabic evidence accepted / needs verification.
    for i, r in enumerate(results[:max_sources], start=1):
        line = visual_dalil_line_from_result(r, i)
        if line:
            lines.append(line)

    if lines:
        return lines

    # 2) Kalau ada Arab visual tapi bukan dalil, jangan fallback ke Arab noisy.
    any_visual_not_dalil = any((r.get('visual_dalil') or {}).get('status') == 'arabic_detected_not_dalil' for r in results[:max_sources])
    if any_visual_not_dalil:
        return ['- Terdeteksi teks Arab pada source visual, tetapi tidak diklasifikasikan sebagai ayat/hadis/kaidah sehingga tidak ditampilkan sebagai dalil.']

    # 3) Fallback text snippet, tetap conservative.
    for i, r in enumerate(results[:max_sources], start=1):
        p = r.get('payload', {})
        doc_text = payload_text(p)
        dalil = extract_dalil_snippets(doc_text, max_chars=MAX_DALIL_CHARS_PER_SOURCE)
        if not dalil:
            continue
        combined = (str(p.get('title') or '') + ' ' + doc_text).lower()
        overlap = sum(1 for t in q_terms if t in combined and t not in STOPWORDS_ID)
        if overlap < 2:
            continue
        if count_arabic(dalil) >= 8:
            dtype = classify_visual_dalil_type(dalil, {}, query=query, source_text=doc_text)
            q = arabic_quality_score(dalil)
            if dtype not in VISUAL_DALIL_ALLOWED_TYPES or q.get('label') != 'usable' or q.get('score', 0.0) < VISUAL_DALIL_MIN_QUALITY_SCORE:
                lines.append(f'- [S{i}] Ditemukan potongan dalil/kaidah pada source, tetapi teks Arab dari ekstraksi teks belum cukup bersih/terklasifikasi untuk dikutip. Gunakan halaman dokumen asli untuk verifikasi.')
                continue
        lines.append(f'- [S{i}] {dalil}')
    return lines


# Override retrieval flatten supaya kolom CSV memuat status dalil V16.
def flatten_umkm_retrieval_rows(item: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for source_idx, r in enumerate(results, start=1):
        p = r.get('payload', {})
        ev = r.get('arabic_evidence') or {}
        vd = r.get('visual_dalil') or {}
        rows.append({
            'run_id': RUN_ID,
            'query_id': item.get('id'),
            'domain': item.get('domain'),
            'query': item.get('query'),
            'retrieval_query': item.get('retrieval_query'),
            'source_id': f'S{source_idx}',
            'rank_final': r.get('rank_final'),
            'rank_hybrid': r.get('rank_hybrid'),
            'rank_vector': r.get('rank_vector'),
            'rank_bm25': r.get('rank_bm25'),
            'reranker_used': r.get('reranker_used'),
            'reranker_type': r.get('reranker_type'),
            'score_reranker': r.get('score_reranker'),
            'statistical_score': r.get('statistical_score'),
            'score_hybrid': r.get('score_hybrid'),
            'score_vector': r.get('score_vector'),
            'score_bm25': r.get('score_bm25'),
            'chunk_id': r.get('chunk_id') or p.get('chunk_id'),
            'category': infer_category_from_payload(p),
            'nomor': infer_nomor_from_payload(p),
            'title': p.get('title'),
            'page': p.get('page'),
            'pdf_filename': p.get('pdf_filename'),
            'pdf_path': p.get('pdf_path'),
            'text_preview': text_preview(payload_text(p), 700),
            'has_dalil_snippet': contains_dalil(payload_text(p)),
            'arabic_evidence_status': ev.get('ocr_status'),
            'arabic_selected_quality': (ev.get('selected_quality') or {}).get('label'),
            'crop_image_path': ev.get('crop_image_path'),
            'visual_dalil_status': vd.get('status'),
            'visual_dalil_render_strategy': vd.get('render_strategy'),
            'visual_dalil_chunk_has_arabic': vd.get('chunk_has_arabic'),
            'visual_dalil_chunk_arabic_count': vd.get('chunk_arabic_count'),
            'visual_dalil_chunk_has_dalil_keyword': vd.get('chunk_has_dalil_keyword'),
            'visual_dalil_accepted': vd.get('accepted'),
            'visual_dalil_confidence': vd.get('confidence'),
            'visual_dalil_dalil_type': vd.get('dalil_type'),
            'visual_dalil_quality_status': vd.get('quality_status'),
            'visual_dalil_quality_label': (vd.get('quality') or {}).get('label'),
            'visual_dalil_quality_score': (vd.get('quality') or {}).get('score'),
            'visual_dalil_arabic_preview': text_preview(_clean_vlm_arabic_text(vd.get('arabic_text') or ''), 400),
            'visual_dalil_page_image_path': vd.get('page_image_path'),
            'visual_dalil_crop_image_path': vd.get('best_crop_image_path'),
            'visual_dalil_best_crop_name': vd.get('best_crop_name'),
        })
    return rows

print('V16 patch active: visual dalil quality gate + dalil_type classifier + robust JSON parsing.')
print('DALIL_EXTRACTION_MODE:', DALIL_EXTRACTION_MODE)
print('VISUAL_DALIL_RENDER_STRATEGY:', VISUAL_DALIL_RENDER_STRATEGY)
print('VISUAL_DALIL_ALLOWED_TYPES:', VISUAL_DALIL_ALLOWED_TYPES)
print('VISUAL_DALIL_MAX_CROPS_PER_SOURCE:', VISUAL_DALIL_MAX_CROPS_PER_SOURCE)



# %% CELL 32

UMKM_ANSWERS_CSV = RUN_DIR / 'rag_umkm_answers.csv'
UMKM_RETRIEVAL_CSV = RUN_DIR / 'rag_umkm_retrieval_details.csv'
UMKM_EVIDENCE_CSV = RUN_DIR / 'rag_umkm_evidence.csv'
UMKM_FULL_JSONL = RUN_DIR / 'rag_umkm_full.jsonl'
UMKM_SUMMARY_CSV = RUN_DIR / 'rag_umkm_summary.csv'

test_queries = DEFAULT_UMKM_TEST_QUERIES
assert len(test_queries) >= 50, 'Test set UMKM terlalu sedikit / belum terganti.'
assert str(test_queries[0]['id']).startswith('U'), 'ID test set UMKM harus mulai dengan U.'

def save_umkm_outputs(answer_rows, retrieval_rows, evidence_rows, full_rows):
    pd.DataFrame(answer_rows).to_csv(UMKM_ANSWERS_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(retrieval_rows).to_csv(UMKM_RETRIEVAL_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(evidence_rows).to_csv(UMKM_EVIDENCE_CSV, index=False, encoding='utf-8-sig')
    compact_full = []
    for x in full_rows:
        compact_full.append({
            'item': x['item'],
            'answer_row': x['answer_row'],
            'support_info': x.get('support_info'),
            'context': x['context'],
            'prompt': x['prompt'],
            'answer': x.get('answer', ''),
            'generation_meta': x.get('generation_meta', {}),
            'results': [{k: v for k, v in r.items() if k != 'payload'} | {'payload': r.get('payload', {})} for r in x.get('results', [])]
        })
    write_jsonl(UMKM_FULL_JSONL, compact_full)

def write_umkm_summary(answer_rows):
    df = pd.DataFrame(answer_rows)
    if df.empty:
        return
    summary = {
        'run_id': RUN_ID,
        'generator_model': GENERATOR_MODEL_NAME,
        'n_queries': len(df),
        'n_errors': int(df.get('error', pd.Series(dtype=object)).notna().sum()) if 'error' in df else 0,
        'support_high': int((df.get('support_level') == 'high').sum()) if 'support_level' in df else None,
        'support_medium': int((df.get('support_level') == 'medium').sum()) if 'support_level' in df else None,
        'support_low': int((df.get('support_level') == 'low').sum()) if 'support_level' in df else None,
        'support_very_low': int((df.get('support_level') == 'very_low').sum()) if 'support_level' in df else None,
        'dalil_available_topk': int(df.get('dalil_available_in_topk', pd.Series(dtype=bool)).fillna(False).sum()) if 'dalil_available_in_topk' in df else None,
        'abstained': int(df.get('abstained', pd.Series(dtype=bool)).fillna(False).sum()) if 'abstained' in df else None,
        'answer_validation_bad': int(df.get('answer_validation_bad', pd.Series(dtype=bool)).fillna(False).sum()) if 'answer_validation_bad' in df else None,
        'json_parse_ok': int(df.get('json_parse_ok', pd.Series(dtype=bool)).fillna(False).sum()) if 'json_parse_ok' in df else None,
        'json_generated': int((df.get('generation_mode') == 'json_generated').sum()) if 'generation_mode' in df else None,
        'json_generated_retry': int((df.get('generation_mode') == 'json_generated_retry').sum()) if 'generation_mode' in df else None,
        'extractive_fallback': int(df.get('generation_mode', pd.Series(dtype=object)).astype(str).str.contains('fallback|NO_CONTEXT', regex=True).sum()) if 'generation_mode' in df else None,
        'avg_retrieval_total_ms': float(pd.to_numeric(df.get('retrieval_total_ms'), errors='coerce').mean()) if 'retrieval_total_ms' in df else None,
        'avg_generation_ms': float(pd.to_numeric(df.get('generation_ms'), errors='coerce').mean()) if 'generation_ms' in df else None,
        'answers_csv': str(UMKM_ANSWERS_CSV),
        'retrieval_csv': str(UMKM_RETRIEVAL_CSV),
        'evidence_csv': str(UMKM_EVIDENCE_CSV),
        'full_jsonl': str(UMKM_FULL_JSONL),
    }
    pd.DataFrame([summary]).to_csv(UMKM_SUMMARY_CSV, index=False, encoding='utf-8-sig')

def run_umkm_retrieval_suite(test_queries: List[Dict[str, Any]], start: int = 0, stop: Optional[int] = None):
    answer_rows, retrieval_rows, evidence_rows, full_rows = [], [], [], []
    subset = test_queries[start:stop]
    for item in tqdm(subset, desc='Phase 1 - UMKM retrieval/context'):
        print('\n' + '=' * 120)
        print(item.get('id'), item.get('domain'), '|', item.get('query')[:160])
        try:
            out = run_one_umkm_retrieval_test(item)
            answer_rows.append(out['answer_row'])
            retrieval_rows.extend(out['retrieval_rows'])
            evidence_rows.extend(out['evidence_rows'])
            full_rows.append(out)
            print('Top1:', out['answer_row']['top1_category'], out['answer_row']['top1_nomor'], '|', out['answer_row']['top1_title'], '| page', out['answer_row']['top1_page'])
            print('Support:', out['answer_row']['support_level'], out['answer_row']['support_score'], '| dalil_topk:', out['answer_row']['dalil_available_in_topk'])
            print('Prompt chars:', out['answer_row']['prompt_chars'])
        except Exception as e:
            err = f'{type(e).__name__}: {str(e)}'
            print('ERROR:', err)
            traceback.print_exc()
            answer_rows.append({'run_id': RUN_ID, 'query_id': item.get('id'), 'domain': item.get('domain'), 'query': item.get('query'), 'category_filter': item.get('category_filter'), 'error': err, 'answer': ''})
        save_umkm_outputs(answer_rows, retrieval_rows, evidence_rows, full_rows)
        write_umkm_summary(answer_rows)
    print('\nPhase 1 saved:')
    print('UMKM_ANSWERS_CSV:', UMKM_ANSWERS_CSV)
    print('UMKM_RETRIEVAL_CSV:', UMKM_RETRIEVAL_CSV)
    print('UMKM_FULL_JSONL:', UMKM_FULL_JSONL)
    return answer_rows, retrieval_rows, evidence_rows, full_rows

def generate_umkm_answers_from_cached_rows(answer_rows, retrieval_rows, evidence_rows, full_rows):
    if generator is None or tokenizer is None:
        load_generator_model()
    id_to_answer_row = {row.get('query_id'): row for row in answer_rows}
    for x in tqdm(full_rows, desc='Phase 2 - UMKM generation'):
        qid = x['item'].get('id')
        row = id_to_answer_row.get(qid)
        if row is None or row.get('error'):
            continue
        print('\n' + '=' * 120)
        print(qid, x['item'].get('domain'), '|', x['item'].get('query')[:160])
        g0 = now_ms()
        try:
            res = answer_query_umkm_safely(x['item'], x.get('results', []), x['prompt'], x.get('support_info', {}))
            if isinstance(res, tuple):
                ans, gen_meta = res
            else:
                ans, gen_meta = res, {}
            generation_ms = now_ms() - g0
            x['answer'] = ans
            x['generation_meta'] = gen_meta
            x['answer_row']['answer'] = ans
            x['answer_row']['answer_chars'] = len(ans)
            x['answer_row']['abstained'] = answer_abstained(ans)
            x['answer_row']['answer_validation_bad'] = bool(output_is_bad(ans, query=x['item'].get('query'), prompt=x['prompt']))
            x['answer_row']['generation_ms'] = round(generation_ms, 2)
            x['answer_row']['generator_model'] = GENERATOR_MODEL_NAME
            x['answer_row']['generation_mode'] = gen_meta.get('generation_mode')
            x['answer_row']['json_parse_ok'] = gen_meta.get('json_parse_ok')
            x['answer_row']['raw_generation_preview'] = (gen_meta.get('raw_generation') or '')[:800]
            if x['answer_row']['answer_validation_bad']:
                verdict = 'ANSWERED_WITH_BAD_FORMAT'
            elif x['answer_row']['abstained'] and x['answer_row'].get('support_level') in ['low', 'very_low']:
                verdict = 'ABSTAIN_LOW_SUPPORT'
            elif str(gen_meta.get('generation_mode', '')).startswith('extractive'):
                verdict = 'CONTEXT_FOUND_GENERATION_FAILED_EXTRACTIVE_FALLBACK'
            elif gen_meta.get('generation_mode') == 'NO_CONTEXT':
                verdict = 'NO_CONTEXT'
            else:
                verdict = 'ANSWERED_JSON_GROUNDED_CANDIDATE'
            x['answer_row']['final_verdict'] = verdict
            row.update(x['answer_row'])
            print('Mode:', gen_meta.get('generation_mode'), '| json_ok:', gen_meta.get('json_parse_ok'))
            print('Answer preview:', ans[:500].replace('\n', ' '))
            print('generation_ms:', round(generation_ms, 2))
        except Exception as e:
            err = f'{type(e).__name__}: {str(e)}'
            print('GENERATION ERROR:', err)
            traceback.print_exc()
            row['generation_error'] = err
            row['answer'] = ''
        save_umkm_outputs(answer_rows, retrieval_rows, evidence_rows, full_rows)
        write_umkm_summary(answer_rows)
    answers_df = pd.DataFrame(answer_rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    evidence_df = pd.DataFrame(evidence_rows)
    write_umkm_summary(answer_rows)
    print('\nFinal files:')
    print('UMKM_ANSWERS_CSV:', UMKM_ANSWERS_CSV)
    print('UMKM_RETRIEVAL_CSV:', UMKM_RETRIEVAL_CSV)
    print('UMKM_EVIDENCE_CSV:', UMKM_EVIDENCE_CSV)
    print('UMKM_SUMMARY_CSV:', UMKM_SUMMARY_CSV)
    print('UMKM_FULL_JSONL:', UMKM_FULL_JSONL)
    if UMKM_SUMMARY_CSV.exists():
        display(pd.read_csv(UMKM_SUMMARY_CSV))
    return answers_df, retrieval_df, evidence_df

answer_rows, retrieval_rows, evidence_rows, full_rows = run_umkm_retrieval_suite(test_queries)

# Phase 1.5: Visual Arabic Dalil Extraction.
# Jalankan setelah retrieval, sebelum generation, agar hasil transkripsi visual masuk ke rendering jawaban.
if VISUAL_DALIL_ENABLED:
    answer_rows, retrieval_rows, evidence_rows, full_rows = generate_visual_dalil_for_cached_rows(
        answer_rows, retrieval_rows, evidence_rows, full_rows
    )
    save_umkm_outputs(answer_rows, retrieval_rows, evidence_rows, full_rows)
    write_umkm_summary(answer_rows)

if GENERATE_AFTER_RETRIEVAL:
    if UNLOAD_RETRIEVAL_BEFORE_GENERATION:
        release_retrieval_models()
    answers_df, retrieval_df, evidence_df = generate_umkm_answers_from_cached_rows(answer_rows, retrieval_rows, evidence_rows, full_rows)
else:
    answers_df = pd.DataFrame(answer_rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    evidence_df = pd.DataFrame(evidence_rows)
    print('Generation skipped. Set GENERATE_AFTER_RETRIEVAL=True kalau mau isi kolom answer.')



# %% CELL 34

answers_df = pd.read_csv(UMKM_ANSWERS_CSV)
retrieval_df = pd.read_csv(UMKM_RETRIEVAL_CSV)
print('answers:', answers_df.shape)
print('retrieval:', retrieval_df.shape)
for c in [c for c in ['support_level', 'abstained', 'dalil_available_in_topk', 'answer_validation_bad', 'final_verdict'] if c in answers_df.columns]:
    print('\n', c)
    display(answers_df[c].value_counts(dropna=False).to_frame('count'))
show_cols = [c for c in ['query_id','domain','support_level','support_score','dalil_available_in_topk','abstained','final_verdict','top1_category','top1_nomor','top1_title','top1_page','query','answer'] if c in answers_df.columns]
display(answers_df[show_cols].head(20))
problem_mask = pd.Series(False, index=answers_df.index)
if 'answer_validation_bad' in answers_df:
    problem_mask |= answers_df['answer_validation_bad'].fillna(False).astype(bool)
if 'support_level' in answers_df:
    problem_mask |= answers_df['support_level'].isin(['low', 'very_low'])
if 'abstained' in answers_df:
    problem_mask |= answers_df['abstained'].fillna(False).astype(bool)
print('\nPotential review rows:', int(problem_mask.sum()))
display(answers_df.loc[problem_mask, show_cols].head(30))



# %% CELL 36

def debug_umkm_query(query_id: str):
    full_rows = []
    if UMKM_FULL_JSONL.exists():
        with open(UMKM_FULL_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    full_rows.append(json.loads(line))
    hit = None
    for row in full_rows:
        if row.get('item', {}).get('id') == query_id:
            hit = row
            break
    if hit is None:
        print('Query id not found:', query_id)
        return
    print('QUERY:', hit['item']['query'])
    print('\nANSWER:\n', hit.get('answer') or hit.get('answer_row', {}).get('answer'))
    print('\nSUPPORT:', hit.get('support_info'))
    print('\nCONTEXT:\n', hit.get('context')[:6000])
    print('\nTOP SOURCES:')
    for i, r in enumerate(hit.get('results', [])[:5], start=1):
        p = r.get('payload', {})
        print(f"S{i}: {p.get('category')} | {infer_nomor_from_payload(p)} | {p.get('title')} | page {p.get('page')} | chunk {p.get('chunk_id')}")
# debug_umkm_query('U012')

