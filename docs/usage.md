# DSN-MUI UMKM RAG Prototype

Panduan ini menjelaskan **inference dan pemrosesan dokumen**. Semua perintah Python dijalankan dari `backend/`, kecuali perintah `cd backend` yang dimulai dari root repo. Untuk preview tanpa model, gunakan [README utama](../README.md). Frontend AMANA dan proxy lokal dijelaskan di [panduan integrasi](integration.md).

Prototype ini adalah sistem **Retrieval-Augmented Generation (RAG)** untuk membantu pelaku UMKM memahami fatwa, ketentuan, dan dalil/kaidah dari dokumen DSN-MUI. Sistem ini sudah dipisahkan menjadi dua bagian besar:

1. **Precomputed pipeline**: memproses dokumen mentah menjadi artifact retrieval.
2. **Runtime/modelling pipeline**: menerima pertanyaan user, melakukan retrieval, optional visual dalil, dan menghasilkan jawaban.

---

## 1. Ringkasan Arsitektur

```text
Dokumen DSN-MUI PDF
        │
        ▼
Precomputed pipeline
(OCR, page extraction, chunking, embedding, BM25 index)
        │
        ▼
Precomputed artifacts
chunks.jsonl + embeddings_bge_m3.npy + bm25_index.pkl
        │
        ▼
Runtime RAG
BGE-M3 query embedding + BM25 + RRF + statistical reranker
        │
        ├── optional Qwen3-VL visual dalil
        │       └── crop PDF → baca teks Arab → klasifikasi dalil
        │
        └── optional Qwen3-VL text-only reasoning
                └── context RAG → JSON answer → fallback jika JSON invalid
```

---

## 2. Struktur Project

```text
DSNMUI_RAG/
├── frontend/                         # React + TypeScript + Vite (AMANA)
├── backend/
│   ├── app_rag_api.py                 # FastAPI endpoint
│   ├── app_streamlit.py               # UI eksperimen Streamlit
│   ├── cli.py                        # Eksperimen terminal
│   ├── requirements.txt              # Dependency inference
│   ├── requirements-test.txt         # Dependency preview/test tanpa model
│   ├── requirements-precompute.txt   # Dependency preprocessing tambahan
│   ├── pyproject.toml
│   ├── .env.example
│   ├── scripts/                      # Crawler, OCR, embedding, worker crop
│   ├── src/dsnmui_rag/                # Config, store, retrieval, context,
│   │                                 # generation, visual dalil, pipeline, utils
│   ├── tests/                        # Tes API tanpa model
│   └── legacy/                       # Export notebook referensi
└── docs/                             # Panduan dan screenshot
```

---

## 3. Precomputed Pipeline

Precomputed pipeline digunakan untuk mengubah PDF DSN-MUI menjadi artifact siap retrieval. Proses ini dijalankan sebelum runtime sehingga API tidak perlu membaca dan memproses ulang PDF setiap user bertanya.

### 3.1 Input

Input utama adalah folder/dataset dokumen DSN-MUI dalam bentuk PDF atau hasil crawl. Metadata dapat berasal dari manifest atau diinfer dari nama file.

### 3.2 Alur Precomputed

```text
Raw PDF DSN-MUI
  → ekstraksi teks per halaman dengan PyMuPDF
  → OCR Arab jika halaman mengandung teks Arab/dalil atau text layer kurang baik
  → normalisasi teks
  → simpan pages.jsonl dan pdf_manifest.jsonl
  → chunking halaman menjadi potongan teks retrieval
  → simpan chunks.jsonl
  → embedding semua chunk dengan BGE-M3
  → simpan embeddings_bge_m3.npy
  → build BM25 lexical index
  → simpan bm25_index.pkl
```

### 3.3 Artifact Wajib

Folder precomputed minimal harus berisi:

```text
dsnmui_rag_precompute_arabic_fix/precomputed/
├── chunks.jsonl
├── embeddings_bge_m3.npy
└── bm25_index.pkl
```

Artifact tambahan yang dapat tersedia:

```text
pages.jsonl
pdf_manifest.jsonl
ocr_stats.json
chunk_ids.json
stats.json
arabic_evidence/crops/
rag_umkm_test_runs/
```

---

## 4. Runtime / Modelling Pipeline

Runtime pipeline berjalan ketika user mengirim pertanyaan melalui Streamlit, CLI, atau FastAPI.

### 4.1 Alur Runtime

```text
User query
  → load artifact precomputed
  → query embedding dengan BGE-M3
  → dense retrieval terhadap embeddings_bge_m3.npy
  → sparse retrieval dengan BM25
  → RRF fusion
  → statistical reranker
  → top-K sources
  → optional visual dalil dengan Qwen3-VL
  → build context
  → optional Qwen3-VL text-only reasoning
  → JSON parsing + structured rendering
  → fallback ekstraktif jika generator tidak aktif / JSON invalid
  → final answer + sources + metadata
```

### 4.2 Mode Runtime

| Mode | `use_visual_dalil` | `use_visual_as_generator` | Keterangan |
|---|---:|---:|---|
| Fast retrieval/fallback | `false` | `false` | Paling cepat. Tidak load Qwen3-VL. Cocok untuk test koneksi dan retrieval. |
| Text-only reasoning | `false` | `true` | Qwen3-VL dipakai untuk menyusun jawaban final dari context. |
| Full visual + reasoning | `true` | `true` | Qwen3-VL membaca dalil Arab dari crop PDF dan juga membuat jawaban final. Paling berat. |
| Visual dalil only | `true` | `false` | Qwen3-VL hanya membaca dalil visual; jawaban final tetap fallback ekstraktif. |

---

## 5. Instalasi

Disarankan menggunakan environment Python 3.10+ dan GPU untuk mode Qwen3-VL.

```bash
cd backend

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Jika menggunakan AI Center/JupyterHub, pastikan environment aktif adalah environment yang memiliki `torch`, `transformers`, `FlagEmbedding`, `fastapi`, dan `uvicorn`.

---

## 6. Environment Variable

Contoh konfigurasi ada di `backend/.env.example`. API preview secara default memakai `RAG_LOAD_PIPELINE=0`. Untuk inference, aktifkan pipeline secara eksplisit. Variable yang paling sering dipakai:

```bash
export PRECOMP_DIR="dsnmui_rag_precompute_arabic_fix/precomputed"
export RAG_LOAD_PIPELINE=1
export FINAL_CONTEXT_K=4
export PRELOAD_BGE=1
export PRELOAD_VL=0
export RAG_API_KEY="kunci-demo-rag"
export CORS_ORIGINS="*"
```

### Konfigurasi Retrieval

```bash
export EMBED_MODEL_NAME="BAAI/bge-m3"
export VECTOR_TOP_K=60
export BM25_TOP_K=60
export HYBRID_TOP_K=36
```

### Konfigurasi Qwen3-VL

```bash
export VISUAL_DALIL_MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
export VISUAL_DALIL_MAX_SOURCES=2
export VISUAL_DALIL_MAX_CROPS_PER_SOURCE=4
export MAX_NEW_TOKENS=520
```

---

## 7. Menjalankan Streamlit

Streamlit digunakan untuk debugging manual dan melihat context/source secara interaktif.

```bash
PYTHONPATH=$PWD/src python -m streamlit run app_streamlit.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

Di sidebar tersedia:

- `Path precomputed`
- `Aktifkan Qwen3-VL visual dalil`
- `Gunakan Qwen3-VL juga untuk reasoning final`
- `Filter kategori`
- `Final context K`

---

## 8. Menjalankan FastAPI

FastAPI digunakan sebagai endpoint model untuk diakses oleh backend website atau frontend demo.

### 8.1 Jalankan API

```bash
cd backend

export RAG_API_KEY="kunci-demo-rag"
export RAG_LOAD_PIPELINE=1
export PRECOMP_DIR="dsnmui_rag_precompute_arabic_fix/precomputed"
export PRELOAD_BGE=1
export PRELOAD_VL=0

PYTHONPATH=$PWD/src uvicorn app_rag_api:app --host 0.0.0.0 --port 8300
```

### 8.2 Health Check Internal

```bash
curl http://127.0.0.1:8300/health
```

Expected response:

```json
{
  "status": "healthy",
  "pipeline_enabled": true,
  "pipeline_loaded": true,
  "precomp_dir": "dsnmui_rag_precompute_arabic_fix/precomputed",
  "preload_bge": true,
  "preload_vl": false,
  "vl_loaded": false,
  "api_key_enabled": true
}
```

---

## 9. Expose API dengan ngrok

Jalankan FastAPI di terminal pertama, lalu jalankan ngrok di terminal kedua:

```bash
ngrok http 8300
```

Ngrok akan memberikan URL seperti:

```text
https://xxxx.ngrok-free.dev
```

### 9.1 Test dari Windows PowerShell

```powershell
Invoke-RestMethod -Uri "https://xxxx.ngrok-free.dev/health"
```

Tes endpoint RAG:

```powershell
$body = @{
  query = "Saya jual produk pre-order di marketplace. Akad apa yang relevan menurut DSN-MUI?"
  expected_focus = ""
  category_filter = "fatwa"
  use_visual_dalil = $false
  use_visual_as_generator = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "https://xxxx.ngrok-free.dev/rag/ask" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ "x-api-key" = "kunci-demo-rag" } `
  -Body $body
```

---

## 10. Dokumentasi Endpoint

FastAPI menyediakan Swagger UI otomatis:

```text
http://127.0.0.1:8300/docs
https://xxxx.ngrok-free.dev/docs
```

### 10.1 `GET /`

Cek service root.

Response contoh:

```json
{
  "status": "ok",
  "service": "DSN-MUI UMKM RAG API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "ask": "/rag/ask"
}
```

### 10.2 `GET /health`

Cek status API dan pipeline.

Auth: tidak wajib.

Response contoh:

```json
{
  "status": "healthy",
  "pipeline_enabled": true,
  "pipeline_loaded": true,
  "precomp_dir": "dsnmui_rag_precompute_arabic_fix/precomputed",
  "preload_bge": true,
  "preload_vl": false,
  "vl_loaded": false,
  "api_key_enabled": true
}
```

### 10.3 `GET /summary`

Mengembalikan ringkasan artifact precomputed.

Auth: wajib jika `RAG_API_KEY` di-set.

Header:

```http
x-api-key: kunci-demo-rag
```

### 10.4 `POST /warmup/vl`

Memuat Qwen3-VL ke RAM/VRAM sebelum request berat.

Auth: wajib jika `RAG_API_KEY` di-set.

Header:

```http
x-api-key: kunci-demo-rag
```

Response contoh:

```json
{
  "status": "ok",
  "message": "Qwen3-VL ready",
  "elapsed_seconds": 91.42
}
```

### 10.5 `POST /rag/ask`

Endpoint utama untuk bertanya ke RAG.

Auth: wajib jika `RAG_API_KEY` di-set.

Header:

```http
Content-Type: application/json
x-api-key: kunci-demo-rag
```

Request body:

```json
{
  "query": "Saya jual produk pre-order di marketplace. Akad apa yang relevan menurut DSN-MUI?",
  "expected_focus": "",
  "category_filter": "fatwa",
  "use_visual_dalil": false,
  "use_visual_as_generator": false,
  "include_debug": false
}
```

Field request:

| Field | Type | Required | Keterangan |
|---|---|---:|---|
| `query` | string | Ya | Pertanyaan user. |
| `expected_focus` | string | Tidak | Catatan fokus opsional. Bisa string kosong. |
| `category_filter` | string/null | Tidak | Filter dokumen: `fatwa`, `pedoman_implementasi`, `talimat`, `unknown`, atau `null`. |
| `use_visual_dalil` | boolean | Tidak | Jika `true`, Qwen3-VL akan membaca crop teks Arab dari PDF. |
| `use_visual_as_generator` | boolean | Tidak | Jika `true`, Qwen3-VL digunakan untuk reasoning final text-only. |
| `include_debug` | boolean | Tidak | Jika `true`, response menyertakan context, prompt, dan item untuk debugging. |

Response body:

```json
{
  "answer": "Jawaban final sebagai plain text.",
  "answer_markdown": "Jawaban final versi markdown.",
  "support_info": {
    "support_score": 0.71,
    "support_level": "high",
    "matched_terms": "...",
    "missing_terms": "...",
    "core_concepts": "...",
    "matched_core_concepts": "...",
    "unsupported_specific_terms": "..."
  },
  "generation_meta": {
    "generation_mode": "json_generated",
    "raw_generation": "...",
    "json_parse_ok": true,
    "schema_version": "v13_short_json"
  },
  "timing": {
    "retrieval_total_ms": 322.05
  },
  "sources": [
    {
      "source": "S1",
      "rank_final": 1,
      "category": "fatwa",
      "nomor": "06",
      "title": "Jual Beli Istishna",
      "page": 2,
      "score_reranker": 0.91,
      "score_vector": 0.72,
      "score_bm25": 12.4,
      "visual_dalil_status": "accepted",
      "dalil_type": "kaidah",
      "arabic_text": "الأصل في المعاملات الإباحة...",
      "confidence": "high",
      "crop": "precomputed/arabic_evidence/crops/...png",
      "preview": "Potongan teks sumber..."
    }
  ],
  "elapsed_seconds": 4.812,
  "debug": null
}
```

### 10.6 `POST /ask`

Alias untuk `POST /rag/ask`. Endpoint ini disediakan agar test lama tetap berjalan.

---

## 11. Contoh Integrasi Backend Node.js

```javascript
const BASE_URL = "https://xxxx.ngrok-free.dev";
const API_KEY = "kunci-demo-rag";

async function askRag(query) {
  const response = await fetch(`${BASE_URL}/rag/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY
    },
    body: JSON.stringify({
      query,
      expected_focus: "",
      category_filter: "fatwa",
      use_visual_dalil: false,
      use_visual_as_generator: false,
      include_debug: false
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`RAG API error: ${response.status} ${errorText}`);
  }

  return await response.json();
}
```

---

## 12. Contoh Request dengan `curl`

### Local AI Center

```bash
curl -X POST "http://127.0.0.1:8300/rag/ask" \
  -H "Content-Type: application/json" \
  -H "x-api-key: kunci-demo-rag" \
  -d '{
    "query": "Saya jual produk pre-order di marketplace. Akad apa yang relevan menurut DSN-MUI?",
    "category_filter": "fatwa",
    "use_visual_dalil": false,
    "use_visual_as_generator": false
  }'
```

### Public ngrok

```bash
curl -X POST "https://xxxx.ngrok-free.dev/rag/ask" \
  -H "Content-Type: application/json" \
  -H "x-api-key: kunci-demo-rag" \
  -d '{
    "query": "Saya jual produk pre-order di marketplace. Akad apa yang relevan menurut DSN-MUI?",
    "category_filter": "fatwa",
    "use_visual_dalil": false,
    "use_visual_as_generator": false
  }'
```

---

## 13. Testing Output Arab di CLI

Jika output Arab terlihat rusak di PowerShell, simpan response ke file atau test dari terminal Linux/AI Center.

```bash
curl -X POST "http://127.0.0.1:8300/rag/ask" \
  -H "Content-Type: application/json" \
  -H "x-api-key: kunci-demo-rag" \
  -d '{
    "query": "Tampilkan dalil atau kaidah tentang akad istishna.",
    "category_filter": "fatwa",
    "use_visual_dalil": true,
    "use_visual_as_generator": true
  }' \
  -o response.json

python -m json.tool response.json | less -R
```

---

## 14. Troubleshooting

### 14.1 `ModuleNotFoundError: No module named 'dsnmui_rag'`

Jalankan dengan `PYTHONPATH` atau install editable package:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
pip install -e .
```

### 14.2 Port sudah digunakan

```bash
ss -ltnp | grep 8300
kill -9 <PID>
```

Atau gunakan port lain:

```bash
PYTHONPATH=$PWD/src uvicorn app_rag_api:app --host 0.0.0.0 --port 8301
```

### 14.3 Ngrok health bisa, tapi POST gagal di PowerShell

Gunakan `Invoke-RestMethod` dan `ConvertTo-Json`, bukan JSON inline yang tidak ter-escape dengan benar.

### 14.4 API mengembalikan `401 Invalid or missing x-api-key`

Pastikan header sesuai:

```http
x-api-key: kunci-demo-rag
```

Atau kosongkan `RAG_API_KEY` hanya untuk local testing.

### 14.5 Full Qwen3-VL lambat

Panggil warmup dulu:

```bash
curl -X POST http://127.0.0.1:8300/warmup/vl \
  -H "x-api-key: kunci-demo-rag"
```

Mode full `use_visual_dalil=true` dan `use_visual_as_generator=true` memang berat karena Qwen3-VL membaca crop visual dan melakukan reasoning final.

### 14.6 Output generator JSON invalid

Jika Qwen3-VL gagal menghasilkan JSON valid, sistem otomatis memakai fallback ekstraktif dari context retrieval. Ini bukan berarti retrieval gagal; artinya hanya format output generatif tidak lolos parser JSON/schema.

---

## 15. Catatan Penggunaan untuk FE/BE

Untuk integrasi dengan website :

1. Backend website memanggil endpoint `POST /rag/ask`.
2. Frontend tidak disarankan memanggil API model langsung agar API key tidak terekspos.
3. Gunakan mode ringan dulu:

```json
{
  "use_visual_dalil": false,
  "use_visual_as_generator": false
}
```

4. Untuk demo full dalil dan reasoning, gunakan:

```json
{
  "use_visual_dalil": true,
  "use_visual_as_generator": true
}
```

5. URL ngrok gratis dapat berubah setiap restart tunnel, jadi base URL harus diperbarui sebelum testing.

---

## 16. Limitasi

- Jawaban hanya berdasarkan corpus DSN-MUI yang tersedia di artifact precomputed.
- Visual dalil bergantung pada kualitas PDF, text layer, crop, dan kemampuan Qwen3-VL membaca teks Arab.
- Teks Arab hasil visual tetap perlu verifikasi jika akan dijadikan kutipan resmi.
- Mode Qwen3-VL membutuhkan GPU/VRAM memadai.
- Endpoint ngrok cocok untuk demo/testing, bukan production permanen.
