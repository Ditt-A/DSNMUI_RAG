# Menyiapkan artifact retrieval

Seluruh perintah di halaman ini dijalankan dari `backend/`. Langkah preprocessing dapat mengunduh dokumen dan memuat model; langkah ini tidak diperlukan untuk preview UI/API.

## Dependency

```bash
python -m pip install -r requirements-precompute.txt
python -m pip install -e .
```

OCR Arab memerlukan `paddleocr` dan paket PaddlePaddle yang sesuai dengan mesin. Kedua paket bersifat opsional dan tidak dipasang oleh requirements preview/test. Stage embedding juga membangun indeks Qdrant lokal; runtime utama membaca embedding NumPy dan BM25.

## 1. Crawling dokumen

```bash
python scripts/dsnmui_crawler.py --out data_dsnmui --delay 1.0
```

Crawler menyimpan PDF per kategori, `manifest.jsonl`, `manifest.csv`, dan `errors.jsonl`. Gunakan `--dry-run` untuk hanya mendata URL dan metadata, atau `--categories fatwa` untuk memilih kategori.

## 2. Ekstraksi halaman dan OCR

```bash
python scripts/01_dsnmui_ocr_pages_arabic_fix.py \
  --data-dir data_dsnmui \
  --project-dir dsnmui_rag_precompute_arabic_fix \
  --ocr-mode arabic_or_dalil_or_low_text
```

Input juga dapat berupa ZIP melalui `--zip data_dsnmui.zip`. Output berada di `<project-dir>/precomputed/`: `pages.jsonl`, `pdf_manifest.jsonl`, dan `ocr_stats.json`.

Jika hanya ingin ekstraksi text layer tanpa OCR, gunakan `--ocr-mode none --use-gpu false`. Kualitas teks Arab tetap bergantung pada PDF sumber.

## 3. Chunking, embedding, dan indexing

```bash
python scripts/02_dsnmui_embed_index.py \
  --project-dir dsnmui_rag_precompute_arabic_fix
```

Stage ini memuat BGE-M3. Gunakan environment dengan resource yang sesuai. Folder precomputed kemudian memuat:

```text
dsnmui_rag_precompute_arabic_fix/precomputed/
├── pages.jsonl
├── pdf_manifest.jsonl
├── chunks.jsonl                 # Wajib saat runtime
├── embeddings_bge_m3.npy        # Wajib saat runtime
├── bm25_index.pkl               # Wajib saat runtime
├── chunk_ids.json
├── stats.json
└── qdrant_local/
```

## 4. Konfigurasi runtime

Atur di `backend/.env`:

```dotenv
RAG_LOAD_PIPELINE=1
PRECOMP_DIR=dsnmui_rag_precompute_arabic_fix/precomputed
```

Path relatif dihitung dari direktori tempat backend dijalankan. Jika artifact berada di lokasi lama sebelum penataan repo, gunakan path absolut ke folder tersebut. Untuk visual dalil, path PDF yang tercatat di artifact juga harus tetap valid.

Dataset, indeks, crop, model weights, serta hasil inference masuk `.gitignore`; repo menyimpan kode dan dokumentasi. `scripts/arabic_ocr_crop_worker.py` tersedia untuk OCR crop tambahan, sementara export notebook lama tersimpan di `legacy/` sebagai referensi.
