# FastAPI + ngrok quick commands

Jalankan dari root project:

```bash
cd ~/adit/pidi-testing/dsnmui_rag_prototype
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Jalankan FastAPI endpoint RAG

Set API key untuk ngrok/public demo:

```bash
export RAG_API_KEY="kunci-demo-rag"
export PRECOMP_DIR="dsnmui_rag_precompute_arabic_fix/precomputed"
export PRELOAD_BGE=1
export PRELOAD_VL=0
```

Jalankan API:

```bash
PYTHONPATH=$PWD/src uvicorn app_rag_api:app --host 0.0.0.0 --port 8300
```

Cek internal:

```bash
curl http://127.0.0.1:8300/health
```

## 3. Jalankan ngrok

Terminal lain:

```bash
ngrok http 8300
```

Ambil URL `https://xxxx.ngrok-free.dev` / `https://xxxx.ngrok-free.app`.

## 4. Test dari Windows PowerShell

```powershell
$body = @{
  query = "Saya jual produk pre-order di marketplace. Akad apa yang relevan menurut DSN-MUI?"
  category_filter = "fatwa"
  use_visual_dalil = $false
  use_visual_as_generator = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "https://xxxx.ngrok-free.dev/rag/ask" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{"x-api-key"="kunci-demo-rag"} `
  -Body $body
```

## 5. Body untuk BE teman

Endpoint:

```text
POST /rag/ask
```

Header:

```text
Content-Type: application/json
x-api-key: kunci-demo-rag
```

Body ringan tanpa VLM:

```json
{
  "query": "Saya jual produk pre-order di marketplace. Akad apa yang relevan menurut DSN-MUI?",
  "category_filter": "fatwa",
  "use_visual_dalil": false,
  "use_visual_as_generator": false
}
```

Body full Qwen3-VL:

```json
{
  "query": "Saya jual produk pre-order di marketplace. Akad apa yang relevan menurut DSN-MUI?",
  "category_filter": "fatwa",
  "use_visual_dalil": true,
  "use_visual_as_generator": true
}
```

Preload Qwen3-VL sebelum request berat:

```bash
curl -X POST http://127.0.0.1:8300/warmup/vl -H "x-api-key: kunci-demo-rag"
```
