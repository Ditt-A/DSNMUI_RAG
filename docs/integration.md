# Integrasi AMANA dan FastAPI

Frontend berada di `frontend/` dan backend di `backend/`. Endpoint dan format JSON backend tetap menggunakan `POST /rag/ask`; tipe yang ditampilkan frontend berada di `frontend/src/types/rag.ts`.

## Pengembangan lokal

Jalankan backend dari `backend/` sesuai [README](../README.md). Vite berjalan dari `frontend/` di port 3000 dan meneruskan request `/rag/*` ke FastAPI.

Isi `frontend/.env.local`:

```dotenv
RAG_API_URL=http://127.0.0.1:8300
RAG_API_KEY=ubah-ini-untuk-demo
```

`RAG_API_URL` adalah base URL tanpa `/rag/ask`. Nilai `RAG_API_KEY` harus sama dengan `backend/.env`. Kedua variable dibaca konfigurasi server Vite dan tidak memakai prefix `VITE_`, sehingga key tidak dimasukkan ke bundle browser. Restart Vite setelah mengubah environment.

```text
Browser → http://localhost:3000/rag/ask
        → proxy Vite + header x-api-key
        → http://127.0.0.1:8300/rag/ask
```

Frontend tidak lagi memakai `VITE_API_URL` atau key demo yang ditulis langsung dalam hook chat.

## Preview tanpa model

`backend/.env.example` menggunakan:

```dotenv
RAG_LOAD_PIPELINE=0
PRELOAD_BGE=0
PRELOAD_VL=0
```

Dalam konfigurasi ini startup tidak membuat pipeline, membaca artifact, mengunduh bobot, atau memuat model. `GET /health` mengembalikan `status: "disabled"` dan `pipeline_loaded: false`. Endpoint inference serta summary mengembalikan 503 setelah pemeriksaan API key.

Untuk inference sungguhan, install `backend/requirements.txt`, siapkan artifact, set `RAG_LOAD_PIPELINE=1`, lalu restart backend. Meski `PRELOAD_BGE=0`, pertanyaan pertama tetap memuat BGE-M3. Opsi visual dan generator memuat Qwen3-VL bila dipakai. Streamlit, CLI, serta script preprocessing adalah alat inference/pemrosesan langsung dan tidak mengikuti flag preview API.

## Build dan preview lokal

```bash
cd frontend
npm ci
npm run build
npm run preview
```

`npm run build` memeriksa TypeScript lalu menghasilkan `frontend/dist/`. `npm run preview` juga menggunakan proxy lokal dari konfigurasi Vite, tetapi hanya ditujukan untuk mengecek hasil build.

Landing page berada di `/`, sedangkan chat ada di `/chat`. Halaman informasi memakai `/about`, `/privacy`, dan `/terms`. Semua halaman tersebut menggunakan client-side routing. Detail interaksi dan pengujiannya tersedia di [panduan antarmuka](interface.md).

## Deployment statis

Proxy pada `vite.config.ts` tidak ikut menjadi server dalam hasil build. Untuk hosting produksi:

1. Sajikan `frontend/dist/` melalui web server.
2. Teruskan `/rag/*` dari origin yang sama ke backend FastAPI.
3. Jika backend memakai `RAG_API_KEY`, sisipkan header `x-api-key` pada reverse proxy di sisi server. Jangan memasukkan key ke JavaScript frontend.
4. Tambahkan autentikasi dan pembatasan request di layanan publik; key backend pada proxy bukan autentikasi pengguna.
5. Atur fallback route frontend ke `index.html` agar refresh langsung pada `/chat`, `/about`, `/privacy`, dan `/terms` tetap membuka aplikasi. Contohnya, gunakan `try_files $uri $uri/ /index.html;` pada location frontend Nginx. Rute `/rag/*` harus ditangani location proxy tersendiri.

Jika backend berjalan di mesin lain, ganti target proxy lokal melalui `RAG_API_URL`. Untuk tunnel demo lihat [FastAPI dan ngrok](deployment.md).

## Asal integrasi

Snapshot `main` dari `sulthonaw/amana-chatbot` berisi aplikasi React/Vite yang memanggil FastAPI, tanpa implementasi server Node.js terpisah. Backend yang digunakan di monorepo ini adalah API dan package Python dari `Ditt-A/DSNMUI_RAG`. Lihat [catatan sumber](sources.md).
