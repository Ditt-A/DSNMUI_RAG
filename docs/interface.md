# Antarmuka AMANA

Landing page mengimplementasikan rancangan v3: palet emerald, putih hangat, dan sage; hero terpusat; preview chat lebar dengan panel sumber; bagian penjelasan rujukan; contoh pertanyaan; dan footer proyek.

## Halaman dan tujuan navigasi

| Rute | Isi |
|---|---|
| `/` | Landing page AMANA |
| `/#cara-kerja` | Penjelasan penelusuran dan sumber dokumen |
| `/#contoh-pertanyaan` | Contoh pertanyaan yang dapat dibuka |
| `/chat` | Percakapan AMANA |
| `/chat?question=...` | Percakapan dengan draft awal; tidak mengirim pesan otomatis |
| `/about` | Tujuan dan konteks proyek |
| `/privacy` | Penanganan percakapan pada prototipe |
| `/terms` | Informasi penggunaan dan batasan prototipe |
| Rute lain | Halaman 404 dengan navigasi pemulihan |

CTA utama dan penutup membuka chat. Tombol kirim pada preview membuka chat dengan draft yang diketik atau contoh yang sedang aktif. Accordion menyediakan tautan untuk memakai pertanyaan tersebut. Tautan sumber membuka direktori dokumen DSN-MUI yang relevan pada tab baru.

Tab preview memiliki navigasi panah kiri/kanan serta Home/End. Panel referensi dapat dibuka, ditutup, dan ditutup melalui Escape. Pada ponsel panel sumber berawal tertutup agar percakapan tetap terbaca. Menu navigasi ponsel dapat ditutup melalui Escape atau setelah memilih tautan.

Pada chat, tombol Beranda kembali ke landing page. Pengaturan membuka filter pertanyaan. Chat Baru membersihkan pesan serta draft awal dan membatalkan request yang masih berjalan. Font Plus Jakarta Sans disertakan sebagai aset lokal dalam hasil build.

## Transisi

- Hero: masuk dengan opacity dan perpindahan vertikal 12 px, stagger pendek antarelemen.
- Bagian halaman: reveal satu kali ketika memasuki viewport.
- Pergantian tab: crossfade singkat pada isi preview, tanpa request inference.
- Panel sumber: opacity dan perpindahan kecil; lebar area chat menyesuaikan secara halus pada desktop.
- Accordion dan menu ponsel: perubahan tinggi dan opacity.
- Perpindahan halaman: fade singkat, diikuti pergeseran vertikal 5 px.
- Hover CTA: perubahan warna dan gerak panah kecil.

Pengaturan `prefers-reduced-motion` menghentikan animasi perpindahan dan scroll halus. Semua tombol dan tautan memiliki fokus keyboard yang terlihat.

## Pengujian browser tanpa model

Jalankan dari `frontend/`:

```bash
npm ci
npx playwright install chromium
npm run test:e2e
```

Playwright menjalankan frontend terpisah pada port 4173. Tes berjalan pada viewport desktop dan ponsel, dengan request jawaban diganti response fixture. Backend, dataset, BGE-M3, dan Qwen3-VL tidak dibutuhkan. Untuk memakai server frontend yang sudah berjalan, set `PLAYWRIGHT_BASE_URL`.

Pengujian mencakup CTA, anchor, tab dengan keyboard, panel sumber, draft pertanyaan, kirim pesan dengan fixture, accordion, halaman footer, reload rute langsung, filter chat, Chat Baru, 404, dan reduced motion.

## Deployment

Server statis harus mengarahkan rute frontend ke `index.html` agar deep link dan refresh bekerja. Konfigurasi proxy API serta fallback dibahas di [panduan integrasi](integration.md).
