# Antarmuka AMANA

Landing page mengimplementasikan rancangan v3: palet emerald, putih hangat, dan sage; hero terpusat; preview chat lebar dengan panel sumber; bagian penjelasan rujukan; contoh pertanyaan; dan footer proyek.

Delapan [ilustrasi khusus](assets/README.md) dipasang sesuai konteks pada hero, penjelasan sumber, contoh pertanyaan, penutup, halaman tentang proyek, dan kondisi chat kosong. PNG asli tersedia dalam paket aset pengguna; website memakai WebP transparan dengan dimensi yang sama. Logo AMANA dibuat sebagai SVG agar tetap tajam pada navbar, footer, dan favicon.

Navbar memiliki menu aktif, perubahan bayangan saat scroll, serta dropdown sumber dokumen yang dapat ditutup dengan Escape atau klik di luar panel. Pada ponsel, sumber rujukan tersedia di menu navigasi bersama tautan halaman.

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

- Hero: masuk dengan opacity dan perpindahan vertikal 12 px; ilustrasi menyelesaikan gerakan masuk satu kali setelah gambar dimuat.
- Bagian halaman: reveal dari bawah atau samping saat memasuki viewport, dengan urutan pendek pada daftar manfaat, pertanyaan, dokumen, dan kolom footer.
- Pergantian tab: indikator aktif bergeser memakai spring; pertanyaan, jawaban, dan isi referensi berganti dengan fade dan perpindahan kecil, tanpa request inference.
- Panel sumber: masuk dari samping; lebar area chat menyesuaikan secara halus pada desktop.
- Accordion: perubahan tinggi diikuti masuknya isi jawaban; ilustrasi dan ikon merespons hover atau fokus.
- Navbar: dropdown dan tautan menu masuk berurutan, ikon menu berganti, serta garis tipis mengikuti progres scroll.
- Perpindahan halaman: fade 220 ms dengan gerak masuk 8 px dan keluar 3 px.
- Gambar: fade setelah dimuat; kartu sumber, tombol, dan ikon merespons hover dengan gerakan singkat.

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
