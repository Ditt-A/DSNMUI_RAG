# Asal kode dan aset

| Bagian | Sumber | Snapshot |
|---|---|---|
| Backend FastAPI, package RAG, preprocessing, dan notebook legacy | [Ditt-A/DSNMUI_RAG](https://github.com/Ditt-A/DSNMUI_RAG) | `0e482ee37e644dd3dfcca1bcbc4cd327e6b404c5` |
| Frontend AMANA | [sulthonaw/amana-chatbot](https://github.com/sulthonaw/amana-chatbot) | `88b4f85ba13435ead6bf84cd7f2d55365a581c65` |
| Screenshot prototipe | [Website-Porto · amana.png](https://github.com/Ditt-A/Website-Porto/blob/f188a6af30baaa933b503dc71d5dea4c163546bc/public/images/projects/amana.png) | `f188a6af30baaa933b503dc71d5dea4c163546bc` |

Frontend diimpor ke `frontend/`, dengan komponen, hook, tipe, dan utility dirapikan di dalam `frontend/src/`. Konfigurasi AI Studio yang tidak digunakan tidak ikut diimpor. Request chat menggunakan proxy Vite, dan API key berada pada konfigurasi server.

Snapshot frontend sumber tidak memiliki implementasi backend Node.js terpisah; FastAPI dari DSNMUI_RAG menjadi backend aplikasi gabungan. Repo frontend sumber bersifat privat sehingga akses tautannya bergantung pada izin GitHub, tetapi kode yang diperlukan sudah tersedia di monorepo ini.

Screenshot disalin utuh ke [`images/amana-prototype.png`](images/amana-prototype.png) agar README dapat menampilkannya dari repo ini. Screenshot menunjukkan contoh sesi prototipe terdahulu; preview tanpa model tidak menghasilkan jawaban tersebut.
