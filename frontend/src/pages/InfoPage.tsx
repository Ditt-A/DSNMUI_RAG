import { Link } from "react-router-dom";
import { ArrowLeft, ArrowRight, ArrowUpRight } from "lucide-react";
import { SiteHeader } from "@/components/site/SiteHeader";
import { SiteFooter } from "@/components/site/SiteFooter";
import { LINKS } from "@/lib/site";
import { Artwork } from "@/components/site/Artwork";

const pages = {
  about: {
    label: "Tentang proyek",
    title: "Rujukan yang bisa dibaca kembali.",
    intro:
      "AMANA adalah prototipe riset untuk membantu pelaku UMKM menelusuri dokumen DSN-MUI melalui percakapan.",
    sections: [
      [
        "Berangkat dari pertanyaan usaha",
        "Istilah dalam dokumen tidak selalu sama dengan bahasa yang digunakan sehari-hari. AMANA membantu menghubungkan pertanyaan tentang usaha dengan dokumen yang relevan, lalu menampilkan referensi agar sumbernya dapat diperiksa kembali.",
      ],
      [
        "Cara menggunakan AMANA",
        "Ceritakan situasi usaha Anda dengan jelas. Periksa rujukan yang menyertai jawaban, kemudian baca dokumen aslinya secara utuh. Anda juga dapat memilih kategori fatwa, pedoman implementasi, atau ta’limat untuk memperjelas penelusuran.",
      ],
      [
        "Riset independen",
        "AMANA bukan layanan resmi DSN-MUI dan tidak menyatakan afiliasi atau sertifikasi dari DSN-MUI. Dokumen tetap menjadi milik dan tanggung jawab penerbitnya. Jawaban prototipe perlu diverifikasi dan tidak menggantikan konsultasi dengan pihak yang kompeten.",
      ],
      [
        "Dikembangkan secara terbuka",
        "Kode dan dokumentasi proyek tersedia di GitHub. Masukan tentang pengalaman penggunaan dan hasil penelusuran dapat disampaikan melalui halaman Issues proyek.",
      ],
    ],
  },
  privacy: {
    label: "Informasi penggunaan",
    title: "Privasi percakapan Anda.",
    intro:
      "Halaman ini menjelaskan penanganan percakapan pada prototipe AMANA yang tersedia dalam repository proyek.",
    sections: [
      [
        "Pesan selama sesi",
        "Tampilan chat menyimpan pesan di memori browser selama sesi halaman aktif. Prototipe ini tidak menyediakan akun, penyimpanan riwayat permanen, atau sinkronisasi percakapan antarperangkat. Memuat ulang halaman akan mengosongkan tampilan percakapan.",
      ],
      [
        "Pemrosesan pertanyaan",
        "Ketika Anda mengirim pesan, isi pertanyaan dan pilihan filter diteruskan ke layanan backend untuk diproses. Membuka contoh pada landing page atau mengisi pertanyaan awal tidak mengirimkannya sebagai permintaan jawaban.",
      ],
      [
        "Informasi yang sebaiknya dihindari",
        "Gunakan konteks usaha secukupnya. Hindari memasukkan kata sandi, nomor identitas, informasi rekening, atau data pribadi pelanggan yang tidak diperlukan untuk pertanyaan.",
      ],
      [
        "Layanan di luar AMANA",
        "Tautan dokumen DSN-MUI dan GitHub membuka situs pihak lain. Kebijakan serta pencatatan akses pada situs tersebut mengikuti pengelolanya masing-masing. Pengelola deployment AMANA juga dapat memiliki pengaturan server dan pencatatan akses tersendiri.",
      ],
    ],
  },
  terms: {
    label: "Informasi penggunaan",
    title: "Gunakan sebagai awal penelusuran.",
    intro:
      "AMANA membantu menemukan rujukan. Keputusan usaha tetap memerlukan pembacaan sumber dan pertimbangan sesuai konteks Anda.",
    sections: [
      [
        "Tujuan prototipe",
        "AMANA dikembangkan untuk riset, pembelajaran, dan eksplorasi penelusuran dokumen. Ketersediaan layanan dan cakupan dokumen dapat berbeda pada setiap deployment.",
      ],
      [
        "Periksa sumber aslinya",
        "Jawaban dapat tidak lengkap atau keliru. Selalu periksa judul, isi, dan konteks dokumen yang dirujuk. Untuk keputusan yang memerlukan penilaian khusus, mintalah penjelasan dari pihak yang kompeten.",
      ],
      [
        "Contoh pada landing page",
        "Percakapan dan cuplikan rujukan pada landing page adalah contoh tampilan, bukan hasil penelusuran langsung. Memilih contoh mengisi pertanyaan awal; jawaban diproses setelah pesan dikirim dari halaman percakapan.",
      ],
      [
        "Tanggung jawab penggunaan",
        "Gunakan layanan secara wajar dan jangan memasukkan informasi rahasia atau data pihak lain tanpa dasar yang sesuai. Hak atas dokumen sumber tetap mengikuti penerbitnya. Jika menemukan kekeliruan, sampaikan masukan melalui repository proyek.",
      ],
    ],
  },
};

export default function InfoPage({ kind }: { kind: keyof typeof pages }) {
  const page = pages[kind];
  return (
    <div className="information-page">
      <SiteHeader />
      <main
        id="main-content"
        className="site-container information-main"
        tabIndex={-1}
      >
        <Link to="/" className="back-to-home">
          <ArrowLeft size={16} /> Kembali ke beranda
        </Link>
        <div
          className={
            kind === "about"
              ? "information-lead information-lead--illustrated"
              : "information-lead"
          }
        >
          <div>
            <p className="information-label">{page.label}</p>
            <h1>{page.title}</h1>
            <p className="information-intro">{page.intro}</p>
          </div>
          {kind === "about" && (
            <Artwork
              name="owner"
              className="about-owner-art"
              alt="Ilustrasi pemilik usaha menyiapkan pesanan pelanggan."
              priority
            />
          )}
        </div>
        <div className="information-sections">
          {page.sections.map(([title, content]) => (
            <section key={title}>
              <h2>{title}</h2>
              <p>{content}</p>
            </section>
          ))}
        </div>
        <div className="information-actions">
          <Link to="/chat" className="site-button">
            Buka AMANA <ArrowRight size={17} />
          </Link>
          <a
            href={kind === "about" ? LINKS.github : LINKS.issues}
            target="_blank"
            rel="noreferrer"
          >
            {kind === "about" ? "Lihat kode sumber" : "Sampaikan masukan"}{" "}
            <ArrowUpRight size={16} />
          </a>
        </div>
      </main>
      <SiteFooter />
    </div>
  );
}
