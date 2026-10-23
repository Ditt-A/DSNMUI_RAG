import { ArrowUpRight, Github } from "lucide-react";
import { Link } from "react-router-dom";
import { LINKS } from "@/lib/site";
import { Brand } from "./Brand";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-container">
        <div className="footer-main">
          <div className="footer-intro">
            <Brand />
            <p>
              Membantu pelaku usaha memahami pertanyaan mereka, dengan rujukan
              yang bisa dibaca kembali.
            </p>
            <span className="footer-project-label">
              <span aria-hidden="true" /> Prototipe riset independen
            </span>
          </div>
          <nav aria-label="Produk">
            <h2>Jelajahi AMANA</h2>
            <Link to="/chat">Mulai percakapan</Link>
            <Link to="/#cara-kerja">Cara kerja</Link>
            <Link to="/#contoh-pertanyaan">Contoh pertanyaan</Link>
          </nav>
          <nav aria-label="Dokumen referensi">
            <h2>Sumber referensi</h2>
            <a href={LINKS.fatwa} target="_blank" rel="noreferrer">
              Fatwa DSN-MUI <ArrowUpRight />
            </a>
            <a href={LINKS.pedoman} target="_blank" rel="noreferrer">
              Pedoman implementasi <ArrowUpRight />
            </a>
            <a href={LINKS.talimat} target="_blank" rel="noreferrer">
              Ta’limat <ArrowUpRight />
            </a>
          </nav>
          <nav aria-label="Tentang proyek">
            <h2>Tentang proyek</h2>
            <Link to="/about">Tentang AMANA</Link>
            <a href={LINKS.docs} target="_blank" rel="noreferrer">
              Dokumentasi <ArrowUpRight />
            </a>
            <a href={LINKS.github} target="_blank" rel="noreferrer">
              <Github size={14} /> Kode sumber <ArrowUpRight />
            </a>
          </nav>
        </div>
        <div className="footer-note">
          AMANA adalah alat bantu penelusuran, bukan layanan resmi DSN-MUI.
          Periksa kembali jawaban dan kutipan pada dokumen aslinya.
        </div>
        <div className="footer-bottom">
          <p>
            © {new Date().getFullYear()} AMANA. Dikembangkan untuk riset dan
            pembelajaran.
          </p>
          <div>
            <Link to="/privacy">Privasi</Link>
            <Link to="/terms">Ketentuan penggunaan</Link>
            <a href={LINKS.issues} target="_blank" rel="noreferrer">
              Sampaikan masukan <ArrowUpRight size={13} />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
