import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ArrowRight, Menu, X } from "lucide-react";
import { Brand } from "./Brand";

export function SiteHeader() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const reduced = useReducedMotion();

  useEffect(() => {
    setOpen(false);
  }, [location.pathname, location.hash]);
  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [open]);

  return (
    <header className="site-header">
      <div className="site-container site-header-inner">
        <Brand />
        <nav className="desktop-navigation" aria-label="Navigasi utama">
          <Link to="/#cara-kerja">Cara kerja</Link>
          <Link to="/#contoh-pertanyaan">Contoh pertanyaan</Link>
          <Link to="/about">Tentang AMANA</Link>
        </nav>
        <div className="site-header-actions">
          <Link className="site-button site-button--small" to="/chat">
            Buka AMANA <ArrowRight size={17} aria-hidden="true" />
          </Link>
          <button
            className="site-icon-button mobile-menu-toggle"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            aria-controls="mobile-navigation"
            aria-label={open ? "Tutup navigasi" : "Buka navigasi"}
          >
            {open ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>
      <AnimatePresence>
        {open && (
          <motion.nav
            id="mobile-navigation"
            className="mobile-navigation"
            aria-label="Navigasi seluler"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: reduced ? 0 : 0.2 }}
          >
            <Link to="/#cara-kerja" onClick={() => setOpen(false)}>
              Cara kerja <ArrowRight size={16} />
            </Link>
            <Link to="/#contoh-pertanyaan" onClick={() => setOpen(false)}>
              Contoh pertanyaan <ArrowRight size={16} />
            </Link>
            <Link to="/about" onClick={() => setOpen(false)}>
              Tentang AMANA <ArrowRight size={16} />
            </Link>
          </motion.nav>
        )}
      </AnimatePresence>
    </header>
  );
}
