import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  ChevronDown,
  FileText,
  Menu,
  X,
} from "lucide-react";
import { LINKS } from "@/lib/site";
import { Brand } from "./Brand";

const sources = [
  {
    name: "Fatwa DSN-MUI",
    description: "Telusuri ketentuan dan akad",
    href: LINKS.fatwa,
    Icon: FileText,
  },
  {
    name: "Pedoman implementasi",
    description: "Baca pedoman penerapan",
    href: LINKS.pedoman,
    Icon: BookOpen,
  },
  {
    name: "Ta’limat",
    description: "Temukan surat edaran",
    href: LINKS.talimat,
    Icon: FileText,
  },
];

export function SiteHeader() {
  const [open, setOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const sourceMenu = useRef<HTMLDivElement>(null);
  const sourceToggle = useRef<HTMLButtonElement>(null);
  const location = useLocation();
  const reduced = useReducedMotion();

  useEffect(() => {
    setOpen(false);
    setSourcesOpen(false);
  }, [location.pathname, location.hash]);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    if (!open && !sourcesOpen) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (sourcesOpen) sourceToggle.current?.focus();
        setOpen(false);
        setSourcesOpen(false);
      }
    };
    const outside = (event: PointerEvent) => {
      if (
        sourceMenu.current &&
        !sourceMenu.current.contains(event.target as Node)
      )
        setSourcesOpen(false);
    };
    window.addEventListener("keydown", close);
    document.addEventListener("pointerdown", outside);
    return () => {
      window.removeEventListener("keydown", close);
      document.removeEventListener("pointerdown", outside);
    };
  }, [open, sourcesOpen]);
  const active = (hash: string) =>
    location.pathname === "/" && location.hash === hash;

  return (
    <header className={`site-header${scrolled ? " is-scrolled" : ""}`}>
      <div className="site-container site-header-inner">
        <Brand tagline />
        <nav className="desktop-navigation" aria-label="Navigasi utama">
          <Link
            to="/#cara-kerja"
            className={active("#cara-kerja") ? "is-active" : ""}
            aria-current={active("#cara-kerja") ? "location" : undefined}
          >
            Cara kerja
          </Link>
          <Link
            to="/#contoh-pertanyaan"
            className={active("#contoh-pertanyaan") ? "is-active" : ""}
            aria-current={active("#contoh-pertanyaan") ? "location" : undefined}
          >
            Contoh pertanyaan
          </Link>
          <div className="nav-source-menu" ref={sourceMenu}>
            <button
              ref={sourceToggle}
              className={`nav-source-toggle${sourcesOpen ? " is-active" : ""}`}
              aria-expanded={sourcesOpen}
              aria-controls="nav-source-dropdown"
              onClick={() => setSourcesOpen(!sourcesOpen)}
            >
              Sumber dokumen <ChevronDown size={14} />
            </button>
            <AnimatePresence>
              {sourcesOpen && (
                <motion.div
                  className="nav-source-dropdown"
                  id="nav-source-dropdown"
                  initial={{ opacity: 0, y: reduced ? 0 : 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: reduced ? 0 : 4 }}
                  transition={{ duration: reduced ? 0 : 0.16 }}
                >
                  <p>Dokumen DSN-MUI</p>
                  {sources.map(({ name, description, href, Icon }) => (
                    <a
                      key={name}
                      href={href}
                      target="_blank"
                      rel="noreferrer"
                      onClick={() => setSourcesOpen(false)}
                    >
                      <span className="nav-source-icon">
                        <Icon size={18} />
                      </span>
                      <span>
                        <strong>{name}</strong>
                        <small>{description}</small>
                      </span>
                      <ArrowUpRight size={14} />
                    </a>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <Link
            to="/about"
            className={location.pathname === "/about" ? "is-active" : ""}
            aria-current={location.pathname === "/about" ? "page" : undefined}
          >
            Tentang AMANA
          </Link>
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
            <div className="mobile-nav-content">
              <span className="mobile-nav-label">Jelajahi AMANA</span>
              <Link to="/#cara-kerja" onClick={() => setOpen(false)}>
                Cara kerja <ArrowRight size={16} />
              </Link>
              <Link to="/#contoh-pertanyaan" onClick={() => setOpen(false)}>
                Contoh pertanyaan <ArrowRight size={16} />
              </Link>
              <Link to="/about" onClick={() => setOpen(false)}>
                Tentang AMANA <ArrowRight size={16} />
              </Link>
              <span className="mobile-nav-label">Sumber referensi</span>
              {sources.map(({ name, href, Icon }) => (
                <a
                  key={name}
                  href={href}
                  target="_blank"
                  rel="noreferrer"
                  onClick={() => setOpen(false)}
                >
                  <span>
                    <Icon size={16} /> {name}
                  </span>
                  <ArrowUpRight size={15} />
                </a>
              ))}
            </div>
          </motion.nav>
        )}
      </AnimatePresence>
    </header>
  );
}
