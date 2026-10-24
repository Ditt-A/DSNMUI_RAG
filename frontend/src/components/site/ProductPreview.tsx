import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  ArrowRight,
  BookOpen,
  ExternalLink,
  FileText,
  History,
  Info,
  MessageSquarePlus,
  PanelRightOpen,
  Send,
  ShieldCheck,
  X,
} from "lucide-react";
import { chatLink, EXAMPLES } from "@/lib/site";

export function ProductPreview() {
  const [active, setActive] = useState(0);
  const [sourceOpen, setSourceOpen] = useState(
    () => !window.matchMedia("(max-width: 640px)").matches,
  );
  const [draft, setDraft] = useState("");
  const example = EXAMPLES[active];
  const reduced = useReducedMotion();
  const navigate = useNavigate();
  const openChat = (event: FormEvent) => {
    event.preventDefault();
    navigate(chatLink(draft.trim() || example.question));
  };

  return (
    <div className="product-stage" id="preview-produk">
      <p className="preview-label">Contoh percakapan</p>
      <div
        className="preview-tabs"
        role="tablist"
        aria-label="Pilih contoh percakapan"
      >
        {EXAMPLES.map((item, index) => (
          <button
            key={item.id}
            id={`tab-${item.id}`}
            role="tab"
            aria-selected={active === index}
            aria-controls="preview-conversation"
            tabIndex={active === index ? 0 : -1}
            className={active === index ? "is-active" : ""}
            onClick={() => setActive(index)}
            onKeyDown={(event) => {
              let next = index;
              if (event.key === "ArrowRight")
                next = (index + 1) % EXAMPLES.length;
              else if (event.key === "ArrowLeft")
                next = (index + EXAMPLES.length - 1) % EXAMPLES.length;
              else if (event.key === "Home") next = 0;
              else if (event.key === "End") next = EXAMPLES.length - 1;
              else return;
              event.preventDefault();
              setActive(next);
              document.getElementById(`tab-${EXAMPLES[next].id}`)?.focus();
            }}
          >
            {active === index && (
              <motion.span
                className="preview-tab-indicator"
                layoutId="preview-tab-indicator"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 420, damping: 36 }
                }
              />
            )}
            <span>{item.tab}</span>
          </button>
        ))}
      </div>

      <div
        className={`preview-window${sourceOpen ? "" : " preview-window--source-closed"}`}
      >
        <aside className="preview-sidebar" aria-label="Menu contoh aplikasi">
          <Link to="/chat" className="preview-new-chat">
            <MessageSquarePlus size={16} /> Chat Baru
          </Link>
          <p className="preview-sidebar-label">Riwayat</p>
          <p className="preview-empty-history">
            <History size={14} /> Belum ada riwayat chat
          </p>
          <div className="preview-sidebar-bottom">
            <p>
              <ShieldCheck size={14} /> Jawaban perlu diperiksa kembali pada
              sumber aslinya.
            </p>
            <Link to="/about">
              <Info size={15} /> Tentang AMANA
            </Link>
          </div>
        </aside>

        <div className="preview-chat-panel">
          <div className="preview-chat-header">
            <span className="preview-avatar" aria-hidden="true">
              A
            </span>
            <div>
              <strong>AMANA</strong>
              <span>Asisten DSN-MUI untuk UMKM</span>
            </div>
            <button
              id="preview-source-toggle"
              className="site-icon-button preview-reopen-source"
              aria-controls="preview-source"
              aria-label={
                sourceOpen
                  ? "Tutup sumber referensi"
                  : "Tampilkan sumber referensi"
              }
              aria-expanded={sourceOpen}
              onClick={() => setSourceOpen(!sourceOpen)}
            >
              <PanelRightOpen size={18} />
            </button>
          </div>
          <div
            className="preview-conversation"
            id="preview-conversation"
            role="tabpanel"
            aria-labelledby={`tab-${example.id}`}
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={example.id}
                initial={{ opacity: 0, x: reduced ? 0 : 7 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: reduced ? 0 : -5 }}
                transition={{ duration: reduced ? 0 : 0.2 }}
              >
                <motion.p
                  className="preview-question"
                  initial={reduced ? false : { opacity: 0, y: 7 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: reduced ? 0 : 0.28,
                    delay: reduced ? 0 : 0.03,
                  }}
                >
                  {example.question}
                </motion.p>
                <motion.div
                  className="preview-answer-row"
                  initial={reduced ? false : { opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: reduced ? 0 : 0.32,
                    delay: reduced ? 0 : 0.11,
                  }}
                >
                  <span className="preview-avatar" aria-hidden="true">
                    A
                  </span>
                  <div className="preview-answer">
                    <p>
                      Berikut rujukan yang dapat Anda telusuri untuk memahami
                      akad yang relevan.
                    </p>
                    <button
                      className="preview-citation"
                      onClick={() => setSourceOpen(true)}
                      aria-controls="preview-source"
                      aria-expanded={sourceOpen}
                    >
                      <FileText size={20} />
                      <strong>[S1] {example.title}</strong>
                      <span>
                        Lihat rujukan <ArrowRight size={13} />
                      </span>
                    </button>
                  </div>
                </motion.div>
              </motion.div>
            </AnimatePresence>
          </div>
          <form className="preview-composer" onSubmit={openChat}>
            <label className="sr-only" htmlFor="preview-question">
              Pertanyaan untuk AMANA
            </label>
            <input
              id="preview-question"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Tanyakan sesuatu kepada AMANA…"
              maxLength={2000}
            />
            <button aria-label="Buka percakapan dengan pertanyaan ini">
              <Send size={19} />
            </button>
          </form>
        </div>

        <AnimatePresence initial={false}>
          {sourceOpen && (
            <motion.aside
              id="preview-source"
              className="preview-source"
              aria-label="Sumber referensi contoh"
              initial={{ opacity: 0, x: reduced ? 0 : 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{
                duration: reduced ? 0 : 0.25,
                ease: [0.22, 1, 0.36, 1],
              }}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  setSourceOpen(false);
                  document.getElementById("preview-source-toggle")?.focus();
                }
              }}
            >
              <div className="preview-source-header">
                <strong>Sumber referensi</strong>
                <button
                  className="site-icon-button"
                  aria-label="Tutup sumber referensi"
                  onClick={() => {
                    setSourceOpen(false);
                    document.getElementById("preview-source-toggle")?.focus();
                  }}
                >
                  <X size={17} />
                </button>
              </div>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={example.id}
                  className="preview-source-content"
                  initial={{ opacity: 0, y: reduced ? 0 : 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: reduced ? 0 : 0.16 }}
                >
                  <div className="source-title-row">
                    <span className="source-file-icon">
                      <FileText size={22} />
                    </span>
                    <div>
                      <span>Fatwa DSN-MUI</span>
                      <strong>[S1] {example.title}</strong>
                    </div>
                  </div>
                  <p>{example.description}</p>
                  <a
                    href={example.href}
                    className="source-document-link"
                    target="_blank"
                    rel="noreferrer"
                  >
                    <ExternalLink size={16} /> Buka dokumen sumber{" "}
                    <ArrowRight size={15} />
                  </a>
                </motion.div>
              </AnimatePresence>
            </motion.aside>
          )}
        </AnimatePresence>
      </div>
      <p className="preview-caption">
        <BookOpen size={12} /> Contoh tampilan prototipe
      </p>
    </div>
  );
}
