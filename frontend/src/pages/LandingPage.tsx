import { useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  ExternalLink,
  FileText,
  MessageCircle,
  Plus,
  Search,
} from "lucide-react";
import { SiteHeader } from "@/components/site/SiteHeader";
import { SiteFooter } from "@/components/site/SiteFooter";
import { Reveal } from "@/components/site/Reveal";
import { ProductPreview } from "@/components/site/ProductPreview";
import { Artwork } from "@/components/site/Artwork";
import { chatLink, EXAMPLES, LINKS } from "@/lib/site";

export default function LandingPage() {
  const [expanded, setExpanded] = useState<number | null>(null);
  const reduced = useReducedMotion();
  const entrance = (delay: number) => ({
    initial: reduced ? (false as const) : { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0 },
    transition: {
      duration: reduced ? 0 : 0.55,
      delay: reduced ? 0 : delay,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  });
  return (
    <div className="landing-page">
      <SiteHeader />
      <main id="main-content" tabIndex={-1}>
        <section
          className="hero-section site-container"
          aria-labelledby="hero-title"
        >
          <div className="hero-copy">
            <motion.p className="hero-eyebrow" {...entrance(0)}>
              Asisten DSN-MUI untuk UMKM
            </motion.p>
            <motion.h1 id="hero-title" {...entrance(0.06)}>
              Sebelum menentukan akad,
              <br />
              <span>pahami rujukannya.</span>
            </motion.h1>
            <motion.p className="hero-description" {...entrance(0.12)}>
              Tanyakan kebutuhan usaha Anda dan telusuri fatwa DSN-MUI
              <br className="desktop-break" /> yang relevan, lengkap dengan
              referensi dokumennya.
            </motion.p>
            <motion.div className="hero-actions" {...entrance(0.18)}>
              <Link to="/chat" className="site-button">
                Mulai bertanya
              </Link>
              <Link
                to="/#cara-kerja"
                className="site-button site-button--outline"
              >
                Lihat cara kerja <ArrowRight size={18} />
              </Link>
            </motion.div>
            <div className="hero-art-pair" aria-hidden="true">
              <motion.div className="hero-merchant-art" {...entrance(0.22)}>
                <Artwork name="merchant" priority />
              </motion.div>
              <motion.div className="hero-books-art" {...entrance(0.28)}>
                <Artwork name="books" priority />
              </motion.div>
            </div>
          </div>
          <motion.div {...entrance(0.24)}>
            <ProductPreview />
          </motion.div>
          <Reveal className="document-strip">
            <div className="document-strip-heading">
              <span />
              Menelusuri dokumen DSN-MUI
              <span />
            </div>
            <div className="document-types">
              <a href={LINKS.fatwa} target="_blank" rel="noreferrer">
                <FileText />
                Fatwa
                <ArrowUpRight className="document-type-arrow" />
              </a>
              <a href={LINKS.pedoman} target="_blank" rel="noreferrer">
                <BookOpen />
                Pedoman implementasi
                <ArrowUpRight className="document-type-arrow" />
              </a>
              <a href={LINKS.talimat} target="_blank" rel="noreferrer">
                <FileText />
                Ta’limat
                <ArrowUpRight className="document-type-arrow" />
              </a>
            </div>
          </Reveal>
        </section>

        <section
          className="how-section"
          id="cara-kerja"
          aria-labelledby="how-title"
        >
          <div className="site-container how-grid">
            <Reveal className="source-showcase">
              <Artwork name="sources" className="source-showcase-art" />
              <div className="source-showcase-card">
                <div className="source-title-row">
                  <span className="source-file-icon">
                    <FileText size={25} />
                  </span>
                  <strong>[S1] Jual Beli Istishna’</strong>
                </div>
                <div className="source-showcase-copy">
                  <span>Rujukan dokumen</span>
                  <p>
                    Telusuri dokumen DSN-MUI tentang jual beli istishna’ dan
                    baca kembali ketentuannya pada sumber asli.
                  </p>
                </div>
                <a href={LINKS.istishna} target="_blank" rel="noreferrer">
                  <ExternalLink size={17} /> Buka dokumen sumber{" "}
                  <ArrowUpRight size={16} />
                </a>
              </div>
            </Reveal>
            <Reveal className="how-copy" delay={0.07}>
              <h2 id="how-title">
                Pahami jawabannya.
                <br />
                Periksa rujukannya.
              </h2>
              <p>
                Setiap penelusuran membantu Anda menemukan dokumen yang relevan
                untuk dibaca kembali.
              </p>
              <ul className="benefit-list">
                <li>
                  <span>
                    <MessageCircle size={22} />
                  </span>
                  <div>
                    <h3>Pertanyaan dalam bahasa sehari-hari</h3>
                    <p>Sampaikan pertanyaan sesuai kebutuhan usaha Anda.</p>
                  </div>
                </li>
                <li>
                  <span>
                    <FileText size={22} />
                  </span>
                  <div>
                    <h3>Referensi dokumen yang terlihat jelas</h3>
                    <p>
                      Temukan rujukan fatwa, pedoman implementasi, atau
                      ta’limat.
                    </p>
                  </div>
                </li>
                <li>
                  <span>
                    <Search size={22} />
                  </span>
                  <div>
                    <h3>Konteks sumber yang bisa diperiksa</h3>
                    <p>
                      Baca dokumen aslinya untuk memahami ketentuan secara utuh.
                    </p>
                  </div>
                </li>
              </ul>
            </Reveal>
          </div>
        </section>

        <section
          className="questions-section site-container"
          id="contoh-pertanyaan"
          aria-labelledby="questions-title"
        >
          <Reveal className="questions-grid">
            <div className="questions-intro">
              <h2 id="questions-title">
                Mulai dari pertanyaan
                <br />
                usaha Anda.
              </h2>
              <p>
                Pilih contoh pertanyaan untuk melihat bagaimana AMANA membantu
                menelusuri rujukan yang relevan.
              </p>
            </div>
            <div className="question-list">
              {EXAMPLES.map((example, index) => (
                <div
                  className={`question-item${expanded === index ? " is-open" : ""}`}
                  key={example.id}
                >
                  <h3>
                    <button
                      aria-expanded={expanded === index}
                      aria-controls={`question-answer-${example.id}`}
                      id={`question-trigger-${example.id}`}
                      onClick={() =>
                        setExpanded(expanded === index ? null : index)
                      }
                    >
                      <span className="question-number">0{index + 1}</span>
                      <Artwork
                        name={
                          (["parcels", "partnership", "sources"] as const)[
                            index
                          ]
                        }
                        className="question-art"
                      />
                      <span>{example.accordion}</span>
                      <Plus size={18} aria-hidden="true" />
                    </button>
                  </h3>
                  <AnimatePresence initial={false}>
                    {expanded === index && (
                      <motion.div
                        id={`question-answer-${example.id}`}
                        role="region"
                        aria-labelledby={`question-trigger-${example.id}`}
                        className="question-answer"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{
                          duration: reduced ? 0 : 0.23,
                          ease: [0.22, 1, 0.36, 1],
                        }}
                      >
                        <div>
                          <p>{example.explanation}</p>
                          <Link to={chatLink(example.question)}>
                            Gunakan pertanyaan ini <ArrowRight size={16} />
                          </Link>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ))}
            </div>
          </Reveal>
        </section>

        <Reveal className="site-container closing-container">
          <section className="closing-banner" aria-label="Mulai percakapan">
            <div>
              <h2>Punya pertanyaan tentang akad usaha?</h2>
              <p>
                Gunakan AMANA untuk menelusuri fatwa DSN-MUI dengan lebih mudah.
              </p>
            </div>
            <Artwork name="plant" className="closing-plant-art" />
            <Link to="/chat" className="site-button site-button--light">
              Mulai bertanya <ArrowRight size={18} />
            </Link>
          </section>
        </Reveal>
      </main>
      <SiteFooter />
    </div>
  );
}
