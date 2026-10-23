import { lazy, Suspense, useEffect } from "react";
import {
  BrowserRouter,
  Link,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import {
  AnimatePresence,
  MotionConfig,
  motion,
  useReducedMotion,
} from "motion/react";
import { Toaster } from "@/components/ui/sonner";
import LandingPage from "@/pages/LandingPage";
import InfoPage from "@/pages/InfoPage";
import { SiteHeader } from "@/components/site/SiteHeader";
import { SiteFooter } from "@/components/site/SiteFooter";
import "@/styles/site.css";

const ChatPage = lazy(() => import("@/pages/ChatPage"));
const titles: Record<string, string> = {
  "/": "AMANA — Asisten DSN-MUI untuk UMKM",
  "/chat": "Percakapan — AMANA",
  "/about": "Tentang AMANA",
  "/privacy": "Privasi — AMANA",
  "/terms": "Ketentuan penggunaan — AMANA",
};

function PageEffects({ pathname, hash }: { pathname: string; hash: string }) {
  const reduced = useReducedMotion();
  useEffect(() => {
    document.title = titles[pathname] || "Halaman tidak ditemukan — AMANA";
    const frame = requestAnimationFrame(() => {
      if (hash)
        document
          .getElementById(decodeURIComponent(hash.slice(1)))
          ?.scrollIntoView({
            behavior: reduced ? "instant" : "smooth",
            block: "start",
          });
      else window.scrollTo({ top: 0, behavior: "instant" });
    });
    return () => cancelAnimationFrame(frame);
  }, [pathname, hash, reduced]);
  return null;
}

function AppRoutes() {
  const location = useLocation();
  const reduced = useReducedMotion();
  return (
    <>
      <a className="skip-link" href="#main-content">
        Langsung ke konten
      </a>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: reduced ? 0 : 5 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reduced ? 0 : 0.18 }}
        >
          <PageEffects pathname={location.pathname} hash={location.hash} />
          <Suspense
            fallback={
              <main
                id="main-content"
                className="route-loading"
                aria-busy="true"
              >
                Membuka AMANA…
              </main>
            }
          >
            <Routes location={location}>
              <Route path="/" element={<LandingPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/about" element={<InfoPage kind="about" />} />
              <Route path="/privacy" element={<InfoPage kind="privacy" />} />
              <Route path="/terms" element={<InfoPage kind="terms" />} />
              <Route
                path="*"
                element={
                  <>
                    <SiteHeader />
                    <main
                      id="main-content"
                      className="site-container not-found"
                    >
                      <span>404</span>
                      <h1>Halaman ini belum ditemukan.</h1>
                      <p>
                        Anda bisa kembali ke beranda atau mulai percakapan baru.
                      </p>
                      <div className="hero-actions">
                        <Link to="/" className="site-button">
                          Kembali ke beranda
                        </Link>
                        <Link
                          to="/chat"
                          className="site-button site-button--outline"
                        >
                          Buka AMANA
                        </Link>
                      </div>
                    </main>
                    <SiteFooter />
                  </>
                }
              />
            </Routes>
          </Suspense>
        </motion.div>
      </AnimatePresence>
      <Toaster position="top-right" richColors closeButton />
    </>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <MotionConfig reducedMotion="user">
        <AppRoutes />
      </MotionConfig>
    </BrowserRouter>
  );
}
