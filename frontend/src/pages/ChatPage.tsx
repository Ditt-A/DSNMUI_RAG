import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useAmanaChat } from "@/hooks/use-amana-chat";
import { AmanaHeader } from "@/components/amana-header";
import { AmanaSidebar } from "@/components/amana-sidebar";
import { AmanaChat } from "@/components/amana-chat";
import { AmanaFilters } from "@/components/amana-filters";
import { AmanaInput } from "@/components/amana-input";

export default function ChatPage() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [composerVersion, setComposerVersion] = useState(0);
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const reduced = useReducedMotion();
  const initialQuestion = searchParams.get("question")?.slice(0, 2000) || "";
  const { messages, isLoading, filters, updateFilter, sendMessage, clearChat } =
    useAmanaChat();

  useEffect(() => {
    if (!sidebarOpen) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSidebarOpen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [sidebarOpen]);

  return (
    <div className="chat-page flex w-full bg-slate-50/30 overflow-hidden font-sans">
      <AmanaSidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={() => {
          clearChat();
          setSidebarOpen(false);
          setComposerVersion((version) => version + 1);
          navigate("/chat", { replace: true });
        }}
        onSettings={() => {
          setSettingsOpen(true);
          setSidebarOpen(false);
        }}
      />

      <div className="flex flex-col flex-1 min-w-0 h-full relative">
        <AmanaHeader onMenuClick={() => setSidebarOpen(true)} />

        <main
          id="main-content"
          tabIndex={-1}
          className="flex-1 flex flex-col relative overflow-hidden"
        >
          <AmanaChat messages={messages} isLoading={isLoading} />

          <div className="w-full px-4 md:px-8 pb-4 pt-2 bg-gradient-to-t from-background via-background/95 to-transparent z-10">
            <div className="max-w-4xl mx-auto w-full flex flex-col gap-2">
              <div>
                <button
                  className="chat-filter-toggle"
                  aria-expanded={settingsOpen}
                  aria-controls="chat-filter-panel"
                  onClick={() => setSettingsOpen(!settingsOpen)}
                >
                  <SlidersHorizontal size={13} /> Filter pertanyaan{" "}
                  <ChevronDown size={14} />
                </button>
                <AnimatePresence initial={false}>
                  {settingsOpen && (
                    <motion.div
                      id="chat-filter-panel"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: reduced ? 0 : 0.2 }}
                      className="overflow-hidden"
                    >
                      <AmanaFilters
                        filters={filters}
                        onChange={updateFilter}
                        disabled={isLoading}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              <AmanaInput
                key={composerVersion}
                initialValue={initialQuestion}
                onSend={sendMessage}
                disabled={isLoading}
              />
              <div className="text-center mt-2">
                <span className="text-[10px] text-muted-foreground/60">
                  AMANA dapat membuat kesalahan. Harap periksa kembali informasi
                  penting referensi DSN-MUI.
                </span>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
