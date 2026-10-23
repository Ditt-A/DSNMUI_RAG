import { useEffect, useRef } from "react";
import { ChatMessage } from "@/types/rag";
import { AmanaMessage } from "./amana-message";
import { motion } from "motion/react";
import { Artwork } from "@/components/site/Artwork";

interface AmanaChatProps {
  messages: ChatMessage[];
  isLoading: boolean;
}

export function LoadingDots() {
  return (
    <div className="flex space-x-1.5 items-center h-5">
      <motion.div
        className="w-1.5 h-1.5 bg-primary/40 rounded-full"
        animate={{ y: [0, -4, 0] }}
        transition={{ duration: 0.6, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="w-1.5 h-1.5 bg-primary/60 rounded-full"
        animate={{ y: [0, -4, 0] }}
        transition={{
          duration: 0.6,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.15,
        }}
      />
      <motion.div
        className="w-1.5 h-1.5 bg-primary/80 rounded-full"
        animate={{ y: [0, -4, 0] }}
        transition={{
          duration: 0.6,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.3,
        }}
      />
    </div>
  );
}

export function AmanaChat({ messages, isLoading }: AmanaChatProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, isLoading]);

  return (
    <div
      className="flex-1 w-full overflow-y-auto px-4 md:px-8 py-6 scroll-smooth"
      ref={scrollRef}
    >
      <div className="max-w-4xl mx-auto w-full flex flex-col min-h-full">
        {messages.length === 0 ? (
          <div className="chat-empty-state m-auto flex flex-col items-center justify-center text-center mb-20">
            <Artwork name="chat" className="chat-empty-art" priority />
            <h2 className="text-xl font-heading font-bold text-foreground mb-2">
              AMANA
            </h2>
            <p className="text-sm text-muted-foreground max-w-sm">
              Asisten khusus konsultasi DSN-MUI untuk UMKM. Tanyakan seputar
              fatwa, talimat, dan pedoman implementasi.
            </p>
          </div>
        ) : (
          <div className="flex flex-col w-full pb-8">
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <AmanaMessage message={msg} />
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
