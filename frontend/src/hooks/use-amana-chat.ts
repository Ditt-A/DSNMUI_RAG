import { useState, useCallback, useEffect, useRef } from "react";
import { ChatMessage, ChatFilters, RagResponse } from "../types/rag";
import { toast } from "sonner";

export function useAmanaChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const pendingRequest = useRef<AbortController | null>(null);
  useEffect(() => () => pendingRequest.current?.abort(), []);
  const [filters, setFilters] = useState<ChatFilters>({
    expectedFocus: "",
    categoryFilter: null,
    useVisualDalil: false,
    useVisualAsGenerator: false,
  });

  const updateFilter = (key: keyof ChatFilters, value: any) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim() || pendingRequest.current) return;
      const controller = new AbortController();
      pendingRequest.current = controller;

      const userMessageId = crypto.randomUUID();
      const assistantMessageId = crypto.randomUUID();

      // Add user message and a loading assistant message
      setMessages((prev) => [
        ...prev,
        { id: userMessageId, role: "user", content },
        {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          isLoading: true,
        },
      ]);

      setIsLoading(true);

      try {
        const payload = {
          query: content,
          expected_focus: filters.expectedFocus,
          category_filter: filters.categoryFilter,
          use_visual_dalil: filters.useVisualDalil,
          use_visual_as_generator: filters.useVisualAsGenerator,
        };

        const response = await fetch("/rag/ask", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(
            response.status === 503
              ? "Layanan jawaban belum tersedia. Silakan coba lagi nanti."
              : "Pesan belum berhasil diproses. Silakan coba lagi.",
          );
        }

        const data: RagResponse = await response.json();

        // Update the assistant message with the response
        setMessages((prev) =>
          prev.map((msg) => {
            if (msg.id === assistantMessageId) {
              return {
                ...msg,
                content: data.answer,
                isLoading: false,
                responseMeta: {
                  support_info: data.support_info,
                  generation_meta: data.generation_meta,
                  timing: data.timing,
                  sources: data.sources,
                  elapsed_seconds: data.elapsed_seconds,
                },
              };
            }
            return msg;
          }),
        );
      } catch (error) {
        if (controller.signal.aborted) return;
        toast.error(
          error instanceof Error && error.message !== "Failed to fetch"
            ? error.message
            : "Koneksi ke AMANA belum tersedia. Periksa koneksi Anda dan coba lagi.",
        );

        setMessages((prev) =>
          prev.filter((msg) => msg.id !== assistantMessageId),
        );
      } finally {
        if (pendingRequest.current === controller) {
          pendingRequest.current = null;
          setIsLoading(false);
        }
      }
    },
    [filters],
  );

  const clearChat = () => {
    pendingRequest.current?.abort();
    pendingRequest.current = null;
    setMessages([]);
    setIsLoading(false);
  };

  return {
    messages,
    isLoading,
    filters,
    updateFilter,
    sendMessage,
    clearChat,
  };
}
