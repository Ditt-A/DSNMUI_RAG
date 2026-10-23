import { useState, useCallback } from "react";
import { ChatMessage, ChatFilters, RagResponse } from "../types/rag";
import { toast } from "sonner";

export function useAmanaChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
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
      if (!content.trim()) return;

      const userMessageId = Date.now().toString();
      const assistantMessageId = (Date.now() + 1).toString();

      // Add user message and a loading assistant message
      setMessages((prev) => [...prev, { id: userMessageId, role: "user", content }, { id: assistantMessageId, role: "assistant", content: "", isLoading: true }]);

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
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
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
        console.error("API Error:", error);
        toast.error("Gagal menghubungi layanan AMANA", {
          action: {
            label: "Coba Lagi",
            onClick: () => sendMessage(content),
          },
        });

        setMessages((prev) => prev.filter((msg) => msg.id !== assistantMessageId));
      } finally {
        setIsLoading(false);
      }
    },
    [filters],
  );

  const clearChat = () => setMessages([]);

  return {
    messages,
    isLoading,
    filters,
    updateFilter,
    sendMessage,
    clearChat,
  };
}
