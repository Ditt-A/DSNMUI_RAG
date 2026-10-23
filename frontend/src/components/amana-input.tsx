import { useState, useRef, useEffect } from "react";
import TextareaAutosize from "react-textarea-autosize";
import { SendHorizonal } from "lucide-react";
import { Button } from "@/components/ui/button";

interface AmanaInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function AmanaInput({ onSend, disabled }: AmanaInputProps) {
  const [content, setContent] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (content.trim() && !disabled) {
      onSend(content.trim());
      setContent("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (!disabled && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [disabled]);

  return (
    <div className="relative flex items-center w-full gap-2 bg-white rounded-xl shadow-[0_2px_12px_rgba(0,0,0,0.04)] border p-2 focus-within:ring-1 focus-within:ring-primary/20 focus-within:border-primary/30 transition-all">
      <TextareaAutosize
        ref={textareaRef}
        minRows={1}
        maxRows={6}
        placeholder="Tanyakan sesuatu kepada AMANA..."
        value={content}
        onChange={(e) => setContent(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className="w-full resize-none bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground disabled:opacity-50"
      />
      <div className="shrink-0 pb-1 pr-1">
        <Button
          size="icon"
          onClick={handleSend}
          disabled={!content.trim() || disabled}
          className="h-10 w-10 shrink-0 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground transition-transform active:scale-95 disabled:scale-100 disabled:opacity-50"
        >
          <SendHorizonal className="w-5 h-5 ml-0.5" />
          <span className="sr-only">Kirim pesan</span>
        </Button>
      </div>
    </div>
  );
}
