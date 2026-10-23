import { ArrowLeft, Menu } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

interface AmanaHeaderProps {
  onMenuClick: () => void;
}

export function AmanaHeader({ onMenuClick }: AmanaHeaderProps) {
  return (
    <header className="flex items-center justify-between px-4 h-16 border-b bg-background shadow-xs z-10 shrink-0">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={onMenuClick}
          aria-label="Buka menu percakapan"
        >
          <Menu className="w-5 h-5 text-muted-foreground" />
        </Button>
        <div className="flex items-center gap-3">
          <img
            src="/brand/amana-mark.svg"
            className="w-10 h-10"
            width="64"
            height="64"
            alt=""
          />
          <div className="flex flex-col">
            <h1 className="font-heading font-bold text-lg leading-tight tracking-tight text-foreground">
              AMANA
            </h1>
            <span className="text-xs text-muted-foreground font-medium">
              Asisten DSN-MUI untuk UMKM
            </span>
          </div>
        </div>
      </div>
      <Link to="/" className="chat-back-home">
        <ArrowLeft size={16} /> Beranda
      </Link>
    </header>
  );
}
