import { Link } from "react-router-dom";

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <Link
      to="/"
      className={`site-brand${compact ? " site-brand--compact" : ""}`}
      aria-label="AMANA — beranda"
    >
      <span className="site-brand-mark" aria-hidden="true">
        A
      </span>
      <span>AMANA</span>
    </Link>
  );
}
