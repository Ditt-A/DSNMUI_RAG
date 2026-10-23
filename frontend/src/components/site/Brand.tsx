import { Link } from "react-router-dom";

export function Brand({
  compact = false,
  tagline = false,
}: {
  compact?: boolean;
  tagline?: boolean;
}) {
  return (
    <Link
      to="/"
      className={`site-brand${compact ? " site-brand--compact" : ""}`}
      aria-label="AMANA — beranda"
    >
      <span className="site-brand-mark" aria-hidden="true">
        <img src="/brand/amana-mark.svg" width="64" height="64" alt="" />
      </span>
      <span className="site-brand-wordmark">
        AMANA{tagline && <small>Rujukan syariah untuk UMKM</small>}
      </span>
    </Link>
  );
}
