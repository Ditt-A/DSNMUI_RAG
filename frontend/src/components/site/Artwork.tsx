import type { ImgHTMLAttributes } from "react";

const artwork = {
  merchant: {
    src: "/assets/visuals/01-gerobak-umkm.webp",
    width: 1536,
    height: 1024,
  },
  owner: {
    src: "/assets/visuals/02-pemilik-usaha.webp",
    width: 1536,
    height: 1024,
  },
  books: {
    src: "/assets/visuals/03-buku-rujukan.webp",
    width: 1536,
    height: 1024,
  },
  chat: {
    src: "/assets/visuals/04-asisten-percakapan.webp",
    width: 1448,
    height: 1086,
  },
  parcels: {
    src: "/assets/visuals/05-paket-produk.webp",
    width: 1448,
    height: 1086,
  },
  partnership: {
    src: "/assets/visuals/06-kerja-sama.webp",
    width: 1774,
    height: 887,
  },
  sources: {
    src: "/assets/visuals/07-telusuri-dokumen.webp",
    width: 1254,
    height: 1254,
  },
  plant: {
    src: "/assets/visuals/08-tanaman-dekoratif.webp",
    width: 1122,
    height: 1402,
  },
} as const;

type ArtworkProps = Omit<
  ImgHTMLAttributes<HTMLImageElement>,
  "src" | "width" | "height"
> & { name: keyof typeof artwork; priority?: boolean };

export function Artwork({
  name,
  alt = "",
  priority = false,
  ...props
}: ArtworkProps) {
  return (
    <img
      {...artwork[name]}
      alt={alt}
      loading={priority ? "eager" : "lazy"}
      decoding="async"
      {...props}
    />
  );
}
