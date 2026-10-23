export const LINKS = {
  github: "https://github.com/Ditt-A/DSNMUI_RAG",
  docs: "https://github.com/Ditt-A/DSNMUI_RAG/tree/main/docs",
  issues: "https://github.com/Ditt-A/DSNMUI_RAG/issues",
  fatwa: "https://dsnmui.or.id/kategori/fatwa/",
  istishna: "https://dsnmui.or.id/kategori/fatwa/page/17/",
  pedoman: "https://dsnmui.or.id/produk/pedoman-implementasi/",
  talimat: "https://dsnmui.or.id/produk/talimat/",
};

export const EXAMPLES = [
  {
    id: "preorder",
    tab: "Pre-order",
    question:
      "Saya menjual produk pre-order. Dokumen apa yang perlu saya pahami?",
    title: "Jual Beli Istishna’",
    description:
      "Telusuri dokumen tentang jual beli istishna’ untuk membaca ketentuan dan konteks akad pada sumber aslinya.",
    href: LINKS.istishna,
    accordion: "Apa yang perlu dipahami tentang pre-order?",
    explanation:
      "Ceritakan barang yang dijual, cara pemesanan, dan waktu penyerahannya. Gunakan konteks tersebut untuk menelusuri dokumen yang relevan.",
  },
  {
    id: "bagihasil",
    tab: "Bagi hasil",
    question:
      "Saya ingin bekerja sama dengan sistem bagi hasil. Rujukan apa yang bisa saya pelajari?",
    title: "Rujukan akad bagi hasil",
    description:
      "Mulai dari direktori fatwa DSN-MUI untuk menelusuri dokumen mengenai kerja sama usaha dan akad bagi hasil.",
    href: LINKS.fatwa,
    accordion: "Bagaimana menelusuri ketentuan bagi hasil?",
    explanation:
      "Jelaskan peran masing-masing pihak, bentuk kontribusi, dan rencana pembagian hasil. AMANA membantu menelusuri rujukan sesuai konteks yang Anda berikan.",
  },
  {
    id: "pembayaran",
    tab: "Pembayaran bertahap",
    question:
      "Pembeli ingin membayar secara bertahap. Dokumen apa yang perlu saya telusuri?",
    title: "Rujukan ketentuan pembayaran",
    description:
      "Baca kembali dokumen fatwa yang berkaitan dengan akad dan ketentuan pembayaran pada situasi usaha Anda.",
    href: LINKS.fatwa,
    accordion: "Di mana menemukan pedoman akad usaha?",
    explanation:
      "Anda dapat menelusuri fatwa, pedoman implementasi, dan ta’limat. Pilih kategori dokumen dalam percakapan untuk memperjelas pencarian.",
  },
] as const;

export const chatLink = (question: string) =>
  `/chat?question=${encodeURIComponent(question)}`;
