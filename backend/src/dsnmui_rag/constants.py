STOPWORDS_ID = set("""
yang dan atau untuk pada dalam dengan dari ke di adalah itu ini sebagai menurut tentang apa saja
jelaskan sebutkan tampilkan berikan apakah bagaimana mengapa cari dokumen bagian hasil ocr teks arab
hadis hadits ayat quran qur'an al qur an fatwa dsn mui no nomor berdasarkan prinsip syariah sebuah
antara serta dapat digunakan isi pokok dasar hukum saya kami mau ingin punya usaha umkm produk bisnis
secara perlu jika oleh para pihak bagi terhadap mereka bisa lebih tidak tanpa menggunakan
""".split())

IMPORTANT_TERMS = {
    "akad", "ijarah", "wakalah", "ujrah", "ju'alah", "jualah", "pemasaran", "bpws",
    "hotel", "wisatawan", "pemandu", "destinasi", "ketentuan", "prinsip", "larangan",
    "dalil", "hadis", "ayat", "kaidah", "definisi", "dimaksud", "pariwisata", "syariah",
}

STAT_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "dalam", "atau", "agar", "apa", "bagaimana",
    "saya", "kami", "mau", "ingin", "punya", "usaha", "umkm", "produk", "bisnis", "secara", "menurut", "berdasarkan",
    "apakah", "boleh", "harus", "perlu", "jika", "itu", "ini", "ada", "sebagai", "antara", "oleh", "para", "pihak", "bagi",
    "sebutkan", "jelaskan", "terhadap", "mereka", "dapat", "bisa", "lebih", "tidak", "tanpa", "menggunakan",
}

STAT_CONCEPT_SYNONYMS = {
    "jual beli": ["jual beli", "jual", "beli", "ba’i", "al-ba’i", "akad jual beli", "transaksi jual beli"],
    "marketplace": ["marketplace", "platform", "lokapasar", "e-commerce", "online", "toko online", "pedagang online"],
    "salam": ["salam", "jual beli salam", "pemesanan", "pre-order", "pre order", "pesanan"],
    "istishna": ["istishna", "istishna'", "istisna", "pembuatan barang", "pesanan pembuatan"],
    "murabahah": ["murabahah", "margin", "jual beli dengan margin", "pembiayaan pembelian"],
    "ijarah": ["ijarah", "sewa", "jasa", "upah", "ujrah", "fee"],
    "wakalah": ["wakalah", "wakalah bil ujrah", "kuasa", "wakil", "perwakilan", "komisi"],
    "ju'alah": ["ju'alah", "jualah", "ju’alah", "imbalan keberhasilan", "success fee"],
    "mudharabah": ["mudharabah", "bagi hasil", "mudharib", "shahibul maal"],
    "musyarakah": ["musyarakah", "syirkah", "kerja sama modal", "patungan modal"],
    "qardh": ["qardh", "pinjaman", "dana talangan"],
    "rahn": ["rahn", "gadai", "jaminan"],
    "kafalah": ["kafalah", "penjaminan", "jaminan"],
    "hawalah": ["hawalah", "pengalihan utang", "alih piutang"],
    "riba": ["riba", "bunga", "tambahan pinjaman", "ribawi"],
    "gharar": ["gharar", "ketidakjelasan", "tidak jelas", "spekulatif"],
    "maysir": ["maysir", "maisir", "judi", "perjudian", "gambling"],
    "tadlis": ["tadlis", "penipuan", "menyembunyikan cacat", "informasi palsu"],
    "najasy": ["najasy", "tanajusy", "fake order", "review palsu", "penawaran palsu"],
    "risywah": ["risywah", "suap", "gratifikasi"],
    "voucher": ["voucher", "kupon", "cashback", "poin", "loyalty", "diskon"],
    "uang elektronik": ["uang elektronik", "e-money", "dompet digital", "wallet", "saldo elektronik", "qr", "qris"],
    "fintech": ["fintech", "layanan pembiayaan berbasis teknologi", "platform pembiayaan", "paylater"],
    "asuransi": ["asuransi", "takaful", "tabarru", "dana tabarru", "polis"],
    "wisata": ["wisata", "pariwisata", "travel", "hotel", "destinasi", "tour"],
    "halal": ["halal", "sertifikasi halal", "produk halal", "makanan halal"],
    "sukuk": ["sukuk", "obligasi syariah", "surat berharga syariah"],
    "reksadana": ["reksadana", "reksa dana", "investasi kolektif"],
}

TITLE_BOOST_TERMS = [
    "marketplace", "jual beli", "akad jual beli", "salam", "istishna", "murabahah", "wakalah", "ijarah", "ju'alah",
    "uang elektronik", "voucher", "fintech", "pembiayaan", "rahn", "qardh", "kafalah", "asuransi", "pariwisata",
    "hotel", "pedoman implementasi", "ta'limat", "pasar modal", "sukuk", "reksa dana",
]

UNSUPPORTED_QUERY_TERMS = {
    "ai", "chatbot", "artificial intelligence", "kecerdasan buatan",
    "nft", "staking", "kripto", "crypto", "token", "blockchain", "game berhadiah",
}

CORE_CONCEPT_TERMS = {
    "jual beli", "marketplace", "akad", "murabahah", "salam", "istishna", "istishna’",
    "ijarah", "wakalah", "wakalah bil ujrah", "ujrah", "jualah", "ju’alah", "ju'alah",
    "mudharabah", "musyarakah", "qardh", "rahn", "kafalah", "hawalah",
    "riba", "gharar", "maysir", "maisir", "tadlis", "najasy", "risywah",
    "uang elektronik", "e-money", "fintech", "asuransi", "tabarru", "takaful",
    "voucher", "cashback", "diskon", "hadiah", "pariwisata", "hotel", "wisata",
    "halal", "haram", "objek", "barang", "jasa", "pembiayaan", "modal usaha",
}

DALIL_KEYWORDS = [
    "dalil", "hadis", "hadits", "ayat", "q.s", "quran", "qur'an", "al-qur", "firman", "riwayat",
    "rasulullah", "nabi", "kaidah", "fikih", "fiqh", "rواه", "قال", "رسول", "الله", "الأصل", "لا ضرر",
]

UMKM_DOMAIN_TERMS = set().union(*[set(v) for v in STAT_CONCEPT_SYNONYMS.values()]) | CORE_CONCEPT_TERMS

V10_BAD_TERMS = [
    "chunk_id", "reranker_score", "pdf_path", "assistant\n", "user\n",
]
