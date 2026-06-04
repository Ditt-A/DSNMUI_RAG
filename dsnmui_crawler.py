#!/usr/bin/env python3
"""
Crawler PDF DSN-MUI untuk kebutuhan pre-computed file RAG.

Target default:
1. Fatwa              : https://dsnmui.or.id/kategori/fatwa/
2. Pedoman Implementasi: https://dsnmui.or.id/produk/pedoman-implementasi/
3. Ta'limat          : https://dsnmui.or.id/produk/talimat/

Output:
- Folder PDF per kategori: <out_dir>/<category>/*.pdf
- Manifest JSONL        : <out_dir>/manifest.jsonl
- Manifest CSV          : <out_dir>/manifest.csv
- Error log JSONL       : <out_dir>/errors.jsonl

Contoh pakai:
    pip install requests beautifulsoup4 gdown
    python dsnmui_crawler.py --out data_dsnmui --delay 1.0

Dry-run tanpa download:
    python dsnmui_crawler.py --dry-run --out data_dsnmui

Crawl hanya fatwa:
    python dsnmui_crawler.py --categories fatwa --out data_dsnmui
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests import Response, Session


DEFAULT_TARGETS = {
    "fatwa": "https://dsnmui.or.id/kategori/fatwa/",
    "pedoman_implementasi": "https://dsnmui.or.id/produk/pedoman-implementasi/",
    "talimat": "https://dsnmui.or.id/produk/talimat/",
}

PDF_ICON_HINTS = (
    "icon_pdf",
    "icon-pdf",
    "pdf-icon",
    "wp-content/uploads/2017/06/icon_pdf.png",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class DocumentItem:
    category: str
    nomor: str
    title: str
    page_url: str
    file_url: str
    direct_download_url: str
    source_type: str
    file_id: str
    resourcekey: str = ""
    local_path: str = ""
    status: str = "discovered"
    size_bytes: int = 0
    sha256: str = ""
    content_type: str = ""
    error: str = ""


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def slugify(text: str, max_len: int = 120) -> str:
    """Slug sederhana tanpa dependency tambahan."""
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "untitled")


def safe_filename(category: str, nomor: str, title: str, file_id: str = "") -> str:
    nomor_slug = slugify(nomor, 60)
    title_slug = slugify(title, 120)
    suffix = f"-{slugify(file_id, 30)}" if file_id else ""
    return f"{slugify(category, 40)}__{nomor_slug}__{title_slug}{suffix}.pdf"


def is_probably_doc_url(raw_url: str) -> bool:
    if not raw_url:
        return False

    value = raw_url.strip().lower()
    if any(hint in value for hint in PDF_ICON_HINTS):
        return False
    if value.startswith(("#", "mailto:", "tel:", "javascript:")):
        return False

    parsed = urlparse(value)
    path = unquote(parsed.path.lower())

    if "drive.google.com" in parsed.netloc:
        return True
    if "docs.google.com" in parsed.netloc:
        return True
    if path.endswith(".pdf"):
        return True
    if ".pdf" in value:
        return True

    return False


def normalize_url(raw_url: str, base_url: str) -> str:
    raw_url = (raw_url or "").strip()
    raw_url = raw_url.replace("&amp;", "&")
    return urljoin(base_url, raw_url)


def extract_drive_file_id(url: str) -> str:
    """Ambil file id dari beberapa format Google Drive."""
    if not url:
        return ""

    decoded = unquote(url)

    patterns = [
        r"drive\.google\.com/file/d/([^/]+)",
        r"drive\.google\.com/open\?id=([^&]+)",
        r"drive\.google\.com/uc\?[^#]*\bid=([^&]+)",
        r"docs\.google\.com/(?:document|spreadsheets|presentation)/d/([^/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, decoded)
        if match:
            return match.group(1)

    parsed = urlparse(decoded)
    query_id = parse_qs(parsed.query).get("id")
    if query_id:
        return query_id[0]
    return ""


def extract_drive_resourcekey(url: str) -> str:
    """Ambil resourcekey Google Drive jika ada.

    Banyak file lama Google Drive memakai URL seperti:
    https://drive.google.com/file/d/<id>/view?resourcekey=0-xxxxx&usp=sharing

    Kalau resourcekey dibuang saat mengubah URL menjadi uc?export=download,
    Google Drive sering mengembalikan halaman HTML, bukan PDF.
    """
    if not url:
        return ""

    decoded = unquote(url).replace("&amp;", "&")
    parsed = urlparse(decoded)
    resourcekey = parse_qs(parsed.query).get("resourcekey", [""])[0]
    if resourcekey:
        return resourcekey

    match = re.search(r"[?&]resourcekey=([^&#]+)", decoded)
    return match.group(1) if match else ""


def is_google_drive_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return (
        "drive.google.com" in host
        or "docs.google.com" in host
        or "drive.usercontent.google.com" in host
    )


def build_drive_download_url(
    file_id: str,
    resourcekey: str = "",
    confirm: str = "",
    usercontent: bool = False,
) -> str:
    """Buat URL download Google Drive sambil mempertahankan resourcekey."""
    params = {"id": file_id, "export": "download"}
    if confirm:
        params["confirm"] = confirm
    if resourcekey:
        params["resourcekey"] = resourcekey

    if usercontent:
        return "https://drive.usercontent.google.com/download?" + urlencode(params)
    return "https://drive.google.com/uc?" + urlencode(params)


def extract_drive_params(url: str) -> tuple[str, str]:
    return extract_drive_file_id(url), extract_drive_resourcekey(url)


def to_direct_download_url(url: str) -> tuple[str, str, str, str]:
    """Return: direct_url, source_type, file_id, resourcekey."""
    file_id, resourcekey = extract_drive_params(url)

    if file_id and is_google_drive_url(url):
        direct = build_drive_download_url(file_id=file_id, resourcekey=resourcekey)
        return direct, "google_drive", file_id, resourcekey

    return url, "direct_pdf_or_url", "", ""


def get_session(timeout: int, retries: int) -> Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Retry manual sederhana dilakukan di fetch_url/download; session tetap dibuat terpisah
    # agar cookie Google Drive dapat tersimpan.
    session.request_timeout = timeout  # type: ignore[attr-defined]
    session.request_retries = retries  # type: ignore[attr-defined]
    return session


def fetch_url(session: Session, url: str, timeout: int, retries: int) -> Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Gagal fetch {url}: {last_exc}")


def collect_candidate_urls(tag, base_url: str) -> list[str]:
    """Ambil URL PDF/Drive dari href, data-*, onclick, dan atribut lain."""
    urls: list[str] = []

    for node in tag.find_all(True):
        for attr, value in node.attrs.items():
            values: Iterable[str]
            if isinstance(value, list):
                values = [str(v) for v in value]
            else:
                values = [str(value)]

            for raw in values:
                # Kadang URL PDF berada di onclick/data-attribute bersama teks lain.
                found_urls = re.findall(r"https?://[^\s'\"<>]+", raw)
                if not found_urls:
                    found_urls = [raw]

                for item in found_urls:
                    item = normalize_url(item, base_url)
                    if is_probably_doc_url(item):
                        urls.append(item)

    # Pertahankan urutan, hapus duplikat.
    seen = set()
    unique = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def parse_nomor_title_from_row(category: str, row) -> tuple[str, str]:
    td_texts = [clean_text(td.get_text(" ", strip=True)) for td in row.find_all("td")]
    td_texts = [t for t in td_texts if t]

    if not td_texts:
        return "", ""

    # Fatwa biasanya: Nomor Fatwa | Tentang | Baca/Unduh
    if category == "fatwa":
        nomor = td_texts[0] if len(td_texts) >= 1 else ""
        title = td_texts[1] if len(td_texts) >= 2 else ""
        return nomor, title

    # Pedoman/Ta'limat biasanya: No | Nomor Keputusan/Surat | Tentang | Baca/Unduh
    if len(td_texts) >= 3 and re.fullmatch(r"\d+", td_texts[0]):
        nomor = td_texts[1]
        title = td_texts[2]
        return nomor, title

    # Fallback umum.
    nomor = td_texts[0]
    title = td_texts[1] if len(td_texts) >= 2 else ""
    return nomor, title


def parse_documents_from_page(category: str, page_url: str, html: str) -> list[DocumentItem]:
    soup = BeautifulSoup(html, "html.parser")
    docs: list[DocumentItem] = []

    # Utamakan baris tabel karena metadata nomor dan judul paling rapi di sana.
    for row in soup.find_all("tr"):
        urls = collect_candidate_urls(row, page_url)
        if not urls:
            continue

        nomor, title = parse_nomor_title_from_row(category, row)
        row_text = clean_text(row.get_text(" ", strip=True))
        if not title and row_text:
            title = row_text

        for file_url in urls:
            direct_url, source_type, file_id, resourcekey = to_direct_download_url(file_url)
            docs.append(
                DocumentItem(
                    category=category,
                    nomor=nomor,
                    title=title,
                    page_url=page_url,
                    file_url=file_url,
                    direct_download_url=direct_url,
                    source_type=source_type,
                    file_id=file_id,
                    resourcekey=resourcekey,
                )
            )

    # Fallback: kalau tidak ada table row yang terbaca, ambil semua URL kandidat di halaman.
    if not docs:
        for file_url in collect_candidate_urls(soup, page_url):
            direct_url, source_type, file_id, resourcekey = to_direct_download_url(file_url)
            docs.append(
                DocumentItem(
                    category=category,
                    nomor="",
                    title="",
                    page_url=page_url,
                    file_url=file_url,
                    direct_download_url=direct_url,
                    source_type=source_type,
                    file_id=file_id,
                    resourcekey=resourcekey,
                )
            )

    return docs


def discover_page_urls(base_url: str, html: str) -> list[str]:
    """Cari URL pagination. Kalau tidak ada, return base_url saja."""
    soup = BeautifulSoup(html, "html.parser")
    parsed_base = urlparse(base_url)
    base_path = parsed_base.path.rstrip("/")

    page_numbers = {1}
    for a in soup.find_all("a", href=True):
        href = normalize_url(a["href"], base_url)
        parsed_href = urlparse(href)
        if parsed_href.netloc != parsed_base.netloc:
            continue
        if not parsed_href.path.startswith(base_path):
            continue
        match = re.search(r"/page/(\d+)/?", parsed_href.path)
        if match:
            page_numbers.add(int(match.group(1)))

    max_page = max(page_numbers)
    if max_page <= 1:
        return [base_url]

    urls = [base_url]
    for n in range(2, max_page + 1):
        urls.append(urljoin(base_url, f"page/{n}/"))
    return urls


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_google_drive_confirm_token(resp: Response) -> str:
    """Ambil token konfirmasi Google Drive jika file butuh konfirmasi download."""
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            return value

    # Fallback dari HTML Google Drive.
    text = resp.text if "text/html" in resp.headers.get("Content-Type", "") else ""
    if not text:
        return ""

    text = (
        text.replace("\\u003d", "=")
        .replace("\\u0026", "&")
        .replace("\\/", "/")
        .replace("&amp;", "&")
    )

    patterns = [
        r"confirm=([0-9A-Za-z_\-]+)",
        r"name=[\"']confirm[\"']\s+value=[\"']([^\"']+)[\"']",
        r"confirm=([^&\"']+)&",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def extract_google_drive_followup_urls(resp: Response, current_url: str) -> list[str]:
    """Cari URL lanjutan dari halaman HTML Google Drive.

    Google Drive bisa menampilkan halaman HTML, bukan file, terutama untuk:
    - file besar yang butuh confirm token;
    - file lama yang memakai resourcekey;
    - redirect baru ke drive.usercontent.google.com.

    Fungsi ini mencoba beberapa pola URL yang umum muncul di HTML Google Drive.
    """
    file_id, resourcekey = extract_drive_params(current_url)
    urls: list[str] = []

    token = get_google_drive_confirm_token(resp)
    if file_id and token:
        urls.append(build_drive_download_url(file_id, resourcekey, token, usercontent=False))
        urls.append(build_drive_download_url(file_id, resourcekey, token, usercontent=True))

    if file_id:
        # Fallback paling sering bekerja untuk public file yang sudah punya resourcekey.
        urls.append(build_drive_download_url(file_id, resourcekey, "t", usercontent=True))
        urls.append(build_drive_download_url(file_id, resourcekey, "t", usercontent=False))

    ctype = resp.headers.get("Content-Type", "")
    html = resp.text if "text/html" in ctype else ""
    if html:
        html = (
            html.replace("\\u003d", "=")
            .replace("\\u0026", "&")
            .replace("\\/", "/")
            .replace("&amp;", "&")
        )
        patterns = [
            r'href=["\']([^"\']*(?:drive\.usercontent\.google\.com/download|/uc\?)[^"\']*)["\']',
            r'"downloadUrl"\s*:\s*"([^"]+)"',
            r'action=["\']([^"\']*drive\.usercontent\.google\.com/download[^"\']*)["\']',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, html):
                candidate = match.group(1)
                candidate = candidate.replace("&amp;", "&")
                candidate = urljoin(current_url, candidate)
                if is_google_drive_url(candidate):
                    urls.append(candidate)

    seen = set()
    unique = []
    for item in urls:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def google_drive_gdown_download(url: str, out_path: Path) -> tuple[int, str]:
    """Fallback memakai gdown untuk Google Drive.

    gdown lebih tahan terhadap variasi halaman Google Drive.
    Install: pip install gdown
    """
    try:
        import gdown  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Fallback gdown tidak tersedia. Jalankan: pip install gdown"
        ) from exc

    file_id, resourcekey = extract_drive_params(url)
    if not file_id:
        raise RuntimeError("Tidak bisa menemukan file_id Google Drive untuk fallback gdown.")

    view_url = f"https://drive.google.com/file/d/{quote(file_id, safe='')}/view"
    if resourcekey:
        view_url += f"?resourcekey={quote(resourcekey, safe='')}"

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    result = gdown.download(url=view_url, output=str(tmp_path), quiet=False, fuzzy=True)
    if not result or not tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("gdown gagal menghasilkan file.")

    with tmp_path.open("rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"gdown menghasilkan konten bukan PDF. header={header!r}")

    tmp_path.replace(out_path)
    return out_path.stat().st_size, "application/pdf"


def stream_download(session: Session, url: str, out_path: Path, timeout: int, retries: int) -> tuple[int, str]:
    """Download umum. Return size_bytes, content_type."""
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
                resp.raise_for_status()

                ctype = resp.headers.get("Content-Type", "")

                # Google Drive public file kadang memberi halaman HTML konfirmasi dulu.
                if "text/html" in ctype and is_google_drive_url(url):
                    followup_urls = extract_google_drive_followup_urls(resp, url)
                    resp.close()

                    for followup_url in followup_urls:
                        try:
                            with session.get(
                                followup_url,
                                stream=True,
                                timeout=timeout,
                                allow_redirects=True,
                            ) as resp2:
                                resp2.raise_for_status()
                                ctype2 = resp2.headers.get("Content-Type", "")
                                if "text/html" in ctype2:
                                    # Masih HTML; coba kandidat berikutnya.
                                    continue
                                return write_response_to_file(resp2, out_path)
                        except Exception as inner_exc:  # noqa: BLE001
                            last_exc = inner_exc
                            continue

                    raise RuntimeError(
                        "Google Drive mengembalikan HTML, bukan file PDF. "
                        "Kemungkinan resourcekey/confirm token belum dipakai atau file tidak public."
                    )

                return write_response_to_file(resp, out_path)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))

    # Fallback terakhir khusus Google Drive.
    if is_google_drive_url(url):
        try:
            return google_drive_gdown_download(url, out_path)
        except Exception as gdown_exc:  # noqa: BLE001
            raise RuntimeError(
                f"Gagal download {url}: {last_exc}; fallback gdown juga gagal: {gdown_exc}"
            ) from gdown_exc

    raise RuntimeError(f"Gagal download {url}: {last_exc}")


def write_response_to_file(resp: Response, out_path: Path) -> tuple[int, str]:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    total = 0
    with tmp_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    if total == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("File kosong.")

    # Validasi ringan: file PDF biasanya dimulai dengan %PDF.
    with tmp_path.open("rb") as f:
        header = f.read(5)
    content_type = resp.headers.get("Content-Type", "")
    if header != b"%PDF-" and "pdf" not in content_type.lower():
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Konten bukan PDF. Content-Type={content_type!r}, header={header!r}")

    tmp_path.replace(out_path)
    return total, content_type


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dedupe_documents(docs: list[DocumentItem]) -> list[DocumentItem]:
    seen = set()
    unique = []
    for doc in docs:
        key = doc.file_id or doc.file_url
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def crawl(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = get_session(args.timeout, args.retries)
    all_docs: list[DocumentItem] = []
    errors: list[dict] = []

    targets = {k: DEFAULT_TARGETS[k] for k in args.categories}

    for category, base_url in targets.items():
        print(f"\n[CRAWL] {category}: {base_url}")
        try:
            first_resp = fetch_url(session, base_url, args.timeout, args.retries)
            page_urls = discover_page_urls(base_url, first_resp.text)

            if args.max_pages and args.max_pages > 0:
                page_urls = page_urls[: args.max_pages]

            print(f"  ditemukan {len(page_urls)} halaman untuk dicek")

            for idx, page_url in enumerate(page_urls, start=1):
                if idx == 1:
                    html = first_resp.text
                else:
                    html = fetch_url(session, page_url, args.timeout, args.retries).text

                docs = parse_documents_from_page(category, page_url, html)
                print(f"  - halaman {idx}/{len(page_urls)}: {len(docs)} kandidat file")
                all_docs.extend(docs)
                time.sleep(args.delay)

        except Exception as exc:  # noqa: BLE001
            err = {"category": category, "page_url": base_url, "error": str(exc)}
            print(f"  ERROR: {err['error']}", file=sys.stderr)
            errors.append(err)

    all_docs = dedupe_documents(all_docs)

    if args.max_files and args.max_files > 0:
        all_docs = all_docs[: args.max_files]

    print(f"\n[INFO] Total unik ditemukan: {len(all_docs)} file")

    for i, doc in enumerate(all_docs, start=1):
        filename = safe_filename(doc.category, doc.nomor, doc.title, doc.file_id)
        category_dir = out_dir / doc.category
        category_dir.mkdir(parents=True, exist_ok=True)
        out_path = category_dir / filename
        doc.local_path = str(out_path)

        if args.dry_run:
            doc.status = "dry_run"
            print(f"[DRY-RUN {i}/{len(all_docs)}] {doc.category} | {doc.nomor} | {doc.title}")
            continue

        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            doc.status = "skipped_exists"
            doc.size_bytes = out_path.stat().st_size
            doc.sha256 = sha256_file(out_path)
            print(f"[SKIP {i}/{len(all_docs)}] {out_path.name}")
            continue

        try:
            print(f"[DOWNLOAD {i}/{len(all_docs)}] {doc.category} | {doc.nomor} | {doc.title}")
            size, ctype = stream_download(session, doc.direct_download_url, out_path, args.timeout, args.retries)
            doc.status = "downloaded"
            doc.size_bytes = size
            doc.content_type = ctype
            doc.sha256 = sha256_file(out_path)
            time.sleep(args.delay)
        except Exception as exc:  # noqa: BLE001
            doc.status = "failed"
            doc.error = str(exc)
            errors.append(asdict(doc))
            print(f"  FAILED: {doc.error}", file=sys.stderr)

    manifest_rows = [asdict(doc) for doc in all_docs]
    write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
    write_csv(out_dir / "manifest.csv", manifest_rows)
    write_jsonl(out_dir / "errors.jsonl", errors)

    downloaded = sum(1 for d in all_docs if d.status == "downloaded")
    failed = sum(1 for d in all_docs if d.status == "failed")
    skipped = sum(1 for d in all_docs if d.status == "skipped_exists")

    print("\n[DONE]")
    print(f"  ditemukan   : {len(all_docs)}")
    print(f"  downloaded  : {downloaded}")
    print(f"  skipped     : {skipped}")
    print(f"  failed      : {failed}")
    print(f"  manifest    : {out_dir / 'manifest.jsonl'}")
    print(f"  csv         : {out_dir / 'manifest.csv'}")
    print(f"  errors      : {out_dir / 'errors.jsonl'}")

    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawler PDF DSN-MUI untuk RAG.")
    parser.add_argument("--out", default="data_dsnmui", help="Folder output hasil download.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_TARGETS.keys()),
        choices=list(DEFAULT_TARGETS.keys()),
        help="Kategori yang ingin di-crawl.",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Delay antar request/download, dalam detik.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout request, dalam detik.")
    parser.add_argument("--retries", type=int, default=3, help="Jumlah retry untuk request/download.")
    parser.add_argument("--max-pages", type=int, default=0, help="Batasi jumlah halaman per kategori. 0 = auto semua.")
    parser.add_argument("--max-files", type=int, default=0, help="Batasi jumlah file total. 0 = semua.")
    parser.add_argument("--dry-run", action="store_true", help="Crawl URL + metadata saja, tidak download PDF.")
    parser.add_argument("--overwrite", action="store_true", help="Download ulang meskipun file sudah ada.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(crawl(parse_args()))
