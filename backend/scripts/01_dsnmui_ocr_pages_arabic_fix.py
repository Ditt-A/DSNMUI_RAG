#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 DSN-MUI RAG precompute: PDF/ZIP -> PyMuPDF + PaddleOCR Arabic -> pages.jsonl."""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time, unicodedata, zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz
import numpy as np
import pandas as pd
import regex as reg
from tqdm.auto import tqdm

ARABIC_PATTERN = reg.compile(r"[\p{Script=Arabic}]")

# Kata kunci untuk halaman yang biasanya memuat dalil Arab.
# Banyak PDF DSN-MUI mengekstrak teks Arab menjadi karakter Latin/punctuation berantakan,
# sehingga mode lama `arabic_or_low_text` bisa salah-skip OCR karena halaman terlihat punya teks cukup panjang.
DALIL_KEYWORDS = [
    "mengingat", "firman allah", "allah s.w.t", "allah swt",
    "q.s", "q.s.", "qs ", "al-qur", "hadis", "hadits", "riwayat",
    "nabi saw", "rasulullah", "kaidah fikih", "kaidah fiqh",
    "al-bukhari", "bukhari", "muslim", "abu dawud", "tirmidzi",
    "at-tirmidzi", "al-tirmidzi", "ibnu majah", "ibn majah",
    "baihaqi", "al-baihaqi", "thabrani", "al-thabrani",
]



def parse_bool(x):
    return x if isinstance(x, bool) else str(x).lower() in {"1", "true", "yes", "y", "on"}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).replace("\x00", " ").replace("\ufeff", " ")
    text = reg.sub(r"[\t\r\f\v]+", " ", text)
    text = reg.sub(r" *\n *", "\n", text)
    text = reg.sub(r"\n{3,}", "\n\n", text)
    text = reg.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def count_arabic_chars(text: str) -> int:
    return len(ARABIC_PATTERN.findall(text or ""))


def extract_arabic_text(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if count_arabic_chars(ln) >= 2]
    return normalize_text("\n".join(lines))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def unzip_if_needed(zip_path: Path, extract_dir: Path, data_dir: Path) -> Path:
    if zip_path.exists() and zip_path.suffix.lower() == ".zip":
        marker = extract_dir / ".unzipped_marker"
        if not marker.exists():
            print(f"Unzipping {zip_path} -> {extract_dir}")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            marker.write_text(str(zip_path), encoding="utf-8")
        else:
            print("ZIP sudah pernah diekstrak, skip unzip.")
        return extract_dir
    print("ZIP_PATH tidak ditemukan/invalid. Pakai DATA_DIR langsung.")
    return data_dir


def infer_category_from_path(path: Path) -> str:
    s = str(path).lower()
    if "pedoman" in s and "implementasi" in s:
        return "pedoman_implementasi"
    if "talimat" in s or "ta'limat" in s or "ta_limat" in s:
        return "talimat"
    if "fatwa" in s:
        return "fatwa"
    return "unknown"


def infer_nomor_title_from_filename(path: Path) -> Tuple[str, str]:
    m = re.match(r"^\s*(\d{1,4})\s*[-–—]\s*(.+?)\s*$", path.stem.strip())
    return (m.group(1), m.group(2).strip()) if m else ("", path.stem.strip())


def load_manifest_records(root: Path) -> Dict[str, Dict[str, Any]]:
    manifests = list(root.rglob("manifest.csv"))
    if not manifests:
        return {}
    manifest_path = manifests[0]
    print("Manifest ditemukan:", manifest_path)
    records = {}
    try:
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            d = row.to_dict()
            title = str(d.get("title") or d.get("judul") or "").strip()
            nomor = str(d.get("nomor") or d.get("number") or "").strip()
            category = str(d.get("category") or d.get("kategori") or "").strip()
            source_url = str(d.get("source_url") or d.get("url") or d.get("pdf_url") or "").strip()
            payload = {"title": title, "nomor": nomor, "category": category, "source_url": source_url}
            for col in ["filename", "file_name", "source_file", "local_path", "path", "filepath", "file_path", "pdf_path"]:
                if col in d and isinstance(d[col], str) and d[col].strip():
                    records[Path(d[col]).name] = payload
    except Exception as e:
        print("Gagal membaca manifest.csv:", e)
    return records


def discover_pdfs(root: Path) -> List[Dict[str, Any]]:
    pdfs = sorted([p for p in root.rglob("*.pdf") if p.is_file()])
    manifest_by_name = load_manifest_records(root)
    rows = []
    for i, pdf_path in enumerate(pdfs):
        num_name, title_name = infer_nomor_title_from_filename(pdf_path)
        man = manifest_by_name.get(pdf_path.name, {})
        category = man.get("category") or infer_category_from_path(pdf_path)
        nomor = man.get("nomor") or num_name
        title = man.get("title") or title_name
        doc_id_raw = f"{category}_{nomor}_{title}_{pdf_path.name}"
        doc_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", doc_id_raw.lower()).strip("_")[:180]
        rows.append({
            "doc_index": i, "doc_id": doc_id, "category": category, "nomor": nomor,
            "title": title, "source_url": man.get("source_url") or "",
            "pdf_path": str(pdf_path), "pdf_filename": pdf_path.name, "pdf_sha256": sha256_file(pdf_path),
        })
    return rows


def init_paddle_ocr(ocr_mode: str, use_gpu: bool):
    if ocr_mode == "none":
        print("OCR_MODE=none, OCR tidak diinisialisasi.")
        return None
    try:
        import paddle
        print("Paddle version:", paddle.__version__)
        paddle.set_device("gpu:0" if use_gpu else "cpu")
        print("Paddle device untuk OCR:", paddle.device.get_device())
        try:
            print("Paddle compiled CUDA:", paddle.version.cuda())
        except Exception:
            pass
    except Exception as e:
        print("WARNING: import/set paddle gagal:", e)
    from paddleocr import PaddleOCR
    try:
        from importlib.metadata import version
        print("PaddleOCR version:", version("paddleocr"))
    except Exception:
        pass
    candidates = [
        {"lang": "ar", "ocr_version": "PP-OCRv3", "use_angle_cls": True, "use_gpu": use_gpu, "show_log": False},
        {"lang": "ar", "use_angle_cls": True, "use_gpu": use_gpu, "show_log": False},
        {"lang": "ar", "use_gpu": use_gpu, "show_log": False},
        {"lang": "ar", "use_angle_cls": True, "use_gpu": use_gpu},
        {"lang": "ar", "use_gpu": use_gpu},
        {"lang": "fa", "ocr_version": "PP-OCRv3", "use_angle_cls": True, "use_gpu": use_gpu, "show_log": False},
        {"lang": "fa", "use_angle_cls": True, "use_gpu": use_gpu, "show_log": False},
        {"lang": "fa", "use_gpu": use_gpu},
    ]
    last_err = None
    for kwargs in candidates:
        try:
            print("Mencoba init PaddleOCR:", kwargs)
            ocr = PaddleOCR(**kwargs)
            print("PaddleOCR siap dengan kwargs:", kwargs)
            return ocr
        except Exception as e:
            last_err = e
            print("Gagal init dengan kwargs ini:", e)
    raise RuntimeError(f"Gagal inisialisasi PaddleOCR. Error terakhir: {last_err}")


def flatten_paddle_result(obj) -> List[str]:
    texts = []
    if obj is None:
        return texts
    if isinstance(obj, str):
        return [obj.strip()] if obj.strip() else []
    if isinstance(obj, dict):
        for key in ["rec_texts", "texts"]:
            if isinstance(obj.get(key), list):
                texts += [t.strip() for t in obj[key] if isinstance(t, str) and t.strip()]
        for key in ["text", "transcription", "rec_text"]:
            if isinstance(obj.get(key), str) and obj[key].strip():
                texts.append(obj[key].strip())
        if not texts:
            for v in obj.values():
                texts.extend(flatten_paddle_result(v))
        return texts
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and isinstance(obj[1], (list, tuple)) and len(obj[1]) >= 1 and isinstance(obj[1][0], str):
            return [obj[1][0].strip()] if obj[1][0].strip() else []
        for item in obj:
            texts.extend(flatten_paddle_result(item))
    return texts


def run_paddle_ocr_page(page: fitz.Page, ocr, dpi: int) -> str:
    if ocr is None:
        return ""
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n > 3:
        img = img[:, :, :3]
    try:
        result = ocr.ocr(img, cls=True)
    except TypeError:
        try:
            result = ocr.ocr(img)
        except AttributeError:
            result = ocr.predict(img)
    except AttributeError:
        result = ocr.predict(img)
    return normalize_text("\n".join(flatten_paddle_result(result)))


def is_dalil_page(text_pdf: str) -> bool:
    """Heuristik halaman dalil: ayat/hadis/kaidah sering butuh OCR visual Arabic."""
    t = normalize_text(text_pdf).lower()
    if not t:
        return False
    return any(k in t for k in DALIL_KEYWORDS)


def should_ocr_page(text_pdf: str, mode: str, min_text_chars: int):
    """Return (run_ocr, reason).

    Mode tambahan:
    - arabic_or_dalil_or_low_text: OCR halaman low text, punya Arabic Unicode,
      atau mengandung kata kunci dalil/hadis/ayat.
    - dalil_or_low_text: OCR halaman low text atau halaman dalil.
    """
    text_pdf = text_pdf or ""
    clean_len = len(normalize_text(text_pdf))
    arabic_count = count_arabic_chars(text_pdf)
    replacement_count = text_pdf.count("�") + text_pdf.count("\ufffd")
    low_text = clean_len < min_text_chars or replacement_count > 0
    has_arabic = arabic_count >= 10
    dalil_page = is_dalil_page(text_pdf)

    if mode == "none":
        return False, "none"
    if mode == "all":
        return True, "all"
    if mode == "low_text":
        return low_text, "low_text" if low_text else "skip"
    if mode == "arabic_or_low_text":
        if low_text:
            return True, "low_text"
        if has_arabic:
            return True, "arabic_unicode"
        return False, "skip"
    if mode == "dalil_or_low_text":
        if low_text:
            return True, "low_text"
        if dalil_page:
            return True, "dalil_keyword"
        return False, "skip"
    if mode == "arabic_or_dalil_or_low_text":
        if low_text:
            return True, "low_text"
        if has_arabic:
            return True, "arabic_unicode"
        if dalil_page:
            return True, "dalil_keyword"
        return False, "skip"
    return low_text, "low_text" if low_text else "skip"

def merge_text_sources(text_pdf: str, text_ocr: str) -> str:
    text_pdf, text_ocr = normalize_text(text_pdf), normalize_text(text_ocr)
    if not text_pdf:
        return text_ocr
    if not text_ocr:
        return text_pdf
    if sha256_text(text_pdf[:2000]) == sha256_text(text_ocr[:2000]):
        return text_pdf
    return normalize_text(text_pdf + "\n\n[OCR_VISUAL_TEXT]\n" + text_ocr)


def parse_one_pdf(record: Dict[str, Any], ocr_engine, args) -> List[Dict[str, Any]]:
    rows = []
    try:
        doc = fitz.open(record["pdf_path"])
    except Exception as e:
        return [{**record, "page": None, "page_count": None, "parse_status": "failed_open_pdf", "error": str(e)}]
    for page_idx in range(len(doc)):
        page_no = page_idx + 1
        try:
            page = doc[page_idx]
            text_pdf = normalize_text(page.get_text("text", sort=True))
            run_ocr, ocr_trigger = should_ocr_page(text_pdf, args.ocr_mode, args.min_text_chars)
            text_ocr, ocr_error = "", ""
            if run_ocr:
                try:
                    text_ocr = run_paddle_ocr_page(page, ocr_engine, args.ocr_dpi)
                except Exception as e:
                    ocr_error = str(e)
            combined = merge_text_sources(text_pdf, text_ocr)
            rows.append({
                **record, "page": page_no, "page_count": len(doc), "parse_status": "ok",
                "ocr_mode": args.ocr_mode, "ocr_trigger": ocr_trigger, "ocr_requested": bool(run_ocr), "ocr_used": bool(run_ocr and text_ocr), "ocr_error": ocr_error,
                "text_pdf": text_pdf, "text_ocr": text_ocr, "text_combined": combined,
                "arabic_text": extract_arabic_text(combined), "text_pdf_len": len(text_pdf),
                "text_ocr_len": len(text_ocr), "text_combined_len": len(combined),
                "arabic_char_count": count_arabic_chars(combined),
            })
        except Exception as e:
            rows.append({**record, "page": page_no, "page_count": len(doc), "parse_status": "failed_parse_page", "error": str(e)})
    doc.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", default="data_dsnmui.zip")
    parser.add_argument("--data-dir", default="dsnmui_data")
    parser.add_argument("--project-dir", default="dsnmui_rag_precompute")
    parser.add_argument("--ocr-mode", choices=["none", "low_text", "arabic_or_low_text", "dalil_or_low_text", "arabic_or_dalil_or_low_text", "all"], default="arabic_or_dalil_or_low_text")
    parser.add_argument("--ocr-dpi", type=int, default=220)
    parser.add_argument("--min-text-chars", type=int, default=250)
    parser.add_argument("--use-gpu", type=parse_bool, default=True)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--max-pdfs", type=int, default=0)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    extract_dir, precomp_dir = project_dir / "extracted", project_dir / "precomputed"
    pages_path, manifest_out, stats_path = precomp_dir / "pages.jsonl", precomp_dir / "pdf_manifest.jsonl", precomp_dir / "ocr_stats.json"
    for p in [project_dir, extract_dir, precomp_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("Python:", sys.version)
    print("PROJECT_DIR:", project_dir)
    print("OCR_MODE:", args.ocr_mode, "OCR_DPI:", args.ocr_dpi, "USE_GPU:", args.use_gpu)
    data_root = unzip_if_needed(Path(args.zip_path), extract_dir, Path(args.data_dir))
    pdf_records = discover_pdfs(data_root)
    if args.max_pdfs > 0:
        pdf_records = pdf_records[:args.max_pdfs]
        print(f"Smoke test: hanya proses {len(pdf_records)} PDF pertama")
    print("Total PDF:", len(pdf_records))
    write_jsonl(manifest_out, pdf_records)

    ocr_engine = init_paddle_ocr(args.ocr_mode, args.use_gpu)
    existing_pages = read_jsonl(pages_path) if args.resume else []
    processed = {(r.get("doc_id"), r.get("page")) for r in existing_pages if r.get("parse_status") == "ok" and r.get("doc_id") and r.get("page")}
    print("Existing parsed pages:", len(existing_pages), "Processed page keys:", len(processed))

    t0, new_total = time.time(), 0
    for rec in tqdm(pdf_records, desc="OCR/PDF parsing"):
        skip_doc = False
        if args.resume:
            try:
                with fitz.open(rec["pdf_path"]) as d:
                    skip_doc = all((rec["doc_id"], p) in processed for p in range(1, len(d) + 1))
            except Exception:
                skip_doc = False
        if skip_doc:
            continue
        rows = parse_one_pdf(rec, ocr_engine, args)
        append_jsonl(pages_path, rows)
        new_total += len(rows)

    pages = read_jsonl(pages_path)
    ok = [r for r in pages if r.get("parse_status") == "ok"]
    stats = {
        "num_pdfs": len(pdf_records), "num_page_rows": len(pages), "num_ok_rows": len(ok),
        "num_new_rows_this_run": new_total, "ocr_requested_pages": sum(1 for r in ok if r.get("ocr_requested")), "ocr_used_pages": sum(1 for r in ok if r.get("ocr_used")),
        "arabic_pages": sum(1 for r in ok if int(r.get("arabic_char_count") or 0) > 0),
        "ocr_trigger_counts": {k: sum(1 for r in ok if r.get("ocr_trigger") == k) for k in sorted({r.get("ocr_trigger") for r in ok})},
        "ocr_mode": args.ocr_mode, "ocr_dpi": args.ocr_dpi, "use_gpu_for_paddleocr": args.use_gpu,
        "elapsed_seconds": round(time.time() - t0, 2), "pages_path": str(pages_path), "manifest_path": str(manifest_out),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("DONE Stage 1 OCR pages. Next: run 02_dsnmui_embed_index.py in embedding env.")


if __name__ == "__main__":
    main()
