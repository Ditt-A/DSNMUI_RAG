from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import gc, json, re, traceback

from .config import RAGConfig
from .domain import (
    payload_text, infer_category_from_payload, infer_nomor_from_payload,
    contains_dalil, should_run_arabic_evidence,
)
from .utils import (
    normalize_text, text_preview, safe_name, count_arabic, arabic_quality_score,
    strip_code_fence, extract_first_json_object_text, extract_json_object,
)

VISUAL_DALIL_ALLOWED_TYPES = {"ayat", "hadis", "kaidah", "kutipan_ulama"}
VISUAL_DALIL_REJECT_TYPES = {"istilah_arab", "unknown", "not_dalil", "none"}


class VisualDalilExtractor:
    """Chunk-aligned Qwen3-VL visual dalil layer with V16 quality gate."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.config.ensure_dirs()

    def release(self) -> None:
        self.processor = None
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(self.config.visual_dalil_model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.visual_dalil_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()

    def has_arabic(self, text: str, min_count: int = 1) -> bool:
        return count_arabic(text or "") >= min_count

    def get_chunk_arabic_signal(self, result: Dict[str, Any]) -> Dict[str, Any]:
        p = result.get("payload", {})
        txt = payload_text(p)
        raw_ar = p.get("arabic_text") or ""
        combined = f"{raw_ar}\n{txt}"
        arabic_count = count_arabic(combined)
        return {
            "has_arabic": arabic_count >= self.config.visual_dalil_min_arabic_in_chunk,
            "arabic_count": arabic_count,
            "has_dalil_keyword": contains_dalil(txt),
            "raw_arabic_text": raw_ar,
            "chunk_text_preview": text_preview(txt, 1000),
        }

    def source_should_use_visual_dalil(self, item: Dict[str, Any], result: Dict[str, Any]) -> bool:
        p = result.get("payload", {})
        if not (p.get("pdf_path") and p.get("page")):
            return False
        q = item.get("query") or ""
        query_needs_dalil = bool(item.get("needs_dalil")) or should_run_arabic_evidence(q)
        signal = self.get_chunk_arabic_signal(result)
        if query_needs_dalil:
            return bool(signal["has_arabic"] or signal["has_dalil_keyword"])
        if self.config.visual_dalil_only_for_dalil_queries:
            return bool(self.config.visual_dalil_force_for_source_with_dalil and signal["has_arabic"] and signal["has_dalil_keyword"])
        return bool(signal["has_arabic"])

    @staticmethod
    def _arabic_terms_for_match(s: str, min_len: int = 3, max_terms: int = 40) -> List[str]:
        toks = re.findall(r"[\u0600-\u06FF]{%d,}" % min_len, s or "")
        seen, out = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) >= max_terms:
                break
        return out

    def find_arabic_bboxes_from_pdf_text_layer(self, pdf_path: str, page_no: int, chunk_signal_text: str = "") -> List[Dict[str, Any]]:
        try:
            import fitz
        except Exception:
            return []
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return []
        terms = self._arabic_terms_for_match(chunk_signal_text)
        rows: List[Dict[str, Any]] = []
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[int(page_no) - 1]
            data = page.get_text("dict")
            for bi, block in enumerate(data.get("blocks", [])):
                if block.get("type") != 0:
                    continue
                for li, line in enumerate(block.get("lines", [])):
                    spans = line.get("spans", [])
                    line_text = "".join(sp.get("text", "") for sp in spans)
                    ar_count = count_arabic(line_text)
                    if ar_count < 2:
                        continue
                    bbox = None
                    for sp in spans:
                        if count_arabic(sp.get("text", "")) < 1:
                            continue
                        rect = fitz.Rect(sp.get("bbox"))
                        bbox = rect if bbox is None else (bbox | rect)
                    if bbox is None:
                        bbox = fitz.Rect(line.get("bbox"))
                    term_hits = sum(1 for t in terms if t in line_text)
                    score = (term_hits * 10.0) + min(ar_count / 8.0, 8.0)
                    rows.append({
                        "bbox": [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)],
                        "text": line_text,
                        "arabic_count": ar_count,
                        "term_hits": term_hits,
                        "score": score,
                        "block_index": bi,
                        "line_index": li,
                    })
        finally:
            doc.close()
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows

    def _pad_rect(self, rect, page_rect):
        import fitz
        pad = self.config.visual_dalil_clip_padding_pt
        r = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
        r.x0 = max(page_rect.x0, r.x0)
        r.y0 = max(page_rect.y0, r.y0)
        r.x1 = min(page_rect.x1, r.x1)
        r.y1 = min(page_rect.y1, r.y1)
        return r

    def merge_close_bboxes(self, bbox_rows: List[Dict[str, Any]], page_rect, max_groups: int = 2) -> List[Dict[str, Any]]:
        import fitz
        if not bbox_rows:
            return []
        selected, used = [], set()
        for row in bbox_rows[:8]:
            key = (row.get("block_index"), row.get("line_index"))
            if key in used:
                continue
            r0 = fitz.Rect(row["bbox"])
            group_rect = fitz.Rect(r0)
            group_texts = [row.get("text", "")]
            group_score = float(row.get("score", 0.0))
            used.add(key)
            center_y = (r0.y0 + r0.y1) / 2
            for other in bbox_rows[:20]:
                okey = (other.get("block_index"), other.get("line_index"))
                if okey in used:
                    continue
                ro = fitz.Rect(other["bbox"])
                oy = (ro.y0 + ro.y1) / 2
                if abs(oy - center_y) <= 85:
                    group_rect |= ro
                    group_texts.append(other.get("text", ""))
                    group_score += float(other.get("score", 0.0))
                    used.add(okey)
            selected.append({
                "rect": self._pad_rect(group_rect, page_rect),
                "score": group_score,
                "text_preview": text_preview(" ".join(group_texts), 500),
                "source": "pdf_text_layer_arabic_bbox",
            })
            if len(selected) >= max_groups:
                break
        selected.sort(key=lambda x: x["score"], reverse=True)
        return selected[:max_groups]

    def render_pdf_clip(self, pdf_path: str, page_no: int, rect, out_path: Path) -> Path:
        import fitz
        out_path = Path(out_path)
        if out_path.exists():
            return out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[int(page_no) - 1]
            zoom = self.config.visual_dalil_render_dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
            pix.save(str(out_path))
        finally:
            doc.close()
        return out_path

    def heuristic_clip_rects(self, page_rect, names: Optional[List[str]] = None) -> List[Tuple[str, Any]]:
        import fitz
        names = names or list(self.config.visual_dalil_heuristic_clips)
        w, h = page_rect.width, page_rect.height
        specs = {
            "upper_middle": fitz.Rect(page_rect.x0 + 0.04*w, page_rect.y0 + 0.08*h, page_rect.x0 + 0.96*w, page_rect.y0 + 0.58*h),
            "middle": fitz.Rect(page_rect.x0 + 0.04*w, page_rect.y0 + 0.22*h, page_rect.x0 + 0.96*w, page_rect.y0 + 0.78*h),
            "lower_middle": fitz.Rect(page_rect.x0 + 0.04*w, page_rect.y0 + 0.38*h, page_rect.x0 + 0.96*w, page_rect.y0 + 0.96*h),
            "upper_70": fitz.Rect(page_rect.x0 + 0.03*w, page_rect.y0 + 0.04*h, page_rect.x0 + 0.97*w, page_rect.y0 + 0.74*h),
            "middle_80": fitz.Rect(page_rect.x0 + 0.03*w, page_rect.y0 + 0.10*h, page_rect.x0 + 0.97*w, page_rect.y0 + 0.90*h),
            "lower_75": fitz.Rect(page_rect.x0 + 0.03*w, page_rect.y0 + 0.25*h, page_rect.x0 + 0.97*w, page_rect.y0 + 0.98*h),
        }
        return [(n, specs[n]) for n in names if n in specs]
    
    def generate_text(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """
        Memakai Qwen3-VL yang sama untuk generate jawaban text-only.
        Ini membuat Qwen3-VL bisa dipakai dua kali:
        1) ekstraksi dalil dari image crop
        2) reasoning final dari prompt teks
        """
        self.load()

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=getattr(self.processor.tokenizer, "eos_token_id", None),
            )

        gen = out_ids[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()

    def crop_visual_candidates_chunk_aligned(self, result: Dict[str, Any], item: Dict[str, Any], source_label: str, max_crops: Optional[int] = None) -> List[Dict[str, Any]]:
        import fitz
        from PIL import Image
        max_crops = max_crops or self.config.visual_dalil_max_crops_per_source
        p = result.get("payload", {})
        pdf_path = p.get("pdf_path")
        page_no = p.get("page")
        chunk_id = str(p.get("chunk_id") or source_label)
        if not pdf_path or not page_no:
            return []
        signal = self.get_chunk_arabic_signal(result)
        if self.config.visual_dalil_require_arabic_in_chunk and not (signal["has_arabic"] or signal["has_dalil_keyword"]):
            return []
        chunk_signal_text = "\n".join([str(p.get("title") or ""), str(signal.get("raw_arabic_text") or ""), str(payload_text(p) or "")])
        crops: List[Dict[str, Any]] = []
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[int(page_no) - 1]
            page_rect = page.rect
        finally:
            doc.close()

        bbox_rows = self.find_arabic_bboxes_from_pdf_text_layer(pdf_path, int(page_no), chunk_signal_text)
        groups = self.merge_close_bboxes(bbox_rows, page_rect, max_groups=max_crops)
        for gi, g in enumerate(groups, start=1):
            rect = g["rect"]
            out = self.config.crop_dir / f"visual_v16_chunkclip_{safe_name(chunk_id)}_p{page_no}_bbox{gi}.png"
            self.render_pdf_clip(pdf_path, int(page_no), rect, out)
            with Image.open(out) as im:
                width, height = im.size
            crops.append({
                "crop_name": f"chunk_arabic_bbox_{gi}",
                "image_path": str(out),
                "page_image_path": None,
                "box_pdf_points": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                "box": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                "width": width,
                "height": height,
                "bbox_source": g.get("source"),
                "bbox_score": g.get("score"),
                "bbox_text_preview": g.get("text_preview"),
                "chunk_arabic_count": signal.get("arabic_count"),
            })
            if len(crops) >= max_crops:
                return crops

        q = item.get("query") or ""
        query_needs_dalil = bool(item.get("needs_dalil")) or should_run_arabic_evidence(q)
        fallback_names = list(self.config.visual_dalil_heuristic_clips)
        if self.config.visual_dalil_enable_wider_same_page_fallback and query_needs_dalil:
            for n in self.config.visual_dalil_wider_clips:
                if n not in fallback_names:
                    fallback_names.append(n)
        if self.config.visual_dalil_fallback_heuristic_clip:
            for name, rect in self.heuristic_clip_rects(page_rect, fallback_names):
                out = self.config.crop_dir / f"visual_v16_chunkclip_{safe_name(chunk_id)}_p{page_no}_{name}.png"
                self.render_pdf_clip(pdf_path, int(page_no), rect, out)
                with Image.open(out) as im:
                    width, height = im.size
                crops.append({
                    "crop_name": f"heuristic_{name}",
                    "image_path": str(out),
                    "page_image_path": None,
                    "box_pdf_points": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                    "box": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                    "width": width,
                    "height": height,
                    "bbox_source": "heuristic_wider_same_page_clip_v16" if name in self.config.visual_dalil_wider_clips else "heuristic_clip_no_text_layer_bbox_v16",
                    "bbox_score": None,
                    "bbox_text_preview": "",
                    "chunk_arabic_count": signal.get("arabic_count"),
                })
                if len(crops) >= max_crops:
                    break
        return crops

    @staticmethod
    def _safe_lower(x: Any) -> str:
        return str(x or "").strip().lower()

    @staticmethod
    def _json_loads_maybe(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    @staticmethod
    def _normalize_vlm_key_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
        key_map = {
            "arabic": "arabic_text", "text_arabic": "arabic_text", "arabic_transcription": "arabic_text",
            "transcription": "arabic_text", "teks_arab": "arabic_text", "type": "dalil_type", "jenis": "dalil_type",
            "jenis_dalil": "dalil_type", "visible": "arabic_visible",
        }
        return {key_map.get(str(k).strip(), str(k).strip()): v for k, v in obj.items()} if isinstance(obj, dict) else {}

    def _clean_vlm_arabic_text(self, value: Any, raw: str = "") -> str:
        if isinstance(value, dict):
            value = value.get("arabic_text") or value.get("arabic") or value.get("text") or ""
        if isinstance(value, list):
            value = " ".join(str(x) for x in value if x is not None)
        text = str(value or "").strip()
        if text.startswith("{") or '"arabic_text"' in text or '"arabic"' in text:
            sub = self._json_loads_maybe(text)
            if isinstance(sub, dict):
                sub = self._normalize_vlm_key_obj(sub)
                text = str(sub.get("arabic_text") or sub.get("arabic") or "")
            else:
                jtxt = extract_first_json_object_text(text)
                sub = self._json_loads_maybe(jtxt) if jtxt else None
                if isinstance(sub, dict):
                    sub = self._normalize_vlm_key_obj(sub)
                    text = str(sub.get("arabic_text") or sub.get("arabic") or "")
        if '"arabic_visible"' in text or '"confidence"' in text or '"dalil_type"' in text:
            text = ""
        text = strip_code_fence(text)
        text = re.sub(r"(?i)^arabic_text\s*[:：]\s*", "", text).strip()
        text = re.sub(r"(?i)^arabic\s*[:：]\s*", "", text).strip()
        return normalize_text(text)

    def classify_visual_dalil_type(self, arabic_text: str, raw_obj: Optional[Dict[str, Any]] = None, query: str = "", source_text: str = "") -> str:
        raw_obj = raw_obj or {}
        supplied = self._safe_lower(raw_obj.get("dalil_type")).replace("-", "_").replace(" ", "_")
        alias = {
            "ayah": "ayat", "quran": "ayat", "al_quran": "ayat", "alquran": "ayat", "ayat_al_quran": "ayat",
            "hadith": "hadis", "hadeeth": "hadis", "hadis_nabi": "hadis",
            "kaidah_fiqh": "kaidah", "kaidah_fikih": "kaidah", "qawaid": "kaidah", "qaidah": "kaidah",
            "quote": "kutipan_ulama", "kutipan": "kutipan_ulama", "ulama": "kutipan_ulama",
            "term": "istilah_arab", "istilah": "istilah_arab", "istilah_arab": "istilah_arab",
            "not_dalil": "not_dalil", "none": "none", "unknown": "unknown",
        }
        supplied = alias.get(supplied, supplied)
        if supplied in VISUAL_DALIL_ALLOWED_TYPES or supplied in VISUAL_DALIL_REJECT_TYPES:
            return supplied
        txt = arabic_text or ""
        combined = f"{txt}\n{source_text}".lower()
        ayat_markers = ["قال الله", "قوله تعالى", "يا أيها", "إن الله", "والله", "وأحل", "وحرم", "البقرة", "النساء", "المائدة"]
        hadis_markers = ["قال رسول", "رسول الله", "صلى الله عليه وسلم", "رواه", "عن ", "حديث", "النبي", "نهى", "لا ضرر"]
        kaidah_markers = ["الأصل", "الضرر", "المشقة", "اليقين", "العادة", "الأمور", "درء", "جلب", "لا ضرر ولا ضرار", "الإباحة", "المعاملات"]
        if any(m in combined for m in hadis_markers):
            return "hadis"
        if any(m in combined for m in kaidah_markers):
            return "kaidah"
        if any(m in combined for m in ayat_markers):
            return "ayat"
        words = re.findall(r"[\u0600-\u06FF]+", txt)
        if count_arabic(txt) < 25 or len(words) <= 3:
            return "istilah_arab"
        return "unknown"

    def is_visual_dalil_accepted(self, arabic_text: str, obj: Dict[str, Any], quality: Dict[str, Any], dalil_type: str) -> bool:
        conf = self._safe_lower(obj.get("confidence") or "low")
        visible = obj.get("arabic_visible")
        if isinstance(visible, str):
            visible = visible.strip().lower() in ["true", "1", "yes", "ya"]
        if visible is not True:
            return False
        if dalil_type not in VISUAL_DALIL_ALLOWED_TYPES:
            return False
        if count_arabic(arabic_text) < self.config.visual_dalil_min_arabic_count:
            return False
        if quality.get("score", 0.0) < self.config.visual_dalil_min_quality_score:
            return False
        if (not self.config.visual_dalil_accept_low_confidence) and conf == "low":
            return False
        if "[TIDAK TERBACA]" in arabic_text or "TIDAK TERBACA" in arabic_text:
            return False
        return True

    def build_visual_dalil_prompt(self, source_label: str, item: Dict[str, Any], crop_name: str) -> str:
        query = item.get("query") or ""
        return f"""
Anda membaca crop gambar dari dokumen DSN-MUI.
Tugas Anda adalah MENYALIN teks Arab yang terlihat dan MENGKLASIFIKASIKAN apakah teks Arab itu benar-benar dalil.

Aturan penting:
- Salin PERSIS teks Arab yang terlihat pada gambar.
- Jangan memperbaiki, jangan melengkapi, jangan menambahkan harakat jika tidak terlihat.
- Jangan membuat dalil baru.
- Kalau bagian teks tidak jelas, tulis [TIDAK TERBACA] pada bagian itu.
- Kalau teks Arab yang terlihat hanya istilah akad/istilah Arab biasa, beri dalil_type="istilah_arab".
- Kalau tidak ada teks Arab terlihat, arabic_visible=false dan arabic_text="".
- Output JSON valid saja, tanpa markdown.

Konteks query pengguna: {query}
Source: {source_label}
Crop: {crop_name}

Schema JSON wajib:
{{
  "arabic_visible": true,
  "arabic_text": "...",
  "dalil_type": "ayat|hadis|kaidah|kutipan_ulama|istilah_arab|unknown|none",
  "confidence": "high|medium|low",
  "unclear_parts": ["..."],
  "notes": "singkat"
}}
""".strip()

    def _vl_parse_json(self, raw: str) -> Dict[str, Any]:
        raw_clean = strip_code_fence(raw or "").strip()
        obj = None
        parsed = self._json_loads_maybe(raw_clean)
        if isinstance(parsed, dict):
            obj = parsed
        if obj is None:
            jtxt = extract_first_json_object_text(raw_clean)
            parsed = self._json_loads_maybe(jtxt) if jtxt else None
            if isinstance(parsed, dict):
                obj = parsed
        if obj is None:
            ext = extract_json_object(raw_clean)
            if isinstance(ext, dict):
                obj = ext
        if obj is None:
            arabic_text = self._clean_vlm_arabic_text(raw_clean, raw=raw_clean)
            obj = {
                "arabic_visible": bool(count_arabic(arabic_text) >= self.config.visual_dalil_min_arabic_count),
                "arabic_text": arabic_text,
                "dalil_type": self.classify_visual_dalil_type(arabic_text, {}, source_text=raw_clean),
                "confidence": "low",
                "unclear_parts": [],
                "notes": "raw_non_json_output",
            }
        else:
            obj = self._normalize_vlm_key_obj(obj)
            arabic_text = self._clean_vlm_arabic_text(obj.get("arabic_text"), raw=raw_clean)
            obj["arabic_text"] = arabic_text
            obj["dalil_type"] = self.classify_visual_dalil_type(arabic_text, obj, source_text=raw_clean)
            obj.setdefault("arabic_visible", bool(count_arabic(arabic_text) >= self.config.visual_dalil_min_arabic_count))
            obj.setdefault("confidence", "low")
            if not isinstance(obj.get("unclear_parts"), list):
                obj["unclear_parts"] = []
            obj.setdefault("notes", "")
        return obj

    def transcribe_arabic_from_image_vlm(self, image_path: str, source_label: str, item: Dict[str, Any], crop_name: str) -> Dict[str, Any]:
        self.load()
        import torch
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        prompt = self.build_visual_dalil_prompt(source_label, item, crop_name)
        messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        except Exception:
            inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.visual_dalil_max_new_tokens,
                do_sample=False,
                temperature=self.config.visual_dalil_temperature,
                pad_token_id=getattr(self.processor.tokenizer, "eos_token_id", None),
            )
        gen = out_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self.processor.decode(gen, skip_special_tokens=True).strip()
        obj = self._vl_parse_json(raw)
        arabic_text = self._clean_vlm_arabic_text(obj.get("arabic_text"), raw=raw)
        dalil_type = self.classify_visual_dalil_type(arabic_text, obj, query=item.get("query") or "", source_text=raw)
        obj["arabic_text"] = arabic_text
        obj["dalil_type"] = dalil_type
        q = arabic_quality_score(arabic_text)
        accepted = self.is_visual_dalil_accepted(arabic_text, obj, q, dalil_type)
        if accepted:
            quality_status = "verified_visual_candidate"
        elif count_arabic(arabic_text) >= self.config.visual_dalil_min_arabic_count and dalil_type in VISUAL_DALIL_ALLOWED_TYPES:
            quality_status = "visual_needs_verification"
        elif count_arabic(arabic_text) >= self.config.visual_dalil_min_arabic_count:
            quality_status = "arabic_detected_not_dalil"
        else:
            quality_status = "no_readable_arabic"
        return {
            "image_path": str(image_path),
            "source_label": source_label,
            "crop_name": crop_name,
            "raw_generation": raw,
            "parsed": obj,
            "arabic_text": arabic_text,
            "dalil_type": dalil_type,
            "confidence": obj.get("confidence") or "low",
            "unclear_parts": obj.get("unclear_parts") if isinstance(obj.get("unclear_parts"), list) else [],
            "notes": obj.get("notes") or "",
            "quality": q,
            "quality_status": quality_status,
            "accepted": accepted,
            "needs_human_verification": True,
        }

    def visual_dalil_for_result(self, result: Dict[str, Any], item: Dict[str, Any], source_idx: int) -> Dict[str, Any]:
        p = result.get("payload", {})
        source_label = f"S{source_idx} | {infer_category_from_payload(p)} {infer_nomor_from_payload(p)} | halaman {p.get('page')}"
        signal = self.get_chunk_arabic_signal(result)
        base = {
            "status": "not_run",
            "source_id": f"S{source_idx}",
            "source_label": source_label,
            "category": infer_category_from_payload(p),
            "nomor": infer_nomor_from_payload(p),
            "title": p.get("title"),
            "page": p.get("page"),
            "pdf_path": p.get("pdf_path"),
            "page_image_path": None,
            "best_crop_image_path": None,
            "best_crop_name": None,
            "arabic_text": "",
            "dalil_type": "none",
            "confidence": "low",
            "quality": {},
            "quality_status": "not_run",
            "accepted": False,
            "needs_human_verification": True,
            "candidates": [],
            "error": None,
            "render_strategy": "chunk_aligned_clip_v16_quality_gate",
            "chunk_has_arabic": signal.get("has_arabic"),
            "chunk_arabic_count": signal.get("arabic_count"),
            "chunk_has_dalil_keyword": signal.get("has_dalil_keyword"),
        }
        if not self.config.visual_dalil_enabled:
            base["status"] = "disabled"
            return base
        if not self.source_should_use_visual_dalil(item, result):
            base["status"] = "skipped_no_arabic_in_same_chunk_or_no_pdf"
            return base
        try:
            crops = self.crop_visual_candidates_chunk_aligned(result, item, source_label, max_crops=self.config.visual_dalil_max_crops_per_source)
            if not crops:
                base["status"] = "no_chunk_aligned_crop"
                return base
            candidates = []
            for crop in crops:
                cand = self.transcribe_arabic_from_image_vlm(crop["image_path"], source_label, item, crop["crop_name"])
                cand.update({
                    "page_image_path": None,
                    "box": crop.get("box"),
                    "box_pdf_points": crop.get("box_pdf_points"),
                    "width": crop.get("width"),
                    "height": crop.get("height"),
                    "bbox_source": crop.get("bbox_source"),
                    "bbox_score": crop.get("bbox_score"),
                    "bbox_text_preview": crop.get("bbox_text_preview"),
                    "chunk_arabic_count": crop.get("chunk_arabic_count"),
                })
                candidates.append(cand)

            def cand_rank(c):
                accepted_bonus = 1000 if c.get("accepted") else 0
                type_bonus = 120 if c.get("dalil_type") in VISUAL_DALIL_ALLOWED_TYPES else 0
                conf_bonus = {"high": 40, "medium": 20, "low": 0}.get(self._safe_lower(c.get("confidence")), 0)
                qscore = (c.get("quality") or {}).get("score", 0.0)
                return accepted_bonus + type_bonus + conf_bonus + qscore

            best = max(candidates, key=cand_rank) if candidates else None
            if best:
                quality_status = best.get("quality_status") or "unknown"
                if best.get("accepted"):
                    status = "accepted"
                elif quality_status == "visual_needs_verification":
                    status = "visual_needs_verification"
                elif quality_status == "arabic_detected_not_dalil":
                    status = "arabic_detected_not_dalil"
                elif quality_status == "no_readable_arabic":
                    status = "no_readable_arabic"
                else:
                    status = "low_quality"
                base.update({
                    "status": status,
                    "best_crop_image_path": best.get("image_path"),
                    "best_crop_name": best.get("crop_name"),
                    "arabic_text": best.get("arabic_text") or "",
                    "dalil_type": best.get("dalil_type") or "unknown",
                    "confidence": best.get("confidence") or "low",
                    "quality": best.get("quality") or {},
                    "quality_status": quality_status,
                    "accepted": bool(best.get("accepted")),
                    "needs_human_verification": True,
                    "candidates": candidates,
                })
            else:
                base["status"] = "no_candidate"
            return base
        except Exception as e:
            base["status"] = "error"
            base["error"] = f"{type(e).__name__}: {str(e)}"
            traceback.print_exc()
            return base

    def enrich_results(self, item: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.config.visual_dalil_enabled:
            return results
        out: List[Dict[str, Any]] = []
        for i, r in enumerate(results, start=1):
            rr = dict(r)
            if i <= self.config.visual_dalil_max_sources:
                rr["visual_dalil"] = self.visual_dalil_for_result(rr, item, i)
            out.append(rr)
        return out
