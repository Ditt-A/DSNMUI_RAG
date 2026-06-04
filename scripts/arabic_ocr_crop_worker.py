#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Arabic OCR crop worker for DSN-MUI RAG evidence.

Run this script using the PaddleOCR environment, for example:
LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH" "$PY_OCR" arabic_ocr_crop_worker.py \
  --input evidence_inputs.jsonl \
  --output evidence_ocr_outputs.jsonl \
  --use-gpu 1
"""
import argparse
import json
from pathlib import Path
from typing import Any, List

import numpy as np
from PIL import Image


def flatten_paddle_result(obj: Any) -> List[str]:
    texts = []
    if obj is None:
        return texts
    if isinstance(obj, str):
        if obj.strip():
            texts.append(obj.strip())
        return texts
    if isinstance(obj, dict):
        for key in ["rec_texts", "texts"]:
            if key in obj and isinstance(obj[key], list):
                for t in obj[key]:
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
        for key in ["text", "transcription", "rec_text"]:
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                texts.append(obj[key].strip())
        if not texts:
            for v in obj.values():
                texts.extend(flatten_paddle_result(v))
        return texts
    if isinstance(obj, (list, tuple)):
        # PaddleOCR v2: [box, (text, confidence)]
        if len(obj) >= 2 and isinstance(obj[1], (list, tuple)) and len(obj[1]) >= 1 and isinstance(obj[1][0], str):
            t = obj[1][0].strip()
            if t:
                texts.append(t)
            return texts
        for item in obj:
            texts.extend(flatten_paddle_result(item))
        return texts
    return texts


def normalize_text(text: str) -> str:
    import re
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file containing image_path rows.")
    parser.add_argument("--output", required=True, help="JSONL output path.")
    parser.add_argument("--lang", default="ar")
    parser.add_argument("--use-gpu", type=int, default=1)
    parser.add_argument("--ocr-version", default="PP-OCRv3")
    args = parser.parse_args()

    from paddleocr import PaddleOCR
    import paddle

    use_gpu = bool(args.use_gpu)
    print("Paddle:", paddle.__version__)
    try:
        print("Paddle device:", paddle.device.get_device())
    except Exception:
        pass

    try:
        ocr = PaddleOCR(
            lang=args.lang,
            ocr_version=args.ocr_version,
            use_angle_cls=True,
            use_gpu=use_gpu,
            show_log=False,
        )
    except TypeError:
        ocr = PaddleOCR(
            lang=args.lang,
            use_angle_cls=True,
            use_gpu=use_gpu,
        )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = row.get("image_path")
            try:
                img = np.array(Image.open(image_path).convert("RGB"))
                try:
                    result = ocr.ocr(img, cls=True)
                except TypeError:
                    result = ocr.ocr(img)
                texts = flatten_paddle_result(result)
                row["ocr_text"] = normalize_text("\n".join(texts))
                row["ocr_status"] = "ok"
                row["ocr_error"] = ""
            except Exception as e:
                row["ocr_text"] = ""
                row["ocr_status"] = "error"
                row["ocr_error"] = str(e)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {output_path}")


if __name__ == "__main__":
    main()
