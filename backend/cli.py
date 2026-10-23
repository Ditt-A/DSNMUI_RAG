from __future__ import annotations
import argparse
from pathlib import Path

from dsnmui_rag import RAGConfig, DSNMUIRAGPipeline
from dsnmui_rag.domain import infer_category_from_payload, infer_nomor_from_payload, short_title
from dsnmui_rag.utils import text_preview


def main():
    parser = argparse.ArgumentParser(description="DSN-MUI UMKM RAG prototype CLI")
    parser.add_argument("query", help="Pertanyaan user/UMKM")
    parser.add_argument("--precomp-dir", default="dsnmui_rag_precompute_arabic_fix/precomputed")
    parser.add_argument("--category", default=None, choices=[None, "fatwa", "pedoman_implementasi", "talimat", "unknown"], nargs="?")
    parser.add_argument("--generator", action="store_true", help="Gunakan Qwen3-VL untuk reasoning final. Default: fallback ekstraktif.")
    parser.add_argument("--visual-dalil", action="store_true", help="Aktifkan Qwen3-VL visual dalil layer.")
    parser.add_argument("--final-k", type=int, default=4)
    args = parser.parse_args()

    cfg = RAGConfig(precomp_dir=Path(args.precomp_dir), final_context_k=args.final_k, visual_dalil_enabled=args.visual_dalil)
    pipe = DSNMUIRAGPipeline(cfg)
    ans = pipe.answer(args.query, category_filter=args.category, use_visual_as_generator=args.generator, use_visual_dalil=args.visual_dalil)

    print("\n" + ans.answer)
    print("\n" + "=" * 90)
    print("SUPPORT:", ans.support_info)
    print("TIMING:", ans.timing)
    print("GENERATION:", ans.generation_meta)
    print("\nSOURCES")
    for i, r in enumerate(ans.results, start=1):
        p = r.get("payload", {})
        print("-", f"S{i}", infer_category_from_payload(p), infer_nomor_from_payload(p), "page", p.get("page"), short_title(p.get("title"), 100))
        print("  ", text_preview(p.get("text_original") or p.get("text_for_embedding") or "", 220).replace("\n", " "))


if __name__ == "__main__":
    main()
