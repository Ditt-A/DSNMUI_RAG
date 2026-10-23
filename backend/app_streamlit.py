from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import streamlit as st

from dsnmui_rag import RAGConfig, DSNMUIRAGPipeline
from dsnmui_rag.domain import infer_category_from_payload, infer_nomor_from_payload, short_title, payload_text
from dsnmui_rag.utils import text_preview, clean_output_text, format_answer_markdown

st.set_page_config(page_title="DSN-MUI UMKM RAG Prototype", layout="wide")
st.title("DSN-MUI UMKM RAG Prototype")
st.caption("Interface testing untuk pipeline RAG full: BGE-M3 + BM25 + RRF + statistical reranker + JSON/fallback + optional visual dalil V16.")

with st.sidebar:
    st.header("Config")
    precomp_dir = st.text_input("Path precomputed", value="dsnmui_rag_precompute_arabic_fix/precomputed")
    use_visual = st.toggle("Aktifkan Qwen3-VL visual dalil", value=False, help="Butuh GPU/VRAM besar dan PDF path asli masih valid.")
    use_visual_as_generator = st.toggle(
        "Gunakan Qwen3-VL juga untuk reasoning final",
        value=False,
        help="Pakai model Qwen3-VL yang sama untuk dalil dan jawaban final.",
    )
    category_filter = st.selectbox("Filter kategori", ["", "fatwa", "pedoman_implementasi", "talimat", "unknown"], index=0)
    final_k = st.slider("Final context K", min_value=1, max_value=8, value=4)
    st.divider()
    st.write("**Tips**")
    st.write("Untuk test cepat, biarkan generator dan visual dalil off dulu. Setelah retrieval cocok, baru aktifkan generator/visual.")

@st.cache_resource(show_spinner="Loading RAG index dan BGE-M3 embedder...")
def load_pipeline(precomp_dir: str, final_k: int) -> DSNMUIRAGPipeline:
    cfg = RAGConfig(
        precomp_dir=Path(precomp_dir),
        final_context_k=final_k,
    )

    pipe = DSNMUIRAGPipeline(cfg)

    pipe.retriever.load_embedder()
    pipe.retriever.embed_query("warmup query")

    return pipe

try:
    pipe = load_pipeline(
        precomp_dir=precomp_dir,
        final_k=final_k,
    )

    if use_visual or use_visual_as_generator:
        with st.spinner("Loading Qwen3-VL untuk visual dalil/reasoning..."):
            pipe.visual.load()
        st.sidebar.success("Qwen3-VL ready")

    with st.expander("Precomputed summary", expanded=False):
        st.json(pipe.summary())

except Exception as e:
    st.error(f"Gagal load pipeline: {type(e).__name__}: {e}")
    st.stop()

query = st.text_area("Pertanyaan UMKM", value="Saya jual produk pre-order di marketplace. Akad apa yang relevan menurut DSN-MUI?", height=90)
expected_focus = st.text_input("Expected focus / catatan fokus opsional", value="")

if st.button("Jalankan RAG", type="primary"):
    if not query.strip():
        st.warning("Isi pertanyaan dulu.")
        st.stop()
    with st.spinner("Menjalankan retrieval dan penyusunan jawaban..."):
        ans = pipe.answer(
            query.strip(),
            expected_focus=expected_focus,
            category_filter=category_filter or None,
            use_visual_dalil=use_visual,
            use_visual_as_generator=use_visual_as_generator,
        )
    st.subheader("Jawaban")
    st.markdown(format_answer_markdown(ans.answer))

    c1, c2, c3 = st.columns(3)
    c1.metric("Support", ans.support_info.get("support_level", "-"), ans.support_info.get("support_score", "-"))
    c2.metric("Retrieval ms", ans.timing.get("retrieval_total_ms", "-"))
    c3.metric("Generation mode", ans.generation_meta.get("generation_mode", "-"))

    st.subheader("Sources")
    rows = []
    for i, r in enumerate(ans.results, start=1):
        p = r.get("payload", {})
        vd = r.get("visual_dalil") or {}
        rows.append({
            "source": f"S{i}",
            "rank_final": r.get("rank_final"),
            "category": infer_category_from_payload(p),
            "nomor": infer_nomor_from_payload(p),
            "title": short_title(p.get("title"), 100),
            "page": p.get("page"),
            "score_reranker": r.get("score_reranker"),
            "score_vector": r.get("score_vector"),
            "score_bm25": r.get("score_bm25"),
            "visual_dalil_status": vd.get("status"),
            "dalil_type": vd.get("dalil_type"),
            "crop": vd.get("best_crop_image_path"),
            "preview": text_preview(clean_output_text(payload_text(p)), 800),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("Context yang masuk prompt"):
        st.text(clean_output_text(ans.context))
    with st.expander("Prompt generator"):
        st.text(clean_output_text(ans.prompt))
    with st.expander("Raw metadata JSON"):
        st.json({
            "support_info": ans.support_info,
            "timing": ans.timing,
            "generation_meta": ans.generation_meta,
        })
