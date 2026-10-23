"""Verify CLI flags reach the current pipeline interface without inference."""

import sys
from types import SimpleNamespace

import cli


def test_generator_flag_uses_visual_generator_argument(monkeypatch):
    calls = []

    def answer(query, *, category_filter, use_visual_as_generator, use_visual_dalil):
        calls.append((query, category_filter, use_visual_as_generator, use_visual_dalil))
        return SimpleNamespace(
            answer="Contoh", support_info={}, timing={}, generation_meta={}, results=[]
        )

    monkeypatch.setattr(cli, "DSNMUIRAGPipeline", lambda config: SimpleNamespace(answer=answer))
    monkeypatch.setattr(sys, "argv", ["cli.py", "Pertanyaan", "--category", "fatwa", "--generator"])
    cli.main()
    assert calls == [("Pertanyaan", "fatwa", True, False)]
