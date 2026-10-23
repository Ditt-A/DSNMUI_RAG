"""Exercise the API contract with a fake pipeline; never load model weights."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app_rag_api as api


@pytest.fixture
def preview_client(monkeypatch):
    def forbid_pipeline(*args, **kwargs):
        raise AssertionError("Preview must not initialize the real pipeline")

    monkeypatch.setattr(api, "LOAD_PIPELINE", False)
    # Even accidental preload settings must not override preview mode.
    monkeypatch.setattr(api, "PRELOAD_BGE", True)
    monkeypatch.setattr(api, "PRELOAD_VL", True)
    monkeypatch.setattr(api, "DSNMUIRAGPipeline", forbid_pipeline)
    monkeypatch.setattr(api, "API_KEY", "test-key")
    monkeypatch.setattr(api, "pipe", None)
    with TestClient(api.app) as client:
        yield client


def test_preview_health_does_not_initialize_pipeline(preview_client):
    assert preview_client.get("/").status_code == 200
    health = preview_client.get("/health").json()
    assert health["status"] == "disabled"
    assert health["pipeline_enabled"] is False
    assert health["pipeline_loaded"] is False
    assert health["vl_loaded"] is False


def test_preview_checks_key_before_pipeline(preview_client):
    assert preview_client.post("/rag/ask", json={"query": "test"}).status_code == 401
    response = preview_client.post(
        "/rag/ask", headers={"x-api-key": "test-key"}, json={"query": "test"}
    )
    assert response.status_code == 503
    for path, method in [("/summary", "get"), ("/warmup/vl", "post")]:
        response = getattr(preview_client, method)(path, headers={"x-api-key": "test-key"})
        assert response.status_code == 503


@pytest.fixture
def fake_client(monkeypatch):
    calls = []

    def answer(query, **kwargs):
        calls.append((query, kwargs))
        return SimpleNamespace(
            answer="Jawaban contoh [S1]",
            results=[{
                "rank_final": 1,
                "payload": {
                    "category": "fatwa", "nomor": "06", "title": "Istishna",
                    "page": 2, "text_original": "Referensi contoh",
                },
            }],
            support_info={"support_level": "high", "support_score": 0.9},
            generation_meta={"generation_mode": "fallback"},
            timing={"retrieval_total_ms": 1.0},
            context="test context", prompt="test prompt", item={"query": query},
        )

    fake_pipe = SimpleNamespace(
        answer=answer,
        summary=lambda: {"chunks": 1},
        visual=SimpleNamespace(model=None, processor=None),
    )
    monkeypatch.setattr(api, "LOAD_PIPELINE", True)
    monkeypatch.setattr(api, "PRELOAD_BGE", False)
    monkeypatch.setattr(api, "PRELOAD_VL", False)
    monkeypatch.setattr(api, "DSNMUIRAGPipeline", lambda config: fake_pipe)
    monkeypatch.setattr(api, "API_KEY", "test-key")
    monkeypatch.setattr(api, "pipe", None)
    with TestClient(api.app) as client:
        yield client, calls


@pytest.mark.parametrize("path", ["/rag/ask", "/ask"])
def test_chat_response_and_filter_contract(fake_client, path):
    client, calls = fake_client
    response = client.post(path, headers={"x-api-key": "test-key"}, json={
        "query": "  Pertanyaan UMKM  ", "expected_focus": "akad",
        "category_filter": "fatwa", "use_visual_dalil": False,
        "use_visual_as_generator": False,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Jawaban contoh [S1]"
    assert body["sources"][0]["source"] == "S1"
    assert body["sources"][0]["page"] == 2
    assert body["support_info"]["support_score"] == 0.9
    assert body["debug"] is None
    assert calls == [("Pertanyaan UMKM", {
        "expected_focus": "akad", "category_filter": "fatwa",
        "use_visual_dalil": False, "use_visual_as_generator": False,
    })]


def test_empty_chat_does_not_call_inference(fake_client):
    client, calls = fake_client
    response = client.post("/rag/ask", headers={"x-api-key": "test-key"}, json={"query": "   "})
    assert response.status_code == 400
    assert calls == []
