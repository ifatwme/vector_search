"""Streamlit AppTest regressions, entirely offline."""
import socket
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_cold_app_and_rerender_need_no_service_model_or_credentials(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Cold UI attempted network")))
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "WEAVIATE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    at = AppTest.from_file(str(APP)).run(timeout=15)
    assert not at.exception
    assert "جستجوی کالا" in at.title[0].value
    assert any("هوش مصنوعی مولد نیست" in x.value for x in at.caption)
    assert at.session_state["messages"] == []
    assert at.session_state["results"] == []
    at.run()
    assert not at.exception
    assert at.session_state["messages"] == []


def test_search_form_runs_only_on_submit_and_uses_target_limit(monkeypatch):
    from types import SimpleNamespace as NS

    import vector_search.ui as ui
    calls = []
    monkeypatch.setattr(ui, "load_settings", lambda: NS(chat_backend="grounded"), raising=False)
    def fake_search(settings, **kwargs):
        calls.append(kwargs)
        return [{"pid": 7, "title": "کفش", "description": "پارچه", "url": "javascript:alert(1)", "image": ""}]
    monkeypatch.setattr(ui, "perform_search", fake_search, raising=False)
    at = AppTest.from_file(str(APP)).run()
    assert not calls
    at.selectbox(key="target").set_value("image")
    at.slider(key="result_limit").set_value(4).run()
    assert not calls
    at.text_input(key="query").set_value("کفش")
    at.button(key="search_submit").click().run()
    assert not at.exception
    assert calls == [{"text": "کفش", "image": None, "target": "image", "limit": 4}]
    assert at.session_state["results"][0]["pid"] == 7
    assert len(at.get("link_button")) == 0
    at.run()
    assert len(calls) == 1


def test_search_client_closed_on_success_and_failure_and_preflight_precedes_models(monkeypatch):
    import sys
    from threading import RLock
    from types import SimpleNamespace as NS

    import pytest

    import vector_search.search as retrieval
    import vector_search.ui as ui
    events = []
    client = NS(close=lambda: events.append("close"))
    monkeypatch.setitem(sys.modules, "vector_search.database", NS(connect=lambda settings: events.append("connect") or client, ensure_collection=lambda c, s, create: events.append(("schema", create))))
    monkeypatch.setattr(ui, "cached_embedder", lambda settings: (events.append("model") or object(), RLock()), raising=False)
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: events.append("search") or [{"pid": 7}])
    assert ui.perform_search(NS(), text="کفش", target="image", limit=3) == [{"pid": 7}]
    assert events == ["connect", ("schema", False), "model", "search", "close"]
    def fail(*a, **kw):
        raise RuntimeError("offline")
    monkeypatch.setattr(retrieval, "search", fail)
    with pytest.raises(RuntimeError):
        ui.perform_search(NS(), text="کفش")
    assert events[-1] == "close"
