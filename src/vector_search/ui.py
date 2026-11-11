"""Persian-first Streamlit UI. Cold rendering never creates backend resources.

Models and Weaviate are touched only when the user presses «جستجو» or sends a
chat message. Model weights are cached once per process (``st.cache_resource``);
search results and chat history are per browser session.
"""
from __future__ import annotations

import importlib
import json
from threading import RLock

import streamlit as st

from vector_search.chat import DISCLAIMER
from vector_search.config import quiet_third_party_warnings

WELCOME = "سلام! کالای مورد نظرت را در ستون جستجو پیدا کن، بعد هر سؤالی درباره نتایج داری از من بپرس."
CSS = """
<style>
[data-testid='stMain'], [data-testid='stSidebar'] {direction: rtl; text-align: right;}
[data-testid='stChatMessageContent'] {direction: rtl; text-align: right;}
.vs-meta {color: #6b7280; font-size: 0.85rem;}
</style>
"""


def load_settings():
    from vector_search.config import Settings
    return Settings.from_env()


@st.cache_resource(show_spinner=False)
def cached_embedder(settings):
    """One Embedder (≈1 GB of CLIPfa weights) shared by all sessions of this process."""
    from vector_search.embeddings import Embedder
    return Embedder(settings), RLock()


def perform_search(settings, *, text: str | None = None, image: bytes | None = None,
                   target: str = "text", limit: int = 10) -> list[dict]:
    """Short-lived client per query: no client survives a rerun or leaks between sessions."""
    database = importlib.import_module("vector_search.database")
    retrieval = importlib.import_module("vector_search.search")
    client = database.connect(settings)
    try:
        database.ensure_collection(client, settings, create=False)
        embedder, lock = cached_embedder(settings)
        with lock:
            return retrieval.search(client, settings, embedder, text=text, image=image,
                                    target=target, limit=limit)
    finally:
        client.close()


def safe_url(value: str) -> str | None:
    from urllib.parse import urlsplit
    try:
        if not isinstance(value, str) or any(ord(c) < 33 for c in value):
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            return None
        return value
    except ValueError:
        return None


def _image_bytes(settings, ref: str) -> bytes | None:
    if not ref:
        return None
    try:
        from vector_search.storage import ImageStore
        return ImageStore(settings).read(ref)
    except Exception:  # missing/corrupt cache entry: show the card without a picture
        return None


def _attributes(value) -> list[tuple[str, str]]:
    try:
        data = json.loads(value) if isinstance(value, str) and value else {}
    except ValueError:
        return []
    return [(str(k), str(v)) for k, v in data.items()][:8] if isinstance(data, dict) else []


def render_results(results: list[dict], settings=None) -> None:
    for rank, product in enumerate(results, 1):
        with st.container(border=True):
            picture, body = st.columns([1, 4])
            with picture:
                data = _image_bytes(settings, product.get("image", "")) if settings else None
                if data:
                    st.image(data, width="stretch")
                else:
                    st.caption("بدون تصویر")
            with body:
                st.markdown(f"**{rank}. {_escape(product.get('title') or 'بدون عنوان')}**")
                meta = f"شناسه کالا: {product.get('pid', '?')}"
                if isinstance(product.get("distance"), (int, float)):
                    meta += f" · شباهت: {1 - product['distance']:.3f}"
                st.caption(meta)
                description = str(product.get("description") or "")
                if description:
                    st.text(description[:600] + ("…" if len(description) > 600 else ""))
                specs = _attributes(product.get("attributes"))
                if specs:
                    with st.expander("مشخصات"):
                        for key, value in specs:
                            st.text(f"{key}: {value[:200]}")
                url = safe_url(product.get("url", ""))
                if url:
                    st.link_button("مشاهده در دیجی‌کالا", url)


def _escape(text: str) -> str:
    """Titles are catalog data: neutralise Markdown/HTML control characters."""
    for char in "\\`*_{}[]()<>#+!|~":
        text = str(text).replace(char, "\\" + char)
    return text


def _reset() -> None:
    from vector_search.chat import reset_session
    reset_session(st.session_state)


def _chat_panel(settings, backend: str) -> None:
    st.subheader("گفت‌وگو با دستیار فروش")
    if backend == "llm":
        st.caption(f"مدل زبانی محلی: {getattr(settings, 'llm_model', '')} (رایگان، بدون API ابری)")
    with st.container(height=520):
        with st.chat_message("assistant"):
            st.write(WELCOME)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        pending = st.session_state.get("pending_prompt")
        if pending:
            st.session_state.pending_prompt = None
            _answer(settings, backend)
    if prompt := st.chat_input("درباره کالاهای پیدا شده بپرس…", max_chars=500, key="chat_prompt"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
        st.rerun()


def _answer(settings, backend: str) -> None:
    from vector_search.chat import ChatService

    options = dict(temperature=st.session_state.get("temperature", 0.2),
                   top_p=st.session_state.get("top_p", 0.9),
                   max_output_tokens=st.session_state.get("max_tokens", 384))
    service = ChatService(settings)
    products = st.session_state.results
    with st.chat_message("assistant"):
        try:
            with st.spinner("در حال فکر کردن…"):
                text = st.write_stream(service.stream(st.session_state.messages, products, **options))
        except (RuntimeError, ValueError) as exc:
            st.warning(str(exc))
            # Retrieval-only fallback keeps the assistant useful when the LLM is down.
            text = ChatService(None).answer(st.session_state.messages, products, **options)
            st.write(text)
    st.session_state.messages.append({"role": "assistant", "content": str(text)})


def run() -> None:
    st.set_page_config(page_title="جستجوی کالا", page_icon="🔎", layout="wide")
    quiet_third_party_warnings()
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("results", [])
    try:
        settings = load_settings()
        settings_error = None
    except ValueError as exc:
        settings, settings_error = None, str(exc)
    backend = getattr(settings, "chat_backend", "grounded")

    st.title("جستجوی کالا و دستیار فروش دیجی‌کالا")
    if backend != "llm":
        st.caption(DISCLAIMER)
    else:
        st.caption("جستجوی برداری متنی و تصویری با CLIPfa و Weaviate · دستیار با مدل زبانی محلی")
    if getattr(settings, "app_notice", ""):
        st.info(settings.app_notice)
    if settings_error:
        st.error(f"پیکربندی نامعتبر است: {settings_error}")

    with st.sidebar:
        st.header("تنظیمات جستجو")
        target = st.selectbox("جستجو در", ("text", "image"), key="target",
                              format_func=lambda x: "عنوان کالاها" if x == "text" else "تصاویر کالاها")
        limit = st.slider("تعداد نتایج", 1, 20, 10, key="result_limit")
        st.divider()
        st.header("تنظیمات مدل زبانی")
        st.slider("temperature", 0.0, 1.5, 0.2, 0.05, key="temperature")
        st.slider("top_p", 0.05, 1.0, 0.9, 0.05, key="top_p")
        st.slider("حداکثر طول پاسخ (توکن)", 64, 1024, 384, 32, key="max_tokens")
        st.button("پاک کردن نتایج و گفت‌وگو", on_click=_reset, key="reset")

    search_col, chat_col = st.columns([3, 2], gap="large")
    with search_col:
        st.subheader("جستجوی کالا")
        st.radio("نوع ورودی", ("text", "image"), key="source", horizontal=True,
                 format_func=lambda x: "متن" if x == "text" else "تصویر")
        with st.form("product_search"):
            text = st.text_input("کالای دلخواه را توصیف کنید", key="query", max_chars=2000,
                                 placeholder="مثلاً: کفش ورزشی مردانه")
            upload = st.file_uploader("یا تصویر کالا (JPEG یا PNG)", type=["jpg", "jpeg", "png"],
                                      key="upload")
            submitted = st.form_submit_button("جستجو", key="search_submit", type="primary")
        if submitted:
            st.session_state.results = []
            st.session_state.search_done = False
            try:
                if settings is None:
                    raise ValueError("ابتدا پیکربندی را اصلاح کنید.")
                image = upload.getvalue() if upload is not None and st.session_state.source == "image" else None
                query = text.strip() if st.session_state.source == "text" else None
                if not query and image is None:
                    raise ValueError("متن جستجو یا تصویر کالا را وارد کنید.")
                if image is not None:
                    st.image(image, caption="تصویر جستجو", width=160)
                with st.spinner("در حال جستجو… (اولین جستجو مدل‌ها را بارگذاری می‌کند)"):
                    st.session_state.results = perform_search(settings, text=query, image=image,
                                                              target=target, limit=limit)
                st.session_state.search_done = True
            except ValueError as exc:
                st.error(str(exc))
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception:
                st.error("جستجو انجام نشد؛ اتصال Weaviate، آماده بودن ایندکس و دسترسی به مدل‌ها را "
                         "بررسی و دوباره «جستجو» را بزنید.")
        if st.session_state.get("search_done") and not st.session_state.results:
            st.info("کالایی پیدا نشد؛ ابتدا داده‌ها را ingest کنید یا جستجوی دیگری انجام دهید.")
        render_results(st.session_state.results, settings)
    with chat_col:
        _chat_panel(settings, backend)
