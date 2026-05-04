"""
Streamlit chat UI: Wikipedia RAG assistant (PRD §5.7).

Run: ``streamlit run app.py``
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

import config
from wiki_rag import cache
from wiki_rag import generator
from wiki_rag import retriever
from wiki_rag import vectorstore

_EMPTY_RESULT: dict = {
    "query": "",
    "query_type": "both",
    "entity_filter": None,
    "chunks": [],
    "sources": [],
    "context": "",
    "is_comparison": False,
    "entities": [],
}

_COMPARISON_QUERY_TYPES = frozenset(
    {"person_person", "place_place", "person_place", "single_type"}
)

_THINKING_HTML = """
<div style="padding:12px; background:#f0f2f6;
border-radius:12px; border-left:3px solid #667eea;">
🧠 <em>Thinking...</em>
</div>
"""

_UI_CSS = """
<style>
/* App chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main content: white surface, dark type */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}
section.main {
    background-color: #ffffff !important;
}
section.main .block-container {
    padding-top: 0.35rem;
    background-color: #ffffff !important;
}
section.main,
section.main .stMarkdown,
section.main p,
section.main h1,
section.main h2,
section.main h3,
section.main h4,
section.main li {
    color: #111827 !important;
}
section.main .stCaption {
    color: #4b5563 !important;
}
[data-testid="stChatMessage"] {
    color: #111827 !important;
}

/* ----- Sidebar: light gray surface ----- */
section[data-testid="stSidebar"] {
    background: #f0f2f6 !important;
    border-right: 1px solid #e5e7eb;
    box-shadow: 2px 0 12px rgba(0, 0, 0, 0.04);
}
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.65rem !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 0.5rem !important;
    padding-left: 0.85rem !important;
    padding-right: 0.85rem !important;
}

section[data-testid="stSidebar"] .wikirag-sb-block {
    margin: 1rem 0 0.65rem 0;
}
section[data-testid="stSidebar"] .wikirag-sb-block:first-of-type {
    margin-top: 0.35rem;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] span {
    color: #31333F;
}

section[data-testid="stSidebar"] .wikirag-sidebar-head {
    color: #111827 !important;
}

section[data-testid="stSidebar"] hr {
    margin: 1rem 0;
    border: none;
    border-top: 1px solid #e5e7eb;
}

/* New Chat, Clear all, Recent rows — white panels, black text */
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
section[data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
    width: 100%;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.65rem !important;
    transition: background-color 0.12s ease, border-color 0.12s ease;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
section[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
    background-color: #f9fafb !important;
    color: #000000 !important;
    border-color: #d1d5db !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
    justify-content: flex-start !important;
    text-align: left !important;
    font-size: 0.88rem !important;
    font-weight: 450 !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    font-weight: 600 !important;
}

/* Recent chat row: delete icon — borderless, fade in on row hover */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(2)) {
    border-radius: 8px;
    padding: 1px 0;
    margin: 2px 0;
    transition: background-color 0.15s ease;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(2)):hover {
    background-color: rgba(0, 0, 0, 0.03);
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(2)) > div:nth-child(1) .stButton > button[kind="tertiary"] {
    width: 100% !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(2)) > div:nth-child(2) .stButton > button {
    width: 2rem !important;
    min-width: 2rem !important;
    max-width: 2.25rem !important;
    padding: 0.3rem 0 !important;
    margin: 0 auto !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #4b5563 !important;
    opacity: 0;
    transition: opacity 0.2s ease-in-out, background-color 0.15s ease, color 0.15s ease;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(2)):hover > div:nth-child(2) .stButton > button {
    opacity: 1;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(2)) > div:nth-child(2) .stButton > button:hover {
    background-color: rgba(0, 0, 0, 0.06) !important;
    color: #111827 !important;
}
/* Footer & meta */
section[data-testid="stSidebar"] .stCaption {
    color: #31333F !important;
    font-size: 0.72rem !important;
    line-height: 1.45 !important;
}

/* Settings */
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #31333F !important;
}

/* Pipeline (orchestrator) — light sidebar */
@keyframes wikirag-pipeline-pulse {
    0%, 100% { opacity: 0.45; transform: scale(0.98); }
    33% { opacity: 1; transform: scale(1); color: #111827; }
    66% { opacity: 0.6; transform: scale(0.99); }
}
.wikirag-pipeline {
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #4b5563;
    margin: 0.5rem 0 0.25rem;
}
.wikirag-pipeline-track {
    height: 3px;
    border-radius: 999px;
    background: #e5e7eb;
    overflow: hidden;
    margin: 8px 0 6px;
}
.wikirag-pipeline-fill {
    height: 100%;
    width: 40%;
    border-radius: 999px;
    background: linear-gradient(90deg, #9ca3af, #d1d5db, #9ca3af);
    background-size: 200% 100%;
    animation: wikirag-pipeline-slide 1.2s ease-in-out infinite;
}
@keyframes wikirag-pipeline-slide {
    0% { transform: translateX(-30%); background-position: 0% 50%; }
    100% { transform: translateX(220%); background-position: 100% 50%; }
}
.wikirag-pipeline-steps {
    display: flex;
    justify-content: space-between;
    gap: 6px;
}
.wikirag-pipeline-steps span {
    flex: 1;
    text-align: center;
    font-size: 0.65rem;
    color: #6b7280;
    animation: wikirag-pipeline-pulse 2.4s ease-in-out infinite;
}
.wikirag-pipeline-steps span:nth-child(1) { animation-delay: 0s; }
.wikirag-pipeline-steps span:nth-child(2) { animation-delay: 0.8s; }
.wikirag-pipeline-steps span:nth-child(3) { animation-delay: 1.6s; }

/* Router pill badge (injected in chat) */
.wikirag-router-badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 6px 0 2px;
    border: 1px solid rgba(99, 102, 241, 0.35);
    background: rgba(99, 102, 241, 0.08);
    color: #4338ca;
}
.wikirag-router-badge.place {
    border-color: rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.1);
    color: #047857;
}
.wikirag-router-badge.both {
    border-color: rgba(245, 158, 11, 0.4);
    background: rgba(245, 158, 11, 0.1);
    color: #b45309;
}

/* Chat input (main) */
.stChatInput {
    border-radius: 12px;
}
.stChatMessage {
    border-radius: 12px;
    margin-bottom: 8px;
}
</style>
"""


def create_new_chat() -> str:
    chat_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().strftime("%H:%M"),
    }
    st.session_state.chat_counter += 1
    st.session_state.active_chat_id = chat_id
    return chat_id


def get_active_messages() -> list:
    return st.session_state.chats[st.session_state.active_chat_id]["messages"]


def update_chat_title(chat_id: str, first_message: str) -> None:
    title = (
        first_message[:28] + "..."
        if len(first_message) > 28
        else first_message
    )
    st.session_state.chats[chat_id]["title"] = title


def _chat_ids_newest_first() -> list[str]:
    def _key(cid: str) -> int:
        try:
            return int(cid.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    return sorted(st.session_state.chats.keys(), key=_key, reverse=True)


def _delete_chat(chat_id: str) -> None:
    was_active = st.session_state.active_chat_id == chat_id
    del st.session_state.chats[chat_id]
    if not st.session_state.chats:
        create_new_chat()
        return
    if was_active:
        st.session_state.active_chat_id = _chat_ids_newest_first()[0]


def _format_chunk_count(n: int) -> str:
    return f"{n:,}"


def _init_session_state() -> None:
    """Bootstrap chat container once per Streamlit session (not on every rerun)."""
    if "_wiki_rag_initialized" not in st.session_state:
        st.session_state._wiki_rag_initialized = True
        if "chats" not in st.session_state:
            st.session_state.chats = {}
        if "chat_counter" not in st.session_state:
            chats = st.session_state.chats
            max_n = -1
            for cid in chats:
                try:
                    max_n = max(max_n, int(cid.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
            st.session_state.chat_counter = (max_n + 1) if max_n >= 0 else 0
        if not st.session_state.chats:
            create_new_chat()
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False
    if "prefilled_query" not in st.session_state:
        st.session_state.prefilled_query = ""


def _ensure_active_chat_valid() -> None:
    """If active id is missing or stale, point at an existing chat or create one."""
    if not st.session_state.chats:
        create_new_chat()
        return
    aid = st.session_state.get("active_chat_id")
    if not aid or aid not in st.session_state.chats:
        st.session_state.active_chat_id = _chat_ids_newest_first()[0]


def _query_type_badge_class(qt: str) -> str:
    if qt == "place":
        return "place"
    if qt == "both":
        return "both"
    return "person"


def _query_type_badge_label(qt: str) -> str:
    if qt == "person_person":
        return "Comparison: People"
    if qt == "place_place":
        return "Comparison: Places"
    if qt == "person_place":
        return "Comparison: Mixed"
    if qt == "single_type":
        return "Comparison"
    if qt == "person":
        return "Target: Person"
    if qt == "place":
        return "Target: Place"
    if qt == "both":
        return "Target: Mixed"
    return "Target: —"


def _router_badge_html(qt: str) -> str:
    if qt in _COMPARISON_QUERY_TYPES:
        cls = "both"
    else:
        cls = _query_type_badge_class(qt)
    label = _query_type_badge_label(qt)
    return (
        f'<div class="wikirag-router-badge {cls}">{label}</div>'
    )


def _render_source_cards(
    sources: list,
    *,
    is_comparison: bool = False,
    entities: list[str] | None = None,
) -> None:
    if not sources:
        return
    ent_list = list(entities or [])
    if is_comparison and ent_list:
        labels = " | ".join(ent_list)
        st.markdown(
            f'<p style="margin:0 0 10px 0;font-size:0.9rem;color:#111827;">'
            f"<strong>⚖️ Comparing:</strong> {labels}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    cols = st.columns(len(sources))
    for i, source in enumerate(sources):
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                  padding: 8px 12px;
                  background: #f8f9fa;
                  border-radius: 8px;
                  border: 1px solid #e9ecef;
                  font-size: 12px;
                  text-align: center;
                ">
                📄 {source}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_assistant_caption(msg: dict) -> None:
    n_src = len(msg.get("sources") or [])
    qt = msg.get("query_type") or ""
    is_comp = bool(msg.get("is_comparison")) or qt in _COMPARISON_QUERY_TYPES

    if is_comp:
        extra = " · ⚡ cached response" if msg.get("from_cache") else ""
        st.caption(f"⚖️ comparison · 📎 {n_src} sources{extra}")
    else:
        st.markdown(_router_badge_html(qt), unsafe_allow_html=True)
        via = (
            "⚡ cached response"
            if msg.get("from_cache")
            else f"⚡ via {config.OLLAMA_LLM_MODEL}"
        )
        st.caption(f"📎 {n_src} sources · {via}")


def _render_message(msg: dict) -> None:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant":
            _render_assistant_caption(msg)
            show_src = st.session_state.show_sources and (
                msg.get("sources") or msg.get("chunks")
            )
            if show_src:
                _render_source_cards(
                    list(msg.get("sources") or []),
                    is_comparison=bool(msg.get("is_comparison")),
                    entities=list(msg.get("entities") or []),
                )
                chunks = msg.get("chunks") or []
                if chunks:
                    with st.expander("Passage excerpts"):
                        for ch in chunks:
                            label = f"{ch['title']} (chunk #{ch['chunk_index']})"
                            with st.expander(label):
                                st.text(ch.get("text", ""))


def _sidebar_section_label(text: str) -> None:
    st.sidebar.markdown(
        f'<div class="wikirag-sb-block"><p class="wikirag-sidebar-head" style="margin:0 0 8px 0;font-size:0.68rem;'
        f'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#31333F;">'
        f"{text}</p></div>",
        unsafe_allow_html=True,
    )


def _render_sidebar_orchestrator(pending_response: bool, pipeline_eligible: bool) -> None:
    if not pending_response or not pipeline_eligible:
        return
    st.sidebar.markdown(
        """
        <div class="wikirag-pipeline">Processing</div>
        <div class="wikirag-pipeline-track"><div class="wikirag-pipeline-fill"></div></div>
        <div class="wikirag-pipeline-steps">
          <span>Router</span><span>Retrieval</span><span>Generation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_system_block() -> None:
    ollama_ok = generator.check_ollama_running()

    with st.sidebar.container():
        _sidebar_section_label("System")
        if ollama_ok:
            st.sidebar.markdown(
                '<p style="margin:0 0 6px 0;font-size:0.88rem;color:#111827;">'
                "🟢 &nbsp;Ollama Online</p>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                '<p style="margin:0 0 6px 0;font-size:0.88rem;color:#111827;">'
                "🔴 &nbsp;Ollama Offline</p>",
                unsafe_allow_html=True,
            )
        try:
            stats = vectorstore.get_stats()
            total = int(stats.get("total", 0))
            st.sidebar.markdown(
                f'<p style="margin:0;font-size:0.88rem;color:#111827;">'
                f"📚 &nbsp;<strong>{_format_chunk_count(total)}</strong> Documents Indexed</p>",
                unsafe_allow_html=True,
            )
            if total == 0:
                st.sidebar.caption("Run `python ingest_pipeline.py` to ingest.")
        except Exception as err:  # noqa: BLE001
            st.sidebar.markdown(
                '<p style="color:#fca5a5;font-size:0.85rem;margin:0;">📚 Index unavailable</p>',
                unsafe_allow_html=True,
            )
            st.sidebar.caption(str(err))

        if config.CACHE_ENABLED:
            try:
                cstats = cache.get_cache_stats()
                nc = int(cstats.get("total_cached", 0))
                st.sidebar.markdown(
                    f'<p style="margin:8px 0 0 0;font-size:0.88rem;color:#111827;">'
                    f"💾 &nbsp;<strong>{nc}</strong> responses cached</p>",
                    unsafe_allow_html=True,
                )
            except Exception:  # noqa: BLE001
                st.sidebar.caption("Cache stats unavailable.")
            if st.sidebar.button(
                "Clear cache",
                use_container_width=True,
                type="secondary",
            ):
                cache.clear_cache()
                st.rerun()


def _render_sidebar_footer() -> None:
    st.sidebar.markdown(
        '<hr style="border:none;border-top:1px solid #e5e7eb;margin:1rem 0 0.35rem;" />',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "Native Python implementation | Custom metadata filtering"
    )


def _render_sidebar(msgs: list[dict]) -> None:
    pending_response = bool(msgs and msgs[-1].get("role") == "user")
    ollama_ok = generator.check_ollama_running()
    try:
        kb_ok = bool(vectorstore.get_stats().get("total", 0) > 0)
    except Exception:  # noqa: BLE001
        kb_ok = False
    pipeline_eligible = ollama_ok and kb_ok

    st.sidebar.markdown(
        """
        <div class="wikirag-sb-block wikirag-sidebar-head" style="margin-bottom:12px;">
          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;
          text-transform:uppercase;color:#31333F;margin-bottom:6px;">Workspace</div>
          <div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.02em;
          color:#111827;line-height:1.2;">WikiRAG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("+ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()

    st.sidebar.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    _render_sidebar_orchestrator(pending_response, pipeline_eligible)

    _sidebar_section_label("Recent")
    for cid in _chat_ids_newest_first():
        chat = st.session_state.chats[cid]
        title = chat["title"]
        active = cid == st.session_state.active_chat_id
        label = f"→  {title}" if active else title
        c_open, c_del = st.sidebar.columns([0.85, 0.15])
        with c_open:
            if st.button(
                label,
                key=f"open_{cid}",
                use_container_width=True,
                type="tertiary",
            ):
                st.session_state.active_chat_id = cid
                st.rerun()
        with c_del:
            if st.button(
                "✕",
                key=f"del_{cid}",
                help="Delete chat",
                use_container_width=True,
                type="tertiary",
            ):
                _delete_chat(cid)
                st.rerun()

    st.sidebar.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    _render_sidebar_system_block()

    st.sidebar.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    _sidebar_section_label("Settings")
    st.sidebar.toggle("Show sources", key="show_sources")
    if st.sidebar.button(
        "Clear all chats",
        use_container_width=True,
        type="secondary",
        help="Remove every conversation and start fresh",
    ):
        st.session_state.chats = {}
        st.session_state.chat_counter = 0
        create_new_chat()
        st.rerun()

    _render_sidebar_footer()


def _render_empty_state() -> None:
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 1rem;">
        <div style="font-size: 3rem; line-height: 1.2;">📚</div>
        <h2 style="margin: 0.5rem 0 0.25rem;">Wikipedia RAG Assistant</h2>
        <p style="color: #666; margin: 0;">Ask me anything about famous people and places</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    quick = [
        "Who was Albert Einstein?",
        "Where is the Eiffel Tower?",
        "What did Marie Curie discover?",
        "Compare Messi and Ronaldo",
    ]
    bc1, bc2 = st.columns(2)
    pairs = [(bc1, quick[0], 0), (bc1, quick[1], 1), (bc2, quick[2], 2), (bc2, quick[3], 3)]
    for col, q, qi in pairs:
        with col:
            if st.button(q, key=f"quick_q_{qi}", use_container_width=True):
                st.session_state.prefilled_query = q
                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="WikiRAG Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_UI_CSS, unsafe_allow_html=True)
    _init_session_state()
    cache.init_cache()
    _ensure_active_chat_valid()
    msgs = get_active_messages()
    _render_sidebar(msgs)
    _ensure_active_chat_valid()

    msgs = get_active_messages()
    if not msgs:
        _render_empty_state()
    else:
        for msg in msgs:
            _render_message(msg)

    pref = (st.session_state.get("prefilled_query") or "").strip()
    if pref:
        st.session_state.prefilled_query = ""

    prompt = st.chat_input("Ask about a famous person or place...")
    if pref:
        prompt = pref

    if prompt:
        active_id = st.session_state.active_chat_id
        if (
            st.session_state.chats[active_id]["title"] == "New Chat"
            and not st.session_state.chats[active_id]["messages"]
        ):
            update_chat_title(active_id, prompt)
        get_active_messages().append({"role": "user", "content": prompt})
        st.rerun()

    msgs = get_active_messages()
    if msgs and msgs[-1]["role"] == "user":
        last_prompt = msgs[-1]["content"]
        hist = list(msgs[:-1][-6:])
        with st.chat_message("assistant"):
            think_ph = st.empty()
            answer = ""
            result: dict = dict(_EMPTY_RESULT)
            caption_rendered = False
            cache_hit = False

            if not generator.check_ollama_running():
                think_ph.empty()
                answer = (
                    "Ollama is not running. Start it with: `ollama serve`, then try again."
                )
                result = dict(_EMPTY_RESULT)
                st.error(answer)
            else:
                try:
                    stats = vectorstore.get_stats()
                    if stats.get("total", 0) == 0:
                        think_ph.empty()
                        answer = (
                            "The knowledge base is empty. Run "
                            "`python ingest_pipeline.py` to fetch Wikipedia pages "
                            "and build the index, then refresh this app."
                        )
                        result = dict(_EMPTY_RESULT)
                        st.warning(answer)
                    else:
                        no_chunks = False
                        # Order: cache → (miss) retrieve_and_format → generate → save
                        cached = (
                            cache.get_cached_response(last_prompt)
                            if config.CACHE_ENABLED
                            else None
                        )
                        if cached:
                            think_ph.empty()
                            cache_hit = True
                            answer = cached["answer"]
                            result = dict(_EMPTY_RESULT)
                            result["sources"] = list(cached["sources"])
                            result["query_type"] = cached["query_type"]
                            result["chunks"] = []
                            qtc = result["query_type"] or ""
                            is_comp_cached = qtc in _COMPARISON_QUERY_TYPES
                            result["is_comparison"] = is_comp_cached
                            result["entities"] = (
                                list(cached["sources"]) if is_comp_cached else []
                            )
                            st.markdown(answer)
                            _render_assistant_caption(
                                {
                                    "sources": result["sources"],
                                    "query_type": result["query_type"],
                                    "from_cache": True,
                                    "is_comparison": is_comp_cached,
                                }
                            )
                            if st.session_state.show_sources and result["sources"]:
                                _render_source_cards(
                                    result["sources"],
                                    is_comparison=is_comp_cached,
                                    entities=result["entities"],
                                )
                            caption_rendered = True
                        else:
                            think_ph.markdown(
                                _THINKING_HTML,
                                unsafe_allow_html=True,
                            )
                            try:
                                result = retriever.retrieve_and_format(
                                    last_prompt, chat_history=hist
                                )
                                ctx_ok = bool(
                                    (result.get("context") or "").strip()
                                )
                                if not ctx_ok:
                                    no_chunks = True
                                    answer = (
                                        "I don't have any indexed passages to answer from. "
                                        "After ingestion completes, restart the app and try again."
                                    )
                                else:
                                    no_chunks = False
                                    answer = generator.generate_answer(
                                        last_prompt,
                                        result["context"],
                                        chat_history=hist,
                                        is_comparison=bool(
                                            result.get("is_comparison")
                                        ),
                                        entities=list(
                                            result.get("entities") or []
                                        ),
                                    )
                                    if config.CACHE_ENABLED:
                                        cache.save_to_cache(
                                            last_prompt,
                                            answer,
                                            list(result.get("sources") or []),
                                            str(
                                                result.get("query_type", "") or ""
                                            ),
                                        )
                            except RuntimeError as err:
                                think_ph.empty()
                                answer = f"Could not complete search: {err}"
                                result = dict(_EMPTY_RESULT)
                                st.error(answer)
                            except Exception as err:  # noqa: BLE001
                                think_ph.empty()
                                answer = f"Something went wrong: {err}"
                                result = dict(_EMPTY_RESULT)
                                st.error(answer)
                            else:
                                think_ph.empty()
                                if no_chunks:
                                    st.warning(
                                        "No passages were retrieved (empty context). "
                                        "Run `python test_pipeline.py` to verify the index, or "
                                        "`python ingest_pipeline.py` if the vector store is empty. "
                                        "Use the **same project folder and venv** for Streamlit and ingest."
                                    )
                                    st.markdown(answer)
                                else:
                                    st.markdown(answer)
                                _render_assistant_caption(
                                    {
                                        "sources": list(result.get("sources") or []),
                                        "query_type": result.get("query_type", ""),
                                        "from_cache": False,
                                        "is_comparison": bool(
                                            result.get("is_comparison")
                                        ),
                                    }
                                )
                                if st.session_state.show_sources and not no_chunks:
                                    src = list(result.get("sources") or [])
                                    chs = result.get("chunks") or []
                                    if src or chs:
                                        _render_source_cards(
                                            src,
                                            is_comparison=bool(
                                                result.get("is_comparison")
                                            ),
                                            entities=list(
                                                result.get("entities") or []
                                            ),
                                        )
                                    if chs:
                                        with st.expander("Passage excerpts"):
                                            st.caption(
                                                f"Query type: `{result.get('query_type', '')}`"
                                            )
                                            for ch in chs:
                                                label = (
                                                    f"{ch['title']} "
                                                    f"(chunk #{ch['chunk_index']})"
                                                )
                                                with st.expander(label):
                                                    st.text(ch.get("text", ""))
                                caption_rendered = True
                except RuntimeError as err:
                    think_ph.empty()
                    answer = f"Could not complete search: {err}"
                    result = dict(_EMPTY_RESULT)
                    st.error(answer)
                except Exception as err:  # noqa: BLE001
                    think_ph.empty()
                    answer = f"Something went wrong: {err}"
                    result = dict(_EMPTY_RESULT)
                    st.error(answer)

            if answer and not caption_rendered:
                qt = result.get("query_type", "") or ""
                is_comp_fallback = qt in _COMPARISON_QUERY_TYPES
                _render_assistant_caption(
                    {
                        "sources": list(result.get("sources") or []),
                        "query_type": qt,
                        "from_cache": cache_hit,
                        "is_comparison": bool(result.get("is_comparison"))
                        or is_comp_fallback,
                    }
                )

            get_active_messages().append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": list(result.get("sources") or []),
                    "query_type": result.get("query_type", ""),
                    "chunks": list(result.get("chunks") or []),
                    "from_cache": cache_hit,
                    "is_comparison": bool(result.get("is_comparison")),
                    "entities": list(result.get("entities") or []),
                }
            )
        st.rerun()


if __name__ == "__main__":
    main()
