"""
app.py — Streamlit entry point for the AI Powered Chatbot.

Run:  streamlit run app.py

Architecture highlights:
  • st.cache_resource in each model module → models load ONCE, never on re-run
  • st.write_stream()  → live token streaming from Groq
  • st.session_state   → persistent chat history across re-runs
  • Sidebar shows real-time NLP insights (intent / sentiment / entities)
"""

import streamlit as st
from src.chatbot import ChatBot
from src.utils import init_session_defaults, to_groq_history, format_entities_md, clean_text
from src.config import config

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────
init_session_defaults(st.session_state, {
    "messages":  [],        # [{"role": "user"|"assistant", "content": "..."}]
    "nlp_meta":  {},        # latest NLP analysis for sidebar
})

# ── Singleton chatbot (no model inside — models are cached per-module) ─────────
@st.cache_resource
def get_chatbot() -> ChatBot:
    return ChatBot()

bot = get_chatbot()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — NLP Insights
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔍 NLP Insights")
    st.caption("Live analysis of the last message")

    meta = st.session_state.nlp_meta
    if meta:
        st.metric("🎯 Intent",    meta.get("intent", "—"))
        st.metric("💬 Sentiment", meta.get("sentiment", "—"))

        st.markdown("**📌 Entities detected**")
        st.markdown(format_entities_md(meta.get("entities", [])))
    else:
        st.info("Send a message to see NLP analysis here.")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.nlp_meta  = {}
        st.rerun()

    st.divider()
    st.markdown(
        f"**Model:** `{config.GROQ_MODEL}`  \n"
        f"**Device:** `{config.TORCH_DEVICE}`"
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
st.title(f"{config.APP_ICON} {config.APP_TITLE}")

# ── Render existing chat history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Type your message…"):

    # Show and store user message immediately
    user_input = clean_text(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build Groq-compatible history (role + content only)
    groq_history = to_groq_history(st.session_state.messages[:-1])  # exclude current msg

    # Stream assistant response
    with st.chat_message("assistant"):
        gen, nlp_meta = bot.process_stream(user_input, groq_history)
        # st.write_stream consumes the generator and renders tokens live
        full_response = st.write_stream(gen)

    # Persist assistant response and NLP metadata
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.nlp_meta = nlp_meta

    # Re-run so sidebar refreshes with latest NLP insights
    st.rerun()