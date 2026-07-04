import streamlit as st
from src.config import config
from src.hf_auth import setup_hf_auth, get_model_info
from src.chatbot import ChatBot
from src.utils import init_session_defaults, to_groq_history, format_entities_md, clean_text

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background — soft white gradient ───────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #F0F4FF 0%, #FAF5FF 40%, #F0FDF4 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit branding only ────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Sidebar toggle arrow — always visible ───────────────────────────────── */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 880px !important;
}

/* ── Page title ──────────────────────────────────────────────────────────── */
h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #6C63FF, #A855F7, #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8F4FF 100%) !important;
    border-right: 1.5px solid #E9D8FD !important;
    box-shadow: 4px 0 20px rgba(108, 99, 255, 0.06) !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    max-width: 100% !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #6C63FF !important;
    -webkit-text-fill-color: #6C63FF !important;
}
[data-testid="stSidebar"] .stCaption {
    color: #A78BFA !important;
    font-size: 0.72rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #6B7280 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] strong {
    color: #374151 !important;
}
[data-testid="stSidebar"] code {
    font-size: 0.7rem !important;
    background: #EDE9FE !important;
    color: #6C63FF !important;
    border: 1px solid #DDD6FE !important;
}

/* ── NLP Metric Cards ────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1.5px solid #E9D8FD !important;
    border-radius: 14px !important;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.6rem !important;
    box-shadow: 0 2px 12px rgba(108, 99, 255, 0.08) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: #A78BFA !important;
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.15) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] p {
    color: #A78BFA !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.9px !important;
}
[data-testid="stMetricValue"] {
    color: #1F2937 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}

/* ── Info box ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stAlert {
    background: #F5F3FF !important;
    border: 1.5px dashed #C4B5FD !important;
    border-radius: 12px !important;
    color: #7C3AED !important;
}

/* ── Chat Messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #6C63FF, #A855F7) !important;
    border-radius: 50% !important;
    min-width: 36px !important;
    height: 36px !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #059669, #34D399) !important;
    border-radius: 10px !important;
    min-width: 36px !important;
    height: 36px !important;
}
[data-testid="stChatMessageContent"] {
    background: #FFFFFF !important;
    border: 1.5px solid #EDE9FE !important;
    border-radius: 18px !important;
    border-top-left-radius: 4px !important;
    padding: 0.9rem 1.2rem !important;
    color: #1F2937 !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    max-width: 85% !important;
    box-shadow: 0 2px 16px rgba(108, 99, 255, 0.07) !important;
}

/* ── Chat Input ──────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 2px solid #DDD6FE !important;
    border-radius: 18px !important;
    color: #1F2937 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 2px 16px rgba(108, 99, 255, 0.08) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #A78BFA !important;
    box-shadow: 0 0 0 4px rgba(167, 139, 250, 0.15) !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #6C63FF, #A855F7) !important;
    border: none !important;
    border-radius: 12px !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {
    border-color: #EDE9FE !important;
    margin: 0.8rem 0 !important;
}

/* ── Clear Chat Button ───────────────────────────────────────────────────── */
.stButton button {
    background: #FFF1F2 !important;
    border: 1.5px solid #FECDD3 !important;
    color: #E11D48 !important;
    border-radius: 12px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    background: #FFE4E6 !important;
    border-color: #FDA4AF !important;
    box-shadow: 0 2px 12px rgba(225, 29, 72, 0.15) !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #FAFAFA !important;
    border: 1.5px solid #EDE9FE !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #7C3AED !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}

/* ── Code ────────────────────────────────────────────────────────────────── */
code {
    background: #EDE9FE !important;
    border: 1px solid #DDD6FE !important;
    border-radius: 6px !important;
    color: #6C63FF !important;
    font-size: 0.78rem !important;
    padding: 0.2rem 0.45rem !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F5F3FF; }
::-webkit-scrollbar-thumb {
    background: #C4B5FD;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: #A78BFA; }
</style>
""", unsafe_allow_html=True)


# ── Paste this to replace your st.markdown subtitle and suggestion cards ───────

# Subtitle under title
st.markdown(
    '<p style="color:#9CA3AF; font-size:0.88rem; '
    'margin-top:-0.5rem; margin-bottom:1rem;">'
    'Powered by your NLP models &nbsp;·&nbsp; '
    'Ask about orders, returns, products &amp; more'
    '</p>',
    unsafe_allow_html=True,
)

# Empty state suggestion cards — light theme version
if not st.session_state.messages:
    st.markdown(
        """
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:2rem 0;">
            <div style="background:#FFFFFF; border:1.5px solid #DDD6FE;
                        border-radius:16px; padding:1.1rem 1.2rem;
                        box-shadow:0 2px 12px rgba(108,99,255,0.08);">
                <p style="color:#6C63FF; font-size:0.85rem; font-weight:700; margin:0 0 5px;">
                    📦 Track Order</p>
                <p style="color:#6B7280; font-size:0.8rem; margin:0;">
                    Where is my order #45678?</p>
            </div>
            <div style="background:#FFFFFF; border:1.5px solid #BBF7D0;
                        border-radius:16px; padding:1.1rem 1.2rem;
                        box-shadow:0 2px 12px rgba(5,150,105,0.08);">
                <p style="color:#059669; font-size:0.85rem; font-weight:700; margin:0 0 5px;">
                    ↩️ Return &amp; Refund</p>
                <p style="color:#6B7280; font-size:0.8rem; margin:0;">
                    I want to return a damaged item</p>
            </div>
            <div style="background:#FFFFFF; border:1.5px solid #BFDBFE;
                        border-radius:16px; padding:1.1rem 1.2rem;
                        box-shadow:0 2px 12px rgba(59,130,246,0.08);">
                <p style="color:#2563EB; font-size:0.85rem; font-weight:700; margin:0 0 5px;">
                    ❓ Product Info</p>
                <p style="color:#6B7280; font-size:0.8rem; margin:0;">
                    What are the specs of this product?</p>
            </div>
            <div style="background:#FFFFFF; border:1.5px solid #FDE68A;
                        border-radius:16px; padding:1.1rem 1.2rem;
                        box-shadow:0 2px 12px rgba(245,158,11,0.08);">
                <p style="color:#D97706; font-size:0.85rem; font-weight:700; margin:0 0 5px;">
                    💳 Payment Issue</p>
                <p style="color:#6B7280; font-size:0.8rem; margin:0;">
                    My payment failed but was deducted</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── HuggingFace auth ───────────────────────────────────────────────────────────
setup_hf_auth()

# ── Session state ──────────────────────────────────────────────────────────────
init_session_defaults(st.session_state, {
    "messages": [],
    "nlp_meta": {},
})


# ── Singleton chatbot ──────────────────────────────────────────────────────────
@st.cache_resource
def get_chatbot() -> ChatBot:
    return ChatBot()

bot = get_chatbot()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — NLP Insights (same logic as your original code)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔍 NLP Insights")
    st.caption("Live analysis of the last message")

    meta = st.session_state.nlp_meta

    # ── Same as your original code ─────────────────────────────────────────────
    if meta:
        # Intent — exactly as original
        st.metric("🎯 Intent", meta.get("intent", "—"))

        # Sentiment — exactly as original
        # NEW — adds emoji based on sentiment value
        sentiment_raw = meta.get("sentiment", "—")

        sentiment_emoji = {
            "POSITIVE": "😊",
             "NEGATIVE": "😟",
             "NEUTRAL":  "😐",
            }.get(sentiment_raw.split(" ")[0].upper(), "💬")

        st.metric("💬 Sentiment", f"{sentiment_emoji} {sentiment_raw}")

        # Entities — exactly as original using format_entities_md
        st.markdown("**📌 Entities detected**")
        st.markdown(format_entities_md(meta.get("entities", [])))

    else:
        # Info box — exactly as original
        st.info("Send a message to see NLP analysis here.")

    st.divider()

    # Clear chat — exactly as original
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.nlp_meta = {}
        st.rerun()

    st.divider()

    # Model info expander
    with st.expander("🤗 Model Info"):
        for name, repo in get_model_info().items():
            st.code(f"{name}:\n{repo}", language=None)

    # Device and model info — exactly as original
    st.markdown(
        f"**Model:** `{config.GROQ_MODEL}`  \n"
        f"**Device:** `{config.TORCH_DEVICE}`"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Gradient title ─────────────────────────────────────────────────────────────
st.markdown("# 🛍️ E-Commerce AI Chatbot")
st.markdown(
    '<p style="color:rgba(148,163,184,0.6); font-size:0.88rem; '
    'margin-top:-0.5rem; margin-bottom:1rem;">'
    'Powered by your NLP models &nbsp;·&nbsp; '
    'Ask about orders, returns, products &amp; more'
    '</p>',
    unsafe_allow_html=True,
)

# ── Empty state suggestion cards ───────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:2rem 0;">
            <div style="background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.2);
                        border-radius:14px;padding:1rem 1.1rem;">
                <p style="color:#A78BFA;font-size:0.82rem;font-weight:600;margin:0 0 4px;">
                    📦 Track Order</p>
                <p style="color:rgba(148,163,184,0.75);font-size:0.78rem;margin:0;">
                    Where is my order #45678?</p>
            </div>
            <div style="background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.2);
                        border-radius:14px;padding:1rem 1.1rem;">
                <p style="color:#34D399;font-size:0.82rem;font-weight:600;margin:0 0 4px;">
                    ↩️ Return &amp; Refund</p>
                <p style="color:rgba(148,163,184,0.75);font-size:0.78rem;margin:0;">
                    I want to return a damaged item</p>
            </div>
            <div style="background:rgba(96,165,250,0.07);border:1px solid rgba(96,165,250,0.2);
                        border-radius:14px;padding:1rem 1.1rem;">
                <p style="color:#60A5FA;font-size:0.82rem;font-weight:600;margin:0 0 4px;">
                    ❓ Product Info</p>
                <p style="color:rgba(148,163,184,0.75);font-size:0.78rem;margin:0;">
                    What are the specs of this product?</p>
            </div>
            <div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);
                        border-radius:14px;padding:1rem 1.1rem;">
                <p style="color:#FBBF24;font-size:0.82rem;font-weight:600;margin:0 0 4px;">
                    💳 Payment Issue</p>
                <p style="color:rgba(148,163,184,0.75);font-size:0.78rem;margin:0;">
                    My payment failed but was deducted</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render existing chat history — exactly as original ─────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input — exactly as original ──────────────────────────────────────────
if user_input := st.chat_input("Type your message…"):

    user_input = clean_text(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    groq_history = to_groq_history(st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        gen, nlp_meta = bot.process_stream(user_input, groq_history)
        full_response = st.write_stream(gen)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.nlp_meta = nlp_meta
    st.rerun()