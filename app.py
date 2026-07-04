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
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');

/* ── Global reset ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0D0D1A 0%, #111128 50%, #0D0D1A 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit default elements ─────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 860px !important;
}

/* ── App Title ───────────────────────────────────────────────────────────── */
h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #A78BFA, #60A5FA, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13132B 0%, #0F0F22 100%) !important;
    border-right: 1px solid rgba(167, 139, 250, 0.2) !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    max-width: 100% !important;
}

/* Sidebar title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.1rem !important;
    color: #A78BFA !important;
    -webkit-text-fill-color: #A78BFA !important;
    letter-spacing: 0.3px;
}

[data-testid="stSidebar"] .stCaption {
    color: rgba(167, 139, 250, 0.6) !important;
    font-size: 0.72rem !important;
}

/* ── NLP Metric Cards ─────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(167, 139, 250, 0.07) !important;
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.6rem !important;
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(167, 139, 250, 0.45) !important;
    background: rgba(167, 139, 250, 0.12) !important;
}

[data-testid="stMetricLabel"] p {
    color: rgba(167, 139, 250, 0.7) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

[data-testid="stMetricValue"] {
    color: #E2E8F0 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Chat Messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
    margin-bottom: 0.5rem !important;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

/* Avatar styling */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #6C63FF, #A78BFA) !important;
    border-radius: 50% !important;
    min-width: 36px !important;
    height: 36px !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #0F766E, #34D399) !important;
    border-radius: 10px !important;
    min-width: 36px !important;
    height: 36px !important;
}

/* Message content area */
[data-testid="stChatMessageContent"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    border-top-left-radius: 4px !important;
    padding: 0.85rem 1.1rem !important;
    color: #E2E8F0 !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    max-width: 85% !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25) !important;
}

/* ── Chat Input ──────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(167, 139, 250, 0.35) !important;
    border-radius: 16px !important;
    color: #E2E8F0 !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(167, 139, 250, 0.75) !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.12) !important;
}

[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #6C63FF, #A78BFA) !important;
    border: none !important;
    border-radius: 10px !important;
    transition: opacity 0.2s ease !important;
}

[data-testid="stChatInputSubmitButton"] button:hover {
    opacity: 0.85 !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {
    border-color: rgba(167, 139, 250, 0.15) !important;
    margin: 0.8rem 0 !important;
}

/* ── Clear Chat Button ───────────────────────────────────────────────────── */
.stButton button {
    background: rgba(239, 68, 68, 0.08) !important;
    border: 1px solid rgba(239, 68, 68, 0.25) !important;
    color: #FCA5A5 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    padding: 0.45rem 1rem !important;
}

.stButton button:hover {
    background: rgba(239, 68, 68, 0.18) !important;
    border-color: rgba(239, 68, 68, 0.5) !important;
    color: #FCA5A5 !important;
}

/* ── Expander (Model Info) ───────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(167, 139, 250, 0.05) !important;
    border: 1px solid rgba(167, 139, 250, 0.15) !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    color: rgba(167, 139, 250, 0.8) !important;
    font-size: 0.8rem !important;
}

/* ── Code blocks ─────────────────────────────────────────────────────────── */
code {
    background: rgba(167, 139, 250, 0.1) !important;
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    border-radius: 6px !important;
    color: #C4B5FD !important;
    font-size: 0.78rem !important;
    padding: 0.2rem 0.4rem !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(167, 139, 250, 0.3);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(167, 139, 250, 0.5); }

/* ── Sidebar markdown ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
}

[data-testid="stSidebar"] strong {
    color: #CBD5E1 !important;
}

[data-testid="stSidebar"] code {
    font-size: 0.7rem !important;
}
</style>
""", unsafe_allow_html=True)


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
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Header
    st.markdown("## 🔍 NLP Insights")
    st.caption("Real-time analysis of your last message")
    st.divider()

    meta = st.session_state.nlp_meta

    if meta:
        # Intent card
        intent_val = meta.get("intent", "—")
        st.metric(
            label="🎯  Detected Intent",
            value=intent_val.replace("_", " ").title()
        )

        # Sentiment card with colour coding
        sentiment_raw = meta.get("sentiment", "—")
        sentiment_label = sentiment_raw.split(" ")[0].upper()

        sentiment_icon = {
            "POSITIVE": "😊",
            "NEGATIVE": "😟",
            "NEUTRAL":  "😐",
        }.get(sentiment_label, "💬")

        st.metric(
            label=f"{sentiment_icon}  Sentiment",
            value=sentiment_raw,
        )

        # Entity badges
        entities = meta.get("entities", [])
        st.markdown("**📌 Entities detected**")

        if entities:
            badge_colors = {
                "PERSON":   ("#6C63FF", "#EDE9FE"),
                "ORG":      ("#0F766E", "#CCFBF1"),
                "GPE":      ("#1D4ED8", "#DBEAFE"),
                "ORDER_ID": ("#B45309", "#FEF3C7"),
                "AMOUNT":   ("#065F46", "#D1FAE5"),
                "PHONE":    ("#7C3AED", "#EDE9FE"),
                "EMAIL":    ("#9D174D", "#FCE7F3"),
                "DATE":     ("#1E3A5F", "#DBEAFE"),
                "LOC":      ("#1D4ED8", "#DBEAFE"),
            }

            badge_html = '<div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:4px;">'
            for ent_str in entities:
                # ent_str looks like: Entity('Delhi', 'GPE', 0.99)
                try:
                    parts    = ent_str.replace("Entity(", "").replace(")", "").split(",")
                    ent_text = parts[0].strip().strip("'\"")
                    ent_type = parts[1].strip().strip("'\"")
                    bg, fg   = badge_colors.get(ent_type, ("#374151", "#F3F4F6"))
                    badge_html += (
                        f'<span style="background:{bg}22; border:1px solid {bg}55; '
                        f'color:{bg}; border-radius:8px; padding:2px 9px; '
                        f'font-size:0.72rem; font-weight:500;">'
                        f'{ent_text} <span style="opacity:0.6;font-size:0.65rem;">({ent_type})</span>'
                        f'</span>'
                    )
                except Exception:
                    pass
            badge_html += '</div>'
            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:rgba(148,163,184,0.5); font-style:italic; font-size:0.8rem; margin:4px 0;">None detected</p>',
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            """
            <div style="
                background: rgba(167,139,250,0.06);
                border: 1px dashed rgba(167,139,250,0.25);
                border-radius: 12px;
                padding: 1.2rem;
                text-align: center;
                margin: 0.5rem 0;
            ">
                <p style="color:rgba(167,139,250,0.6); font-size:0.82rem; margin:0;">
                    💬 Send a message to see live NLP analysis here
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Clear chat
    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.nlp_meta = {}
        st.rerun()

    st.divider()

    # Model info expander
    with st.expander("🤗  Model info"):
        for name, repo in get_model_info().items():
            st.code(f"{name}\n{repo}", language=None)

    # Footer info
    st.markdown(
        f"""
        <div style="margin-top:0.5rem;">
            <p style="color:rgba(148,163,184,0.45); font-size:0.7rem; margin:2px 0;">
                🤖 LLM &nbsp;→&nbsp; <code style="font-size:0.68rem;">{config.GROQ_MODEL}</code>
            </p>
            <p style="color:rgba(148,163,184,0.45); font-size:0.7rem; margin:2px 0;">
                ⚙️ Device &nbsp;→&nbsp; <code style="font-size:0.68rem;">{config.TORCH_DEVICE}</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🛍️ E-Commerce Chatbot")
st.markdown(
    '<p style="color:rgba(148,163,184,0.6); font-size:0.88rem; margin-top:-0.5rem; margin-bottom:1rem;">'
    'Powered by your NLP models &nbsp;·&nbsp; Ask about orders, returns, products & more'
    '</p>',
    unsafe_allow_html=True,
)

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div style="
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 2rem 0;
        ">
            <div style="background:rgba(167,139,250,0.07); border:1px solid rgba(167,139,250,0.2);
                        border-radius:14px; padding:1rem 1.1rem;">
                <p style="color:#A78BFA; font-size:0.82rem; font-weight:600; margin:0 0 4px;">📦 Track Order</p>
                <p style="color:rgba(148,163,184,0.75); font-size:0.78rem; margin:0;">Where is my order #45678?</p>
            </div>
            <div style="background:rgba(52,211,153,0.07); border:1px solid rgba(52,211,153,0.2);
                        border-radius:14px; padding:1rem 1.1rem;">
                <p style="color:#34D399; font-size:0.82rem; font-weight:600; margin:0 0 4px;">↩️ Return & Refund</p>
                <p style="color:rgba(148,163,184,0.75); font-size:0.78rem; margin:0;">I want to return a damaged item</p>
            </div>
            <div style="background:rgba(96,165,250,0.07); border:1px solid rgba(96,165,250,0.2);
                        border-radius:14px; padding:1rem 1.1rem;">
                <p style="color:#60A5FA; font-size:0.82rem; font-weight:600; margin:0 0 4px;">❓ Product Info</p>
                <p style="color:rgba(148,163,184,0.75); font-size:0.78rem; margin:0;">What are the specs of this product?</p>
            </div>
            <div style="background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.2);
                        border-radius:14px; padding:1rem 1.1rem;">
                <p style="color:#FBBF24; font-size:0.82rem; font-weight:600; margin:0 0 4px;">💳 Payment Issue</p>
                <p style="color:rgba(148,163,184,0.75); font-size:0.78rem; margin:0;">My payment failed but was deducted</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about your order, returns, products..."):
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