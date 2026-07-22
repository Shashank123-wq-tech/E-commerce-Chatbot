from src import memory
from src import database as db
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Inter',sans-serif;
}

/* ---------------- App Background ---------------- */

.stApp{
    background:linear-gradient(135deg,#F8FAFC 0%,#EEF2FF 50%,#FFFFFF 100%);
    color:#1E293B;
}

/* ---------------- Hide Branding ---------------- */

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header[data-testid="stHeader"]{
    background:transparent;
}

/* ---------------- Sidebar Toggle Button ---------------- */

button[kind="header"],
[data-testid="collapsedControl"]{
    display:flex !important;
    visibility:visible !important;
    opacity:1 !important;
    z-index:9999 !important;
}

/* make arrow black */
[data-testid="collapsedControl"] svg{
    fill:#111827 !important;
    color:#111827 !important;
}

/* ---------------- Main Container ---------------- */

.block-container{
    max-width:900px;
    padding-top:1.5rem;
    padding-bottom:2rem;
}

/* ---------------- Heading ---------------- */

h1{
    font-family:'Poppins',sans-serif !important;
    font-size:2rem !important;
    font-weight:700;
    background:linear-gradient(90deg,#6D28D9,#3B82F6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* ---------------- Sidebar ---------------- */

[data-testid="stSidebar"]{
    background:#FFFFFF !important;
    border-right:1px solid #E5E7EB;
}

[data-testid="stSidebar"] .block-container{
    padding-top:1rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2{
    color:#6D28D9 !important;
    -webkit-text-fill-color:#6D28D9 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li{
    color:#475569 !important;
}

[data-testid="stSidebar"] strong{
    color:#111827 !important;
}

/* ---------------- Metric Cards ---------------- */

[data-testid="stMetric"]{
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:15px;
    box-shadow:0 3px 10px rgba(0,0,0,.05);
    transition:.25s;
}

[data-testid="stMetric"]:hover{
    transform:translateY(-2px);
    border-color:#8B5CF6;
}

[data-testid="stMetricLabel"] p{
    color:#64748B !important;
    font-size:12px;
}

[data-testid="stMetricValue"]{
    color:#111827 !important;
}

/* ---------------- Chat ---------------- */

[data-testid="stChatMessage"]{
    background:transparent;
    border:none;
}

[data-testid="stChatMessageContent"]{
    background:white !important;
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:15px;
    color:#111827 !important;
    box-shadow:0 3px 10px rgba(0,0,0,.05);
}

/* User Avatar */

[data-testid="chatAvatarIcon-user"]{
    background:linear-gradient(135deg,#7C3AED,#A78BFA);
    border-radius:50%;
}

/* Assistant Avatar */

[data-testid="chatAvatarIcon-assistant"]{
    background:linear-gradient(135deg,#0EA5E9,#38BDF8);
    border-radius:10px;
}

/* ---------------- Chat Input ---------------- */

[data-testid="stChatInput"]{
    background:white !important;
    border:1px solid #CBD5E1 !important;
    border-radius:14px;
}

[data-testid="stChatInput"]:focus-within{
    border-color:#8B5CF6 !important;
    box-shadow:0 0 0 3px rgba(139,92,246,.15);
}

/* Send Button */

[data-testid="stChatInputSubmitButton"] button{
    background:linear-gradient(135deg,#7C3AED,#8B5CF6) !important;
    color:white !important;
    border:none;
    border-radius:10px;
}

/* ---------------- Buttons ---------------- */

.stButton button{
    background:white;
    color:#DC2626;
    border:1px solid #FCA5A5;
    border-radius:10px;
}

.stButton button:hover{
    background:#FEE2E2;
}

/* ---------------- Expander ---------------- */

[data-testid="stExpander"]{
    background:#FFFFFF;
    border:1px solid #E5E7EB;
    border-radius:10px;
}

/* ---------------- Code ---------------- */

code{
    background:#EEF2FF !important;
    color:#4C1D95 !important;
    border-radius:6px;
    padding:2px 6px;
}

/* ---------------- Scrollbar ---------------- */

::-webkit-scrollbar{
    width:7px;
}

::-webkit-scrollbar-thumb{
    background:#A78BFA;
    border-radius:20px;
}

::-webkit-scrollbar-track{
    background:transparent;
}

</style>
""", unsafe_allow_html=True)

# ── Database init ──────────────────────────────────────────────────────────────
db.init_db()   # creates tables if they don't exist (safe to call every run)

# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE LOGIN GATE — NEW
# ══════════════════════════════════════════════════════════════════════════════
if not st.user.is_logged_in:
    st.markdown(
        """
        <div style="display:flex; flex-direction:column; align-items:center;
                    justify-content:center; min-height:65vh; text-align:center;">
            <h1 style="margin-bottom:0.3rem;">🛍️ E-Commerce AI Chatbot</h1>
            <p style="color:#64748B; font-size:0.95rem; margin-bottom:2rem;">
                Sign in to start chatting — your conversation history is
                saved to your account
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button(
            "🔐 Continue with Google",
            on_click=st.login,
            use_container_width=True,
        )
    st.stop()   # ← nothing below this runs until the user logs in


# ── HuggingFace auth ───────────────────────────────────────────────────────────
setup_hf_auth()

# ── Session state ──────────────────────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    user_id, conversation_id = memory.init_conversation()
    st.session_state.user_id = user_id
    st.session_state.conversation_id = conversation_id
    # Load past messages from DB so refresh doesn't lose chat history
    st.session_state.messages = memory.load_history_for_ui(conversation_id)

init_session_defaults(st.session_state, {
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

    # ── Logged-in Google user card — NEW ────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px;
                    background:#F5F3FF; border:1px solid #DDD6FE;
                    border-radius:10px; padding:8px 12px; margin-bottom:0.8rem;">
            <img src="{st.user.picture}" style="width:30px; height:30px;
                    border-radius:50%;" />
            <div>
                <p style="margin:0; font-size:0.82rem; font-weight:600; color:#4C1D95;">
                    {st.user.name}
                </p>
                <p style="margin:0; font-size:0.7rem; color:#8B5CF6;">
                    {st.user.email}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚪 Log out", use_container_width=True):
        st.logout()

    st.divider()

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

    # Clear chat — MODIFIED: also clears DB-persisted messages for this conversation
    if st.button("🗑️ Clear chat", use_container_width=True):
        memory.clear_conversation(st.session_state.conversation_id)
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

# ── Chat input — MODIFIED: uses conversation_id (DB memory) instead of raw history,
#    and persists both messages to PostgreSQL after the response ─────────────────
if user_input := st.chat_input("Type your message…"):

    user_input = clean_text(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        gen, nlp_meta = bot.process_stream(
            user_input,
            st.session_state.conversation_id     # ← DB-backed context, not raw history
        )
        full_response = st.write_stream(gen)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.nlp_meta = nlp_meta

    # Save this turn to PostgreSQL so it survives refresh/restart
    memory.save_turn(
        st.session_state.conversation_id,
        user_input,
        full_response,
        nlp_meta,
    )

    st.rerun()