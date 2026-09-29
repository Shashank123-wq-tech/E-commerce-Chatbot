"""
pages/1_📊_Analytics_Dashboard.py — Business analytics dashboard.

Streamlit automatically adds any file in pages/ as a separate navigable
page — accessible via the sidebar page selector, right below app.py's
main page (called "app" by default).

Shows:
  - KPI overview cards
  - Intent distribution bar chart
  - Sentiment distribution pie chart
  - Sentiment trend over time (line chart)
  - Top detected entities
  - AI-generated business recommendations
"""

import streamlit as st
import pandas as pd
from src.config import config
from src import analytics
from src import recommendations as reco

st.set_page_config(
    page_title=f"{config.APP_TITLE} — Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Reuse the same light theme CSS as main app ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');

html, body, [class*="css"]{ font-family:'Inter',sans-serif; }

.stApp{
    background:linear-gradient(135deg,#F8FAFC 0%,#EEF2FF 50%,#FFFFFF 100%);
    color:#1E293B;
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}

h1, h2, h3 {
    font-family:'Poppins',sans-serif !important;
}

[data-testid="stMetric"]{
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:15px;
    box-shadow:0 3px 10px rgba(0,0,0,.05);
}

[data-testid="stMetricValue"]{ color:#111827 !important; }
[data-testid="stMetricLabel"] p{ color:#64748B !important; }

[data-testid="stSidebar"]{
    background:#FFFFFF !important;
    border-right:1px solid #E5E7EB;
}
</style>
""", unsafe_allow_html=True)


# ── Auth gate — reuse same login check as main app ─────────────────────────────
if not st.user.is_logged_in:
    st.warning("Please log in from the main chat page first.")
    st.stop()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 Customer Insights Dashboard")
st.caption("Aggregated intent, sentiment, and entity trends across all conversations")

# ── Time range selector ─────────────────────────────────────────────────────────
col_range, _ = st.columns([1, 3])
with col_range:
    days = st.selectbox(
        "Time range",
        options=[7, 14, 30, 90],
        index=2,
        format_func=lambda x: f"Last {x} days",
    )

st.divider()

# ── KPI Overview cards ───────────────────────────────────────────────────────────
stats = analytics.get_overview_stats(days)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💬 Total Messages", stats["total_messages"])
k2.metric("🗂️ Conversations", stats["total_conversations"])
k3.metric("👥 Unique Users", stats["total_users"])
k4.metric("😊 Positive Sentiment", f"{stats['positive_pct']}%")
k5.metric("😟 Negative Sentiment", f"{stats['negative_pct']}%")

st.divider()

# ── Charts row 1 — Intent + Sentiment distribution ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Intent Distribution")
    intent_dist = analytics.get_intent_distribution(days)
    if intent_dist:
        df_intent = pd.DataFrame(
            list(intent_dist.items()), columns=["Intent", "Count"]
        ).sort_values("Count", ascending=True)
        st.bar_chart(df_intent.set_index("Intent"), horizontal=True)
    else:
        st.info("No intent data available for this time range yet.")

with col2:
    st.subheader("💬 Sentiment Breakdown")
    sentiment_dist = analytics.get_sentiment_distribution(days)
    if sentiment_dist:
        df_sentiment = pd.DataFrame(
            list(sentiment_dist.items()), columns=["Sentiment", "Count"]
        )
        st.bar_chart(df_sentiment.set_index("Sentiment"))

        # Quick readable summary underneath
        total = sum(sentiment_dist.values())
        for label, count in sentiment_dist.items():
            pct = round(count / total * 100, 1)
            emoji = {"POSITIVE": "😊", "NEGATIVE": "😟", "NEUTRAL": "😐"}.get(label, "💬")
            st.caption(f"{emoji} {label}: {count} messages ({pct}%)")
    else:
        st.info("No sentiment data available for this time range yet.")

st.divider()

# ── Sentiment trend over time ────────────────────────────────────────────────────
st.subheader("📈 Sentiment Trend Over Time")
trend_data = analytics.get_sentiment_trend(days)
if trend_data:
    df_trend = pd.DataFrame(trend_data).set_index("date")
    st.line_chart(df_trend)
else:
    st.info("Not enough data yet to show a trend.")

st.divider()

# ── Top entities ──────────────────────────────────────────────────────────────────
st.subheader("📌 Most Mentioned Entity Types")
top_entities = analytics.get_top_entities(days)
if top_entities:
    df_entities = pd.DataFrame(top_entities, columns=["Entity Type", "Count"])
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(df_entities.set_index("Entity Type"))
    with col_table:
        st.dataframe(df_entities, use_container_width=True, hide_index=True)
else:
    st.info("No entities detected in this time range yet.")

st.divider()

# ── AI-generated recommendations ─────────────────────────────────────────────────
st.subheader("💡 AI-Generated Business Recommendations")

with st.spinner("Analyzing patterns and generating recommendations..."):
    recs = reco.generate_recommendations(
        intent_dist if intent_dist else {},
        sentiment_dist if sentiment_dist else {},
        top_entities if top_entities else [],
    )

st.markdown(
    f"""
    <div style="background:#F5F3FF; border:1px solid #DDD6FE; border-radius:14px;
                padding:1.2rem 1.5rem;">
        {recs}
    </div>
    """,
    unsafe_allow_html=True,
)