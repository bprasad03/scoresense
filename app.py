import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScoreSense · AI Grade Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Exo+2:wght@300;400;500;600&display=swap');

:root {
    --blue:         #00d4ff;
    --purple:       #7b2fff;
    --dark-bg:      #020b18;
    --dark-card:    #050f1f;
    --dark-surface: #091929;
    --text-primary: #e8f4ff;
    --text-muted:   #6a8fb5;
    --danger:       #ff3860;
    --warning:      #ffdd57;
    --success:      #00e68c;
    --border:       rgba(255,255,255,0.08);
    --border-blue:  rgba(0,212,255,0.2);
}

html, body, .stApp {
    background-color: var(--dark-bg) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }
.stDeployButton { display: none; }

/* NAVBAR */
.navbar {
    position: sticky;
    top: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem;
    height: 62px;
    background: rgba(2,11,24,0.97);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-blue);
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: var(--blue);
}
.navbar-brand span { font-weight: 400; color: var(--text-muted); font-size: 0.82rem; letter-spacing: 3px; }
.navbar-links { display: flex; gap: 2rem; }
.navbar-links a {
    font-size: 0.8rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    text-decoration: none;
    transition: color 0.2s;
}
.navbar-links a:hover { color: var(--text-primary); }
.navbar-badge {
    background: rgba(0,212,255,0.06);
    border: 1px solid var(--border-blue);
    color: var(--text-muted);
    font-size: 0.7rem;
    font-family: 'Orbitron', monospace;
    letter-spacing: 1px;
    padding: 4px 14px;
    border-radius: 20px;
}

/* HERO */
.hero {
    padding: 3.5rem 3rem 2.8rem;
    background: var(--dark-bg);
    border-bottom: 1px solid var(--border);
}
.hero-eyebrow {
    font-family: 'Orbitron', monospace;
    font-size: 0.68rem;
    letter-spacing: 5px;
    color: var(--blue);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 900;
    line-height: 1.1;
    color: var(--text-primary);
    margin-bottom: 0.9rem;
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    max-width: 520px;
    line-height: 1.6;
}
.hero-pills { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1.4rem; }
.pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.72rem;
    font-family: 'Orbitron', monospace;
    letter-spacing: 1px;
    padding: 5px 14px;
    border-radius: 20px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: var(--dark-card) !important;
    border-right: 1px solid var(--border) !important;
}
.sidebar-header {
    padding: 1.5rem 1.2rem 0.5rem;
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 3px;
    color: var(--blue);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}
.sidebar-section {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: var(--text-muted);
    text-transform: uppercase;
    padding: 0.8rem 0 0.3rem;
}

[data-testid="stSlider"] > div > div > div { background: var(--blue) !important; }
[data-testid="stSlider"] > div > div > div > div {
    background: var(--blue) !important;
    border: 2px solid var(--blue) !important;
}
[data-testid="stSelectbox"] select,
.stSelectbox > div > div {
    background: var(--dark-surface) !important;
    border: 1px solid var(--border-blue) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
.stSlider label { color: var(--text-muted) !important; font-size: 0.82rem !important; }

.stButton > button {
    width: 100%;
    background: rgba(0,212,255,0.07) !important;
    border: 1px solid var(--border-blue) !important;
    color: var(--blue) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 6px !important;
    transition: background 0.2s !important;
    margin-top: 1rem !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.14) !important;
}

/* SECTION */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1.2rem;
    border-left: 3px solid var(--blue);
    padding-left: 12px;
}

/* SCORE CARD */
.score-card {
    background: var(--dark-surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-bottom: 1.2rem;
}
.score-number {
    font-family: 'Orbitron', monospace;
    font-size: 5rem;
    font-weight: 900;
    line-height: 1;
    color: var(--blue);
}
.score-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 5px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* INSIGHT CARDS */
.insight-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 1rem 0;
}
.insight-card {
    background: var(--dark-surface);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    border-left: 3px solid var(--blue);
    font-size: 0.88rem;
    line-height: 1.5;
    color: var(--text-muted);
}
.insight-card.warn   { border-left-color: var(--warning); }
.insight-card.danger { border-left-color: var(--danger); }
.insight-card.good   { border-left-color: var(--success); }
.insight-card strong { color: var(--text-primary); display: block; margin-bottom: 2px; font-size: 0.82rem; }

/* DATA TABLE */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.5rem; }
.data-table th {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--blue);
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    text-align: left;
}
.data-table td { padding: 10px 14px; color: var(--text-muted); border-bottom: 1px solid var(--border); }
.data-table tr:last-child td { border-bottom: none; }
.data-table td:last-child {
    color: var(--text-primary);
    font-weight: 500;
    text-align: right;
    font-family: 'Orbitron', monospace;
    font-size: 0.82rem;
}

/* STATUS BANNER */
.status-banner {
    border-radius: 8px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.92rem;
    font-weight: 500;
    margin: 1rem 0;
    border: 1px solid;
}
.status-banner.danger  { background: rgba(255,56,96,0.08);  border-color: rgba(255,56,96,0.3);  color: #ffa0b8; }
.status-banner.warning { background: rgba(255,221,87,0.08); border-color: rgba(255,221,87,0.3); color: #ffe98a; }
.status-banner.success { background: rgba(0,230,140,0.08);  border-color: rgba(0,230,140,0.3);  color: #5fffbe; }
.status-icon { font-size: 1.5rem; }

/* FOOTER */
.footer {
    margin-top: 4rem;
    border-top: 1px solid var(--border);
    padding: 2rem 3rem;
    background: var(--dark-card);
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 2rem;
}
.footer-brand {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    font-weight: 800;
    color: var(--blue);
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}
.footer-text { font-size: 0.8rem; color: var(--text-muted); line-height: 1.6; }
.footer-heading {
    font-family: 'Orbitron', monospace;
    font-size: 0.62rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.8rem;
}
.footer-links { list-style: none; padding: 0; margin: 0; }
.footer-links li { margin-bottom: 0.4rem; }
.footer-links a { font-size: 0.8rem; color: var(--text-muted); text-decoration: none; transition: color 0.2s; }
.footer-links a:hover { color: var(--text-primary); }
.footer-bottom {
    border-top: 1px solid var(--border);
    padding: 1rem 3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: #3a5478;
    background: var(--dark-card);
}
.footer-bottom span { font-family: 'Orbitron', monospace; letter-spacing: 1px; }

/* CHART CONTAINER */
.chart-wrap {
    background: var(--dark-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
}

@media (max-width: 768px) {
    .footer { grid-template-columns: 1fr; }
    .hero-title { font-size: 1.8rem; }
    .score-number { font-size: 3.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ─── NAVBAR ────────────────────────────────────────────────────────────────────
st.markdown("""
<nav class="navbar">
  <div class="navbar-brand">
    ⚡&nbsp; SCORE<span>SENSE</span>
  </div>
  <div class="navbar-links">
    <a href="#">Dashboard</a>
    <a href="#">Insights</a>
    <a href="#">About</a>
  </div>
  <div class="navbar-badge">AI Powered</div>
</nav>
""", unsafe_allow_html=True)


# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<section class="hero">
  <div class="hero-eyebrow">⚡ AI-Powered Academic Intelligence</div>
  <div class="hero-title">Predict Your Score.<br>Own Your Future.</div>
  <div class="hero-sub">
    Feed in your habits, get back your grade prediction — powered by machine learning.
    Know where you stand before the exam does.
  </div>
  <div class="hero-pills">
    <span class="pill">📚 Study Analytics</span>
    <span class="pill purple">🧠 Mental Health</span>
    <span class="pill cyan">⚡ Real-time Prediction</span>
    <span class="pill">😴 Sleep Score</span>
  </div>
</section>
""", unsafe_allow_html=True)


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-header">⚡ Input Parameters</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section">📊 Academic</div>', unsafe_allow_html=True)
study_hours     = st.sidebar.slider("📚 Study Hours / day", 0.0, 12.0, 2.0, 0.5)
attendance      = st.sidebar.slider("🏫 Attendance (%)", 0.0, 100.0, 75.0, 1.0)

st.sidebar.markdown('<div class="sidebar-section">🧬 Lifestyle</div>', unsafe_allow_html=True)
mental_health   = st.sidebar.slider("🧠 Mental Health (1–10)", 1, 10, 5)
sleep_hours     = st.sidebar.slider("😴 Sleep Hours / night", 0.0, 12.0, 7.0, 0.5)
exercise_freq   = st.sidebar.slider("🏃 Exercise (days/week)", 0, 7, 3)

st.sidebar.markdown('<div class="sidebar-section">💼 Employment</div>', unsafe_allow_html=True)
part_time_job   = st.sidebar.selectbox("Part-time Job?", ["No", "Yes"])
ptj_encoded     = 1 if part_time_job == "Yes" else 0

predict_clicked = st.sidebar.button("⚡ PREDICT NOW")


# ─── MAIN CONTENT ──────────────────────────────────────────────────────────────
if not predict_clicked:
    st.markdown("""
    <div style="text-align:center; padding: 5rem 2rem; color: #2a4a6e;">
      <div style="font-family:'Orbitron',monospace; font-size:4rem; margin-bottom:1rem; opacity:0.2;">⚡</div>
      <div style="font-family:'Orbitron',monospace; font-size:0.8rem; letter-spacing:4px; margin-bottom:0.5rem;">
        AWAITING INPUT
      </div>
      <div style="font-size:0.9rem; color:#1e3a5a;">
        Configure your parameters in the sidebar and click <strong style="color:#00d4ff40;">⚡ PREDICT NOW</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Load model & predict ──
    try:
        model      = joblib.load("best_model.pkl")
        input_arr  = np.array([[study_hours, attendance, mental_health,
                                 sleep_hours, exercise_freq, ptj_encoded]])
        prediction = float(np.clip(model.predict(input_arr)[0], 0, 100))
    except Exception:
        # Demo mode — formula-based fallback if model not found
        prediction = float(np.clip(
            study_hours * 4.5 +
            (attendance / 100) * 20 +
            mental_health * 1.5 +
            sleep_hours * 1.2 +
            exercise_freq * 0.8 -
            ptj_encoded * 5 + 10,
            0, 100
        ))

    score = round(prediction, 1)

    # ── Layout ──────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Speedometer + Score ────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-label">Performance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Score Speedometer</div>', unsafe_allow_html=True)

        # Score card
        st.markdown(f"""
        <div class="score-card">
          <div class="score-number">{score}</div>
          <div class="score-label">Predicted Score / 100</div>
        </div>
        """, unsafe_allow_html=True)

        # Status banner
        if score < 40:
            st.markdown("""
            <div class="status-banner danger">
              <span class="status-icon">📉</span>
              <span>Bro... Netflix is winning this round. Serious intervention needed!</span>
            </div>""", unsafe_allow_html=True)
        elif score < 70:
            st.markdown("""
            <div class="status-banner warning">
              <span class="status-icon">⚖️</span>
              <span>Not bad, not great. A little more focus and you're there.</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-banner success">
              <span class="status-icon">🔥</span>
              <span>Topper vibes activated. Keep this energy going!</span>
            </div>""", unsafe_allow_html=True)

        # Plotly Gauge
        needle_color = ("#ff3860" if score < 40 else
                        "#ffdd57" if score < 70 else "#00e68c")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Score", 'font': {'color': '#6a8fb5', 'size': 14,
                                              'family': 'Orbitron'}},
            number={'font': {'color': '#00d4ff', 'size': 42, 'family': 'Orbitron'}},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickcolor': '#1a3a5c',
                    'tickfont': {'color': '#3a5478', 'size': 10, 'family': 'Orbitron'},
                },
                'bar': {'color': needle_color, 'thickness': 0.08},
                'bgcolor': 'rgba(9,25,41,1)',
                'bordercolor': 'rgba(0,212,255,0.12)',
                'borderwidth': 1,
                'steps': [
                    {'range': [0,  40], 'color': 'rgba(255,56,96,0.18)'},
                    {'range': [40, 70], 'color': 'rgba(255,221,87,0.18)'},
                    {'range': [70,100], 'color': 'rgba(0,230,140,0.18)'},
                ],
                'threshold': {
                    'line': {'color': '#00d4ff', 'width': 3},
                    'thickness': 0.85,
                    'value': score,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgb(5,15,31)',
            plot_bgcolor='rgb(5,15,31)',
            font={'color': '#00d4ff'},
            height=280,
            margin=dict(t=30, b=0, l=20, r=20),
        )
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: Bar chart ─────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-label">Your Inputs</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Comparison</div>', unsafe_allow_html=True)

        features = ['Study\nHours', 'Attendance\n/10', 'Mental\nHealth',
                    'Sleep\nHours', 'Exercise\nDays']
        values   = [study_hours, attendance / 10, mental_health,
                    sleep_hours, exercise_freq]
        colors   = ['#00d4ff', '#7b2fff', '#00ffea', '#00d4ff', '#7b2fff']
        edge_c   = ['rgba(0,212,255,0.6)', 'rgba(123,47,255,0.6)',
                    'rgba(0,255,234,0.6)', 'rgba(0,212,255,0.6)', 'rgba(123,47,255,0.6)']

        fig_bar = go.Figure(go.Bar(
            x=features,
            y=values,
            marker_color=colors,
            marker_line_color=edge_c,
            marker_line_width=1.5,
            text=[f"{v:.1f}" for v in values],
            textposition='outside',
            textfont={'color': '#a0bfd8', 'size': 11, 'family': 'Orbitron'},
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgb(9,25,41)',
            plot_bgcolor='rgb(5,15,31)',
            font={'color': '#6a8fb5', 'family': 'Exo 2'},
            xaxis=dict(
                tickfont={'color': '#3a5478', 'size': 10, 'family': 'Orbitron'},
                gridcolor='#0a1e33',
                linecolor='#0a1e33',
            ),
            yaxis=dict(
                tickfont={'color': '#3a5478', 'size': 10},
                gridcolor='#0a1e33',
                linecolor='#0a1e33',
                zeroline=False,
            ),
            bargap=0.35,
            height=320,
            margin=dict(t=20, b=10, l=10, r=10),
        )
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary data table
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)
        rows = [
            ("📚 Study Hours",   f"{study_hours} hrs/day"),
            ("🏫 Attendance",    f"{attendance:.0f}%"),
            ("🧠 Mental Health", f"{mental_health} / 10"),
            ("😴 Sleep Hours",   f"{sleep_hours} hrs/night"),
            ("🏃 Exercise",      f"{exercise_freq} days/week"),
            ("💼 Part-time Job", part_time_job),
        ]
        rows_html = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows
        )
        st.markdown(f"""
        <div class="chart-wrap">
          <table class="data-table">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    # ── INSIGHTS ─────────────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Smart Insights</div>', unsafe_allow_html=True)

    insights = []
    if study_hours >= 6:
        insights.append(("good", "🔥 Study Hours", "Excellent study discipline — your score reflects this effort."))
    elif study_hours < 2:
        insights.append(("danger", "📚 Study Hours Critical", "2 hours? Even reels take more time. Boost this first."))
    else:
        insights.append(("warn", "📚 Study Hours", "Decent, but increasing to 5+ hrs would push your grade significantly."))

    if sleep_hours < 6:
        insights.append(("danger", "😴 Sleep Deficit", "Sleep low → brain slow. Aim for 7–8 hrs for optimal retention."))
    elif sleep_hours > 9:
        insights.append(("warn", "😴 Oversleeping", "Too much sleep can reduce productive hours. 7–8 hrs is optimal."))
    else:
        insights.append(("good", "😴 Sleep Quality", f"{sleep_hours} hrs is solid. Your brain is well-rested."))

    if mental_health < 4:
        insights.append(("danger", "🧠 Mental Health Alert", "Take care of yourself first — no grade is worth your wellbeing. ❤️"))
    elif mental_health >= 8:
        insights.append(("good", "🧠 Mental Resilience", "Strong mental health — you're set up for peak performance."))
    else:
        insights.append(("warn", "🧠 Mental Health", "Could be better. Regular breaks and exercise can help a lot."))

    if exercise_freq == 0:
        insights.append(("danger", "🏃 Zero Exercise", "No exercise? Even code needs a refresh. 3 days/week minimum."))
    elif exercise_freq >= 4:
        insights.append(("good", "🏃 Active Lifestyle", "Great exercise habit — this boosts brain function and reduces stress."))

    if attendance < 50:
        insights.append(("danger", "🏫 Attendance Critical", "Below 50%? Teachers noticed. This hurts your grade hard."))
    elif attendance < 75:
        insights.append(("warn", "🏫 Attendance Low", "Risky zone. Push attendance above 80% for a safety net."))
    else:
        insights.append(("good", "🏫 Good Attendance", f"{attendance:.0f}% attendance — well within the safe zone."))

    if ptj_encoded:
        insights.append(("warn", "💼 Part-time Job", "Working part-time limits study hours. Try to keep a strict schedule."))

    cards_html = "".join(
        f'<div class="insight-card {cls}"><strong>{title}</strong>{msg}</div>'
        for cls, title, msg in insights
    )
    st.markdown(f'<div class="insight-grid">{cards_html}</div>', unsafe_allow_html=True)


# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<footer>
  <div class="footer">
    <div>
      <div class="footer-brand">⚡ SCORESENSE</div>
      <div class="footer-text">
        AI-powered academic performance predictor.<br>
        Built with machine learning to help students understand
        the factors that drive their success.
      </div>
    </div>
    <div>
      <div class="footer-heading">Features</div>
      <ul class="footer-links">
        <li><a href="#">Score Prediction</a></li>
        <li><a href="#">Smart Insights</a></li>
        <li><a href="#">Input Analysis</a></li>
        <li><a href="#">Performance Gauge</a></li>
      </ul>
    </div>
    <div>
      <div class="footer-heading">Technology</div>
      <ul class="footer-links">
        <li><a href="#">Scikit-learn Model</a></li>
        <li><a href="#">Streamlit</a></li>
        <li><a href="#">Plotly Visualizations</a></li>
        <li><a href="#">Custom CSS Theme</a></li>
      </ul>
    </div>
  </div>
  <div class="footer-bottom">
    <span>© 2025 SCORESENSE · ALL RIGHTS RESERVED</span>
    <span style="color:#1a3a5c;">Built with ⚡ for students who want to win</span>
  </div>
</footer>
""", unsafe_allow_html=True)