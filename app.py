import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="ScoreSense", page_icon="📊", layout="wide")

# Custom dark style
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: white;
    }
    h1, h2, h3 {
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("best_model.pkl")

# Title
st.title("📊 ScoreSense Dashboard")
st.markdown("### Predict your score... and your excuses 😄")

# Sidebar
st.sidebar.header("🧾 Enter Your Details")

study_hours = st.sidebar.slider("📚 Study Hours", 0.0, 12.0, 2.0)
attendance = st.sidebar.slider("🏫 Attendance (%)", 0.0, 100.0, 75.0)
mental_health = st.sidebar.slider("🧠 Mental Health", 1, 10, 5)
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 0.0, 12.0, 7.0)
exercise_frequency = st.sidebar.slider("🏃 Exercise (days/week)", 0, 7, 3)
part_time_job = st.sidebar.selectbox("💼 Part-time Job?", ["No", "Yes"])

ptj_encoded = 1 if part_time_job == "Yes" else 0

# Predict
if st.sidebar.button("🚀 Predict Now"):

    input_data = np.array([[
        study_hours,
        attendance,
        mental_health,
        sleep_hours,
        exercise_frequency,
        ptj_encoded
    ]])

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))

    # Layout
    col1, col2 = st.columns([1, 1])

    # 🚗 SPEEDOMETER
    with col1:
        st.subheader("🎯 Performance Speedometer")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "cyan", 'width': 4},
                    'value': prediction
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # Funny category
        if prediction < 40:
            st.error("📉 Bro... Netflix is winning 😭")
        elif prediction < 70:
            st.warning("😐 Not bad, not great... balance needed!")
        else:
            st.success("🔥 Topper vibes activated 😎")

    # 📊 BAR GRAPH
    with col2:
        st.subheader("📊 Your Inputs")

        features = ['Study', 'Attendance', 'Mental', 'Sleep', 'Exercise']
        values = [study_hours, attendance/10, mental_health, sleep_hours, exercise_frequency]

        fig2, ax = plt.subplots()
        ax.bar(features, values)
        ax.set_title("Input Comparison")

        st.pyplot(fig2)

    # 🧠 INSIGHTS
    st.subheader("🧠 Smart Insights")

    if study_hours < 2:
        st.warning("📚 2 hours? Even reels take more time 💀")

    if sleep_hours < 6:
        st.warning("😴 Sleep low → brain slow 🧠")

    if mental_health < 4:
        st.warning("🧠 Take care of yourself first ❤️")

    if exercise_frequency == 0:
        st.warning("🏃 No exercise? Even code needs refresh 😅")

    if attendance < 50:
        st.warning("🏫 Attendance risky... teachers know 👀")

    # 📋 SUMMARY TABLE
    st.subheader("📋 Summary")

    df = pd.DataFrame({
        "Feature": ["Study Hours", "Attendance", "Mental Health", "Sleep", "Exercise", "Part-time Job"],
        "Value": [study_hours, attendance, mental_health, sleep_hours, exercise_frequency, part_time_job]
    })

    st.dataframe(df)