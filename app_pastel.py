import streamlit as st
import pickle
import pandas as pd
import json

st.set_page_config(page_title="Flight Delay Risk Predictor", page_icon="🌤️", layout="centered")

with open('flight_delay_lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# ---------- Pastel sky theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #FDF6F0 0%, #F2FBFF 100%);
    color: #4A4458;
}

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: #4A4458;
    margin-bottom: 2px;
}

.hero-sub {
    color: #8A8398;
    font-size: 0.95rem;
    margin-bottom: 28px;
}

.pastel-card {
    background-color: #FFFFFF;
    border-radius: 20px;
    padding: 28px 30px;
    box-shadow: 0 8px 24px rgba(174, 227, 245, 0.35);
    margin-bottom: 20px;
}

.cloud-divider {
    text-align: center;
    color: #AEE3F5;
    font-size: 1.3rem;
    margin: 18px 0;
    letter-spacing: 8px;
}

.stTextInput input, div[data-baseweb="select"] > div, .stNumberInput input {
    background-color: #F7FBFD !important;
    border: 1.5px solid #AEE3F5 !important;
    border-radius: 12px !important;
    color: #4A4458 !important;
}

.stSlider label, .stSelectbox label, .stTextInput label, .stNumberInput label {
    color: #8A8398 !important;
    font-weight: 600;
    font-size: 0.85rem;
}

.stButton button {
    background-color: #AEE3F5;
    color: #3A5568;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 14px;
    padding: 10px 26px;
    width: 100%;
    transition: 0.2s;
}

.stButton button:hover {
    background-color: #93D9F0;
    color: #2E4456;
}

.result-card {
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}

.result-ontime {
    background-color: #E4F8EF;
    border: 2px solid #B8E8D4;
}

.result-delayed {
    background-color: #FDECEA;
    border: 2px solid #F7B9B0;
}

.result-status {
    font-family: 'Poppins', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.result-prob {
    font-size: 1rem;
    color: #6B6577;
}
</style>
""", unsafe_allow_html=True)

# ---------- Session state setup ----------
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'prediction' not in st.session_state:
    st.session_state.prediction = None


def go_to(page):
    st.session_state.page = page


# ---------- LOGIN PAGE ----------
if st.session_state.page == 'login':
    st.markdown('<div class="hero-title">🌤️ Flight Delay Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Know your flight\'s odds before you pack your bags.</div>', unsafe_allow_html=True)

    st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
    name = st.text_input("Your name", placeholder="e.g. Luckmitha")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Continue"):
        if name.strip():
            st.session_state.username = name.strip()
            go_to('input')
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")

# ---------- INPUT PAGE ----------
elif st.session_state.page == 'input':
    st.markdown(f'<div class="hero-title">Hi, {st.session_state.username} ✈️</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Fill in your flight details below.</div>', unsafe_allow_html=True)

    st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("Month", list(range(1, 13)))
        hour = st.slider("Departure hour", 0, 23, 12)
        carrier = st.selectbox("Carrier", ['AA','DL','UA','WN','B6','AS','NK','F9','HA','G4','OO','YX','MQ','OH','YV','VX','EV'])
    with col2:
        day_of_week = st.selectbox("Day of week (0 = Mon)", list(range(0, 7)))
        distance = st.number_input("Distance (miles)", min_value=0, value=500)
        origin = st.selectbox("Origin airport", ['ATL','DFW','CLT','SFO','IAH','LAS','DTW','SEA','ORD','OTHER'])

    dest = st.selectbox("Destination airport", ['ATL','DFW','CLT','SFO','IAH','LAS','DTW','SEA','ORD','OTHER'])
    st.markdown('</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("⬅ Back"):
            go_to('login')
            st.rerun()
    with colB:
        if st.button("Check my flight ✈"):
            row = pd.DataFrame([[0] * len(feature_columns)], columns=feature_columns)
            row['month'] = month
            row['day_of_week'] = day_of_week
            row['hour'] = hour
            row['DISTANCE'] = distance

            for prefix, val in [('OP_CARRIER_', carrier), ('ORIGIN_', origin), ('DEST_', dest)]:
                col = f'{prefix}{val}'
                if col in row.columns:
                    row[col] = 1

            prob = model.predict_proba(row)[:, 1][0]
            st.session_state.prediction = {
                'prob': prob,
                'is_delayed': prob > 0.20
            }
            go_to('result')
            st.rerun()

# ---------- RESULT PAGE ----------
elif st.session_state.page == 'result':
    st.markdown('<div class="hero-title">Your flight forecast 🌥️</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-divider">☁ ☁ ☁</div>', unsafe_allow_html=True)

    pred = st.session_state.prediction
    if pred:
        status_class = "result-delayed" if pred['is_delayed'] else "result-ontime"
        status_text = "Likely Delayed" if pred['is_delayed'] else "Likely On Time"
        emoji = "🌧️" if pred['is_delayed'] else "☀️"

        st.markdown(f"""
        <div class="result-card {status_class}">
            <div style="font-size:2.4rem;">{emoji}</div>
            <div class="result-status">{status_text}</div>
            <div class="result-prob">Delay probability: {pred['prob']:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Check another flight"):
            go_to('input')
            st.rerun()
    with col2:
        if st.button("Log out"):
            st.session_state.username = ''
            st.session_state.prediction = None
            go_to('login')
            st.rerun()
