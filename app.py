import streamlit as st
import pickle
import pandas as pd
import json

with open('flight_delay_lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

st.title("Flight Delay Risk Predictor")

month = st.selectbox("Month", list(range(1,13)))
day_of_week = st.selectbox("Day of Week (0=Mon)", list(range(0,7)))
hour = st.slider("Departure Hour", 0, 23)
distance = st.number_input("Distance (miles)", min_value=0, value=500)
carrier = st.selectbox("Carrier", ['AA','DL','UA','WN','B6','AS','NK','F9','HA','G4','OO','YX','MQ','OH','YV','VX','EV'])
origin = st.selectbox("Origin Airport", ['ATL','DFW','CLT','SFO','IAH','LAS','DTW','SEA','ORD','OTHER'])
dest = st.selectbox("Destination Airport", ['ATL','DFW','CLT','SFO','IAH','LAS','DTW','SEA','ORD','OTHER'])

if st.button("Predict"):
    row = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)
    row['month'] = month
    row['day_of_week'] = day_of_week
    row['hour'] = hour
    row['DISTANCE'] = distance

    for prefix, val in [('OP_CARRIER_', carrier), ('ORIGIN_', origin), ('DEST_', dest)]:
        col = f'{prefix}{val}'
        if col in row.columns:
            row[col] = 1

    prob = model.predict_proba(row)[:, 1][0]
    prediction = "Delayed" if prob > 0.20 else "On-time"
    st.write(f"### Prediction: {prediction}")
    st.write(f"Delay probability: {prob:.2%}")
