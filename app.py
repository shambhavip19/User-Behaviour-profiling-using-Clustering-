import streamlit as st
import joblib
import numpy as np

# Load model and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('kmeans_k4.pkl')

# Cluster labels based on your results
cluster_profiles = {
    0: ("Moderate Older User", "You use your phone steadily and consistently. Not too much, not too little — just habitual, reliable usage."),
    1: ("Power User", "You are heavily engaged with your phone. High screen time, high data, high everything. Your phone is basically an extension of you."),
    2: ("Minimal User", "You use your phone sparingly. Calls, maybe messages — that's about it. Your battery thanks you."),
    3: ("Active Young User", "You're active on your phone with solid app usage. Likely social media, entertainment, or gaming keeps you engaged.")
}

st.title("What kind of phone user are you?")
st.write("Adjust the sliders based on your daily phone habits and find out your user profile.")

st.divider()

app_usage = st.slider("App Usage Time (min/day)", 0, 600, 200)
screen_time = st.slider("Screen On Time (hours/day)", 0.0, 12.0, 4.0)
battery_drain = st.slider("Battery Drain (mAh/day)", 300, 3500, 1500)
num_apps = st.slider("Number of Apps Installed", 10, 100, 40)
data_usage = st.slider("Data Usage (MB/day)", 100, 2500, 800)
age = st.slider("Your Age", 18, 70, 30)

st.divider()

if st.button("Find My Profile"):
    user_input = np.array([[app_usage, screen_time, battery_drain, num_apps, data_usage, age]])
    user_scaled = scaler.transform(user_input)
    cluster = model.predict(user_scaled)[0]

    label, description = cluster_profiles[cluster]

    st.subheader(f"You are: {label}")
    st.write(description)