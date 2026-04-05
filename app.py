import streamlit as st
import joblib
import numpy as np

# Load model and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('kmeans_k4.pkl')

# Cluster labels based on your results
cluster_profiles = {
    0: ("Moderate Older User",
        "Your phone usage is steady and intentional. You know what you use it for and stick to that — no endless scrolling or chasing new apps. Usage is spread across the day rather than concentrated in long sessions, and your data consumption reflects someone who's online but not constantly."),
    1: ("Power User",
        "Your phone sees heavy use across the board , long screen time, high data, lots of apps. You're likely juggling multiple things at once, whether that's work, social media, or entertainment. The phone is rarely far from reach and almost always in use."),
    2: ("Minimal User",
        "You keep phone usage to the essentials. Screen time is low, data consumption is minimal, and the apps installed reflect actual needs rather than impulse downloads. You're reachable, but the phone doesn't dominate your day."),
    3: ("Active Young User",
        "You have strong engagement across apps and a decent amount of screen time, but it's not excessive. Social media, entertainment, and messaging likely make up the bulk of it. You're comfortable with technology and pick up new apps fairly easily.")
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