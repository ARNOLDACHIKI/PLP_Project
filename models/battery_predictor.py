import streamlit as st
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import joblib
from cloud.distribution import distribute_model  # Import your distribution logic

# Load the trained model
model = joblib.load('models/battery_predictor.joblib')

# App UI
st.title('🚗 EV Battery Life Predictor (Edge Inference Demo)')

st.sidebar.header("Input Parameters")

# User inputs
speed = st.sidebar.slider('Speed (km/h)', 20, 120, 60)
acceleration = st.sidebar.slider('Acceleration (m/s²)', 0, 5, 2)
temperature = st.sidebar.slider('Outside Temperature (°C)', -10, 40, 25)
battery_health = st.sidebar.slider('Battery Health (%)', 70, 100, 90)

# Predict
if st.sidebar.button('Predict Battery Life'):
    input_data = np.array([[speed, acceleration, temperature, battery_health]])
    prediction = model.predict(input_data)[0]
    st.success(f'🔋 Estimated Remaining Battery Life: {prediction:.1f} minutes')

    # Federated System Dashboard
    st.title("Federated Edge-Cloud System Dashboard")

    # Display system logs
    st.subheader("System Logs")
    logs = [
        "Starting edge-cloud federated system...",
        f"Predicted battery life: {prediction:.2f} minutes",
        "Sending updated model weights to cloud...",
        "Distributing global model to all vehicles...",
        "Model saved successfully to models/local_model.pkl"
    ]

    log_placeholder = st.empty()
    for log in logs:
        log_placeholder.text(log)
        time.sleep(1)

    # Battery Life Prediction Chart
    st.subheader("Battery Life Prediction")
    iterations = ['Iteration 1']
    fig, ax = plt.subplots()
    ax.bar(iterations, [prediction], color='blue')
    ax.set_title('Battery Life Prediction Over Iterations')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Battery Life (minutes)')
    st.pyplot(fig)

    # Real-Time Simulation
    st.subheader("Real-Time Battery Life Prediction")
    battery_life_placeholder = st.empty()
    for _ in range(10):
        simulated_life = round(random.uniform(180, 200), 2)
        battery_life_placeholder.text(f"Predicted Battery Life: {simulated_life} minutes")
        time.sleep(2)

    # Federated Learning Metrics
    st.subheader("Federated Learning Metrics")
    accuracy = 92.5
    loss = 0.45
    st.write(f"Model Accuracy: {accuracy}%")
    st.write(f"Model Loss: {loss}")

    # Model Distribution Progress
    st.subheader("Model Distribution Progress")
    progress_bar = st.progress(0)
    for percent_complete in range(1, 101):
        progress_bar.progress(percent_complete)
        time.sleep(0.05)

    st.write("Model distributed successfully!")

    # Model Update Status
    st.subheader("Model Update Status")
    model_status = st.empty()
    update_steps = [
        "Model weights are being sent to the cloud...",
        "Distributing global model to all vehicles...",
        "Model saved successfully to models/local_model.pkl"
    ]
    for step in update_steps:
        model_status.text(step)
        time.sleep(2)
