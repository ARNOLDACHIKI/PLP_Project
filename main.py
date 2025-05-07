from data.data_collection import collect_vehicle_data
from data.data_preprocessing import preprocess_data
from data.data_utils import save_data
from vehicle.local_training import local_train
from vehicle.inference import predict_battery_life
from vehicle.data_sender import send_model_update
from cloud.aggregation import aggregate_models
from cloud.global_model import create_global_model
from cloud.distribution import distribute_model
from models.model_utils import save_model
import numpy as np
import os

def main():
    print("Starting edge-cloud federated system...")

    # Simulate vehicles collecting and training locally
    vehicle_data = collect_vehicle_data(200)
    vehicle_data = preprocess_data(vehicle_data)

    # Prepare input data (features and labels)
    X = vehicle_data[['speed', 'acceleration', 'temperature', 'battery_health']]
    y = vehicle_data['battery_life_minutes']

    # Local training
    local_model = local_train(X, y)

    # Local inference example
    sample_input = [0.5, 0.3, 0.6, 0.9]  # normalized input
    prediction = predict_battery_life(local_model, sample_input)
    print(f"Predicted battery life: {prediction:.2f} minutes")

    # Send model update to cloud
    send_model_update(local_model)

    # Cloud aggregation simulation
    models = [local_model]  # Add other local models as needed for aggregation
    global_model = create_global_model(models, X)  # Pass X for prediction in global model aggregation

    # Distribute the global model to other vehicles or cloud systems
    distribute_model(global_model)

    # Save the local model to a file
    model_path = os.path.join("models", "local_model.pkl")  # Define the path where to save the model
    save_model(local_model, model_path)

if __name__ == "__main__":
    main()
