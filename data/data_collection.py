# data_collection.py
import pandas as pd
import random

def collect_vehicle_data(num_samples=100):
    # Simulate real-time data collection from a vehicle
    data = {
        'speed': [random.uniform(0, 120) for _ in range(num_samples)],
        'acceleration': [random.uniform(0, 5) for _ in range(num_samples)],
        'temperature': [random.uniform(15, 40) for _ in range(num_samples)],
        'battery_health': [random.uniform(70, 100) for _ in range(num_samples)],
        'battery_life_minutes': [random.uniform(30, 300) for _ in range(num_samples)]
    }
    return pd.DataFrame(data)
