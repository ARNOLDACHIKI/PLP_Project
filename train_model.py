# train_model.py

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import joblib

# --- Step 1: Ensure directories exist ---
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- Step 2: Generate synthetic data ---
np.random.seed(42)
n_samples = 1000

speed = np.random.uniform(20, 120, n_samples)            # 20km/h to 120km/h
acceleration = np.random.uniform(0, 5, n_samples)         # 0m/s² to 5m/s²
temperature = np.random.uniform(-10, 40, n_samples)       # -10°C to 40°C
battery_health = np.random.uniform(70, 100, n_samples)    # 70% to 100%

battery_life_minutes = (battery_health * 2) - (speed * 0.3) - (acceleration * 10) + (temperature * 0.5)
battery_life_minutes += np.random.normal(0, 5, n_samples)  # Add noise

df = pd.DataFrame({
    'speed': speed,
    'acceleration': acceleration,
    'temperature': temperature,
    'battery_health': battery_health,
    'battery_life_minutes': battery_life_minutes
})

# Save synthetic dataset
data_path = 'data/synthetic_battery_data.csv'
df.to_csv(data_path, index=False)
print(f"✅ Synthetic dataset saved to {data_path}")

# --- Step 3: Load the dataset ---
data = pd.read_csv(data_path)

# Select features and target
features = ['speed', 'acceleration', 'temperature', 'battery_health']
target = 'battery_life_minutes'

X = data[features]
y = data[target]

# --- Step 4: Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 5: Train an optimized & pruned Decision Tree model ---
model = DecisionTreeRegressor(
    max_depth=4,             # smaller depth (more pruning)
    min_samples_leaf=10,     # minimum 10 samples per leaf
    min_samples_split=20,    # minimum 20 samples to split
    random_state=42
)

model.fit(X_train, y_train)

# --- Step 6: Evaluate model ---
y_pred = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
print(f"✅ Model trained and pruned. Test MAE: {mae:.2f} minutes")

# --- Step 7: Save the model with compression ---
model_path = 'models/battery_predictor.joblib'
joblib.dump(model, model_path, compress=3)
print(f"✅ Optimized model saved with compression to {model_path}")
