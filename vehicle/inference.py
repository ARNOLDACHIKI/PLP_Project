# inference.py
def predict_battery_life(model, input_features):
    return model.predict([input_features])[0]
