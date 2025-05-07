# models/model_utils.py
import joblib
import os

def load_model(model_path):
    """
    Load a machine learning model from the specified file path.
    
    Parameters:
    - model_path (str): The file path to the model file.
    
    Returns:
    - model: The loaded model.
    
    Raises:
    - FileNotFoundError: If the model file does not exist.
    - ValueError: If the model loading fails due to incompatible file formats.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except ValueError as e:
        print(f"ValueError: Unable to load model from {model_path} - {e}")
        raise
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise

def save_model(model, model_path):
    """
    Save a machine learning model to the specified file path.
    
    Parameters:
    - model: The machine learning model to be saved.
    - model_path (str): The file path where the model should be saved.
    
    Raises:
    - IOError: If there is an issue with writing to the specified path.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the model to the specified path
        joblib.dump(model, model_path)
        print(f"Model saved successfully to {model_path}")
    except IOError as e:
        print(f"IOError: Unable to save model to {model_path} - {e}")
        raise
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")
        raise

# Example usage:
if __name__ == "__main__":
    try:
        # Test loading and saving a model
        model = load_model("models/example_model.joblib")
        save_model(model, "models/example_model_copy.joblib")
    except Exception as e:
        print(f"Error in model operations: {e}")
