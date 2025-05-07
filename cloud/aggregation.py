import numpy as np
import joblib
from models.edge_model import EdgeModel  # Assuming you have a custom model class
from models.model_utils import load_model, save_model  # Assuming these utilities are defined

def aggregate_models(model_paths, aggregation_type='average'):
    """
    Aggregates multiple models into a single model.

    Args:
        model_paths (list): List of file paths to the models to aggregate.
        aggregation_type (str): Type of aggregation ('average', 'weighted', etc.).

    Returns:
        model: Aggregated model.
    """
    models = [load_model(path) for path in model_paths]
    
    if aggregation_type == 'average':
        return average_models(models)
    elif aggregation_type == 'weighted':
        return weighted_aggregation(models)
    else:
        raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

def average_models(models):
    """
    Averages the weights of the models.

    Args:
        models (list): List of models to average.

    Returns:
        model: The aggregated model with averaged weights.
    """
    # Assuming models are of the same architecture and have the same parameters
    # Convert models to numpy arrays for weights
    model_weights = [model.get_weights() for model in models]
    avg_weights = np.mean(model_weights, axis=0)
    
    # Assuming the model is a neural network, this would set the weights to the average
    aggregated_model = models[0]
    aggregated_model.set_weights(avg_weights)
    
    return aggregated_model

def weighted_aggregation(models):
    """
    Perform weighted aggregation of model weights.
    
    Args:
        models (list): List of models to aggregate.
    
    Returns:
        model: The aggregated model with weighted averages.
    """
    weights = [0.5, 0.5]  # Example of weights, adjust as needed
    model_weights = [model.get_weights() for model in models]
    
    weighted_weights = np.average(model_weights, axis=0, weights=weights)
    
    aggregated_model = models[0]
    aggregated_model.set_weights(weighted_weights)
    
    return aggregated_model

def save_aggregated_model(aggregated_model, save_path="cloud/aggregated_model.joblib"):
    """
    Save the aggregated model to a file.

    Args:
        aggregated_model: The model to save.
        save_path (str): The path where the model will be saved.
    """
    joblib.dump(aggregated_model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Example usage:
    model_paths = ['models/battery_predictor.joblib', 'models/edge_model.py']  # Add paths of models to aggregate
    aggregated_model = aggregate_models(model_paths, aggregation_type='average')
    save_aggregated_model(aggregated_model)
