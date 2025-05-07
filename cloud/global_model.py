import numpy as np

def create_global_model(models, X):
    """
    Aggregates the predictions of multiple models and creates a global model based on them.

    Args:
    - models (list): List of models to aggregate (trained models).
    - X (array-like): The input data to be passed to the models for prediction.

    Returns:
    - global_model: A model that is the result of aggregating the predictions of the input models.
    """
    # Aggregate model predictions
    aggregated_predictions = aggregate_models(models, X)
    
    # Here you can create and return a new model based on aggregated predictions.
    # For simplicity, we'll just return the aggregated predictions for now.
    return aggregated_predictions

def aggregate_models(models, X):
    """
    Aggregates the predictions of models by averaging them.

    Args:
    - models (list): List of models to aggregate.
    - X (array-like): The input data to be passed to the models for prediction.

    Returns:
    - predictions: Aggregated predictions from all models.
    """
    # Simple averaging of model predictions (simulation)
    # Assuming all models take the same input format and produce similar output
    all_predictions = [model.predict(X) for model in models]  # Now using `X` as input to models
    aggregated_predictions = np.mean(all_predictions, axis=0)
    return aggregated_predictions
