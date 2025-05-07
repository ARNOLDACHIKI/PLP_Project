from sklearn.tree import DecisionTreeRegressor

def create_edge_model(max_depth=5, min_samples_leaf=5, random_state=42):
    """
    Function to create and return a DecisionTreeRegressor model with specific parameters.
    
    Args:
    - max_depth (int): The maximum depth of the tree.
    - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
    - random_state (int): The seed used by the random number generator.
    
    Returns:
    - model: A DecisionTreeRegressor model.
    """
    return DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)

class EdgeModel:
    def __init__(self, max_depth=5, min_samples_leaf=5, random_state=42, model=None):
        """
        Initialize the EdgeModel with a DecisionTreeRegressor or use a custom model.
        
        Args:
        - model (optional): A pre-trained model to use. If not provided, a new model is created using create_edge_model().
        """
        self.model = model if model else create_edge_model(max_depth, min_samples_leaf, random_state)

    def fit(self, X, y):
        """
        Train the model with the provided data.
        
        Args:
        - X: The feature data.
        - y: The target labels.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the target labels for the provided feature data.
        
        Args:
        - X: The feature data.
        
        Returns:
        - The predicted labels.
        """
        return self.model.predict(X)

    def get_model(self):
        """
        Return the underlying model (e.g., for saving, exporting).
        
        Returns:
        - model: The underlying DecisionTreeRegressor model.
        """
        return self.model

    def set_params(self, max_depth=5, min_samples_leaf=5, random_state=42):
        """
        Set new parameters for the model.
        
        Args:
        - max_depth (int): The maximum depth of the tree.
        - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        - random_state (int): The seed used by the random number generator.
        """
        self.model.set_params(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
