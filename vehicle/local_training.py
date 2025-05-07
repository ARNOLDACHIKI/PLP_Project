from models.edge_model import create_edge_model

def local_train(X, y, max_depth=5, min_samples_leaf=5, random_state=42):
    # Use create_edge_model with the desired parameters
    model = create_edge_model(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    
    # Train the model
    model.fit(X, y)
    
    # Return the trained model
    return model
