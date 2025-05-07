# models/__init__.py

# Importing essential functions or models for easy access
from .edge_model import EdgeModel, create_edge_model
from .model_utils import load_model, save_model

# If you have other models or utilities, you can import them here
# from .other_model import SomeModel

# Optionally, you can define an `__all__` list to specify what gets imported
# when `from models import *` is used.
__all__ = ["EdgeModel", "create_edge_model", "load_model", "save_model"]
