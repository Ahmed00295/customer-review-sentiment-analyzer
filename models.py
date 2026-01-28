"""
Models Module - ML Model Wrapper Classes for Sentiment Analysis
================================================================

This module demonstrates core OOP principles:
1. ABSTRACTION - Abstract base class defines the interface
2. INHERITANCE - Concrete classes inherit from base class
3. POLYMORPHISM - Same interface, different implementations
4. ENCAPSULATION - Model details hidden within classes

The module provides multiple ML models that can be swapped
without changing the main execution logic (Polymorphism).

Author: Muhammad Ahmad
Date: January 2026
"""

from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import joblib
from typing import Any, Optional


class BaseModel(ABC):
    """
    Abstract Base Class for all ML models.
    
    This class demonstrates ABSTRACTION by:
    1. Defining a common interface (train, predict) that all models must implement
    2. Using @abstractmethod decorator to enforce implementation in subclasses
    3. Providing common functionality that can be inherited
    
    Note: Abstract classes cannot be instantiated directly.
    You must create a subclass that implements all abstract methods.
    
    OOP Principles:
        - ABSTRACTION: Defines what methods must exist without implementing them
        - INHERITANCE: Subclasses inherit from this base class
    """
    
    def __init__(self):
        """Initialize the base model with common attributes."""
        self._model = None  # Protected attribute (single underscore)
        self._is_trained = False
        self._model_name = "BaseModel"
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given data.
        
        This is an ABSTRACT METHOD - must be implemented by subclasses.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        This is an ABSTRACT METHOD - must be implemented by subclasses.
        
        Args:
            X: Feature matrix to predict on
            
        Returns:
            Predicted labels for each sample
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model.
        
        This is an ABSTRACT METHOD - must be implemented by subclasses.
        
        Returns:
            String name of the model
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (if supported by the model).
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability estimates for each class
        """
        if hasattr(self._model, 'predict_proba'):
            return self._model.predict_proba(X)
        else:
            # Return dummy probabilities for models without probability support
            predictions = self.predict(X)
            return np.column_stack([1 - predictions.astype(float), 
                                   predictions.astype(float)])
    
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before saving.")
        joblib.dump(self._model, filepath)
        print(f"[SAVED] Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self._model = joblib.load(filepath)
        self._is_trained = True
        print(f"[LOADED] Model loaded from: {filepath}")


class SVMModel(BaseModel):
    """
    Support Vector Machine model for sentiment classification.
    
    SVM works by finding the optimal hyperplane that separates
    different classes in high-dimensional space.
    
    OOP Principles Demonstrated:
        - INHERITANCE: Inherits from BaseModel
        - POLYMORPHISM: Implements train() and predict() differently
        - ENCAPSULATION: Hides SVC implementation details
    """
    
    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        """
        Initialize SVM model with configurable parameters.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
        """
        super().__init__()  # Call parent constructor
        self.__model = SVC(
            kernel=kernel,
            C=C,
            probability=True,  # Enable probability estimates
            random_state=42
        )
        self._model = self.__model
        self._model_name = "Support Vector Machine (SVM)"
        self.__kernel = kernel
        self.__C = C
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM model.
        
        POLYMORPHISM: This implementation is specific to SVM,
        but follows the same interface as other models.
        
        Args:
            X: TF-IDF feature matrix
            y: Sentiment labels
        """
        print(f"[TRAINING] {self._model_name}...")
        print(f"   Kernel: {self.__kernel} | C: {self.__C}")
        
        self.__model.fit(X, y)
        self._is_trained = True
        
        print(f"[OK] {self._model_name} trained successfully!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained SVM model.
        
        Args:
            X: Feature matrix to predict on
            
        Returns:
            Predicted sentiment labels
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.__model.predict(X)
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self._model_name
    
    def get_support_vectors_count(self) -> int:
        """Get the number of support vectors (SVM-specific method)."""
        if self._is_trained:
            return len(self.__model.support_vectors_)
        return 0


class RandomForestModel(BaseModel):
    """
    Random Forest classifier for sentiment analysis.
    
    Random Forest is an ensemble method that builds multiple
    decision trees and merges their predictions.
    
    OOP Principles Demonstrated:
        - INHERITANCE: Inherits from BaseModel
        - POLYMORPHISM: Same interface, different algorithm
        - ENCAPSULATION: Forest details hidden within class
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
        """
        super().__init__()
        self.__model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        self._model = self.__model
        self._model_name = "Random Forest"
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest model.
        
        POLYMORPHISM: Same method signature as SVM,
        but trains a completely different algorithm.
        
        Args:
            X: TF-IDF feature matrix
            y: Sentiment labels
        """
        print(f"[TRAINING] {self._model_name}...")
        print(f"   Trees: {self.__n_estimators} | Max Depth: {self.__max_depth}")
        
        self.__model.fit(X, y)
        self._is_trained = True
        
        print(f"[OK] {self._model_name} trained successfully!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using Random Forest."""
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.__model.predict(X)
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self._model_name
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances (Random Forest-specific method)."""
        if self._is_trained:
            return self.__model.feature_importances_
        return np.array([])


class KNNModel(BaseModel):
    """
    K-Nearest Neighbors classifier for sentiment analysis.
    
    KNN classifies samples based on the majority class among
    the k nearest neighbors in the feature space.
    
    OOP Principles Demonstrated:
        - INHERITANCE: Inherits common functionality from BaseModel
        - POLYMORPHISM: Implements the same interface differently
        - ENCAPSULATION: KNN algorithm details encapsulated
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        """
        Initialize KNN model.
        
        Args:
            n_neighbors: Number of neighbors to consider
            weights: Weight function ('uniform' or 'distance')
        """
        super().__init__()
        self.__model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            n_jobs=-1
        )
        self._model = self.__model
        self._model_name = "K-Nearest Neighbors (KNN)"
        self.__n_neighbors = n_neighbors
        self.__weights = weights
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the KNN model.
        
        Note: KNN is a lazy learner - it doesn't truly "train"
        but stores the training data for later comparison.
        
        Args:
            X: TF-IDF feature matrix
            y: Sentiment labels
        """
        print(f"[TRAINING] {self._model_name}...")
        print(f"   Neighbors: {self.__n_neighbors} | Weights: {self.__weights}")
        
        self.__model.fit(X, y)
        self._is_trained = True
        
        print(f"[OK] {self._model_name} trained successfully!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using KNN."""
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.__model.predict(X)
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self._model_name


class NaiveBayesModel(BaseModel):
    """
    Naive Bayes classifier - excellent for text classification.
    
    Multinomial Naive Bayes is particularly suited for text data
    with TF-IDF or count-based features.
    
    OOP Principles: Inheritance, Polymorphism, Encapsulation
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Naive Bayes model.
        
        Args:
            alpha: Smoothing parameter (Laplace smoothing)
        """
        super().__init__()
        self.__model = MultinomialNB(alpha=alpha)
        self._model = self.__model
        self._model_name = "Naive Bayes"
        self.__alpha = alpha
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Naive Bayes model."""
        print(f"[TRAINING] {self._model_name}...")
        print(f"   Alpha (smoothing): {self.__alpha}")
        
        self.__model.fit(X, y)
        self._is_trained = True
        
        print(f"[OK] {self._model_name} trained successfully!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using Naive Bayes."""
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.__model.predict(X)
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self._model_name


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier for sentiment analysis.
    
    A simple but effective linear model for binary classification.
    
    OOP Principles: Inheritance, Polymorphism, Encapsulation
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """
        Initialize Logistic Regression model.
        
        Args:
            C: Inverse regularization strength
            max_iter: Maximum iterations for solver
        """
        super().__init__()
        self.__model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            n_jobs=-1
        )
        self._model = self.__model
        self._model_name = "Logistic Regression"
        self.__C = C
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Logistic Regression model."""
        print(f"[TRAINING] {self._model_name}...")
        print(f"   Regularization (C): {self.__C}")
        
        self.__model.fit(X, y)
        self._is_trained = True
        
        print(f"[OK] {self._model_name} trained successfully!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using Logistic Regression."""
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.__model.predict(X)
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self._model_name


class ModelFactory:
    """
    Factory class for creating model instances.
    
    This class demonstrates the FACTORY PATTERN - a creational
    design pattern that provides an interface for creating objects.
    
    Usage:
        model = ModelFactory.create_model("svm")
    """
    
    # Dictionary mapping model names to their classes
    _models = {
        'svm': SVMModel,
        'random_forest': RandomForestModel,
        'knn': KNNModel,
        'naive_bayes': NaiveBayesModel,
        'logistic_regression': LogisticRegressionModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        Create and return a model instance.
        
        This method demonstrates POLYMORPHISM - it returns different
        model types through the same interface.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments for the model constructor
            
        Returns:
            An instance of the requested model type
            
        Raises:
            ValueError: If the model type is not recognized
        """
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available models: {available}")
        
        return cls._models[model_type](**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_display_names(cls) -> dict:
        """Get display names for all models."""
        return {
            'svm': 'Support Vector Machine (SVM)',
            'random_forest': 'Random Forest',
            'knn': 'K-Nearest Neighbors (KNN)',
            'naive_bayes': 'Naive Bayes',
            'logistic_regression': 'Logistic Regression'
        }


# Demonstration of POLYMORPHISM
def demonstrate_polymorphism():
    """
    Demonstrate how different models can be used interchangeably.
    
    This function shows POLYMORPHISM in action - the same code
    works with any model that inherits from BaseModel.
    """
    print("=" * 60)
    print("POLYMORPHISM DEMONSTRATION")
    print("=" * 60)
    
    # Create different models using the factory
    models = [
        ModelFactory.create_model('svm'),
        ModelFactory.create_model('random_forest'),
        ModelFactory.create_model('knn')
    ]
    
    # Same interface works for all models (POLYMORPHISM)
    for model in models:
        print(f"\n  Model: {model.get_model_name()}")
        print(f"   Is Trained: {model.is_trained()}")
        # model.train(X, y)  # Same method call for all models
        # model.predict(X)   # Same method call for all models


if __name__ == "__main__":
    demonstrate_polymorphism()
    
    print("\n" + "=" * 60)
    print("Available Models:")
    print("=" * 60)
    for key, name in ModelFactory.get_model_display_names().items():
        print(f"   • {key}: {name}")
