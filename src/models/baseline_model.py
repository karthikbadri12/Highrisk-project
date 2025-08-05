"""
Baseline model implementation for healthcare tasks.
This template can be adapted for classification, regression, or other tasks.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from ..utils import config, logger

class BaselineModel:
    """Baseline model class for healthcare tasks."""
    
    def __init__(self, task_type: str = "classification", model_type: str = "random_forest"):
        """
        Initialize baseline model.
        
        Args:
            task_type: "classification" or "regression"
            model_type: "random_forest", "logistic_regression", or "linear_regression"
        """
        self.task_type = task_type
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
        # Initialize model based on type
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model based on task and model type."""
        if self.task_type == "classification":
            if self.model_type == "random_forest":
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.model_type == "logistic_regression":
                self.model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                raise ValueError(f"Unknown classification model type: {self.model_type}")
        elif self.task_type == "regression":
            if self.model_type == "random_forest":
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == "linear_regression":
                self.model = LinearRegression()
            else:
                raise ValueError(f"Unknown regression model type: {self.model_type}")
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training.
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = self.label_encoder.fit_transform(X[col])
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the baseline model.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary containing training results
        """
        logger.info(f"Training {self.model_type} baseline model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if self.task_type == "classification" else None
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        
        # Store results
        results = {
            'model_type': self.model_type,
            'task_type': self.task_type,
            'metrics': metrics,
            'feature_importance': self._get_feature_importance(),
            'test_predictions': y_test_pred,
            'test_true': y_test
        }
        
        logger.info(f"Training completed. Test accuracy: {metrics['test_accuracy']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_train: np.ndarray, y_train_pred: np.ndarray, 
                          y_test: np.ndarray, y_test_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        if self.task_type == "classification":
            # Classification metrics
            metrics['train_accuracy'] = np.mean(y_train == y_train_pred)
            metrics['test_accuracy'] = np.mean(y_test == y_test_pred)
            
            # Additional classification metrics
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
            
        elif self.task_type == "regression":
            # Regression metrics
            metrics['train_r2'] = r2_score(y_train, y_train_pred)
            metrics['test_r2'] = r2_score(y_test, y_test_pred)
            metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
            metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        return metrics
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {}
            for i, importance in enumerate(self.model.feature_importances_):
                feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
                importance_dict[feature_name] = importance
            return importance_dict
        return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Apply preprocessing
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """Save the trained model."""
        if model_path is None:
            model_dir = Path(config.get("paths.models"))
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"baseline_{self.model_type}_{self.task_type}.joblib"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
    @classmethod
    def load_model(cls, model_path: str) -> 'BaselineModel':
        """Load a trained model."""
        model_data = joblib.load(model_path)
        
        # Create instance
        instance = cls(task_type=model_data['task_type'], model_type=model_data['model_type'])
        
        # Load components
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoder = model_data['label_encoder']
        instance.feature_names = model_data['feature_names']
        
        return instance

def run_baseline_experiment(df: pd.DataFrame, target_col: str, task_type: str = "classification") -> Dict[str, Any]:
    """
    Run a complete baseline experiment.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        task_type: "classification" or "regression"
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("Starting baseline experiment")
    
    # Initialize model
    model = BaselineModel(task_type=task_type)
    
    # Preprocess data
    X, y = model.preprocess_data(df, target_col)
    
    # Train model
    results = model.train(X, y)
    
    # Save model
    model_path = model.save_model()
    results['model_path'] = model_path
    
    # Save results
    results_path = Path(config.get("paths.results")) / "baseline_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Baseline experiment completed. Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Run baseline experiment
    results = run_baseline_experiment(df, 'target', 'classification')
    print("Baseline experiment results:", results) 