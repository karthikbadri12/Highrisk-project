"""
Configuration management for the high-risk AI healthcare project.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for managing project settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = config_path or self.project_root / "config.yaml"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or create default."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "project": {
                "name": "high-risk-ai-healthcare",
                "version": "1.0.0",
                "description": "High-risk AI project in healthcare"
            },
            "paths": {
                "data": str(self.project_root / "data"),
                "results": str(self.project_root / "results"),
                "models": str(self.project_root / "models"),
                "logs": str(self.project_root / "logs")
            },
            "data": {
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
                "random_seed": 42
            },
            "model": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 100,
                "early_stopping_patience": 10
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "save_predictions": True,
                "save_model": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save configuration to file."""
        os.makedirs(self.config_path.parent, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        for path_key in ["data", "results", "models", "logs"]:
            path = Path(self.get(f"paths.{path_key}"))
            path.mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = Config() 