# Data Exploration Template
# This script can be converted to a Jupyter notebook for interactive exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json

# Add src to path for imports
sys.path.append(str(Path.cwd().parent / "src"))

# Import project utilities
from utils import config, logger

def load_data():
    """Load your dataset here."""
    # Replace with your actual data loading code
    # Example:
    # data_path = Path(config.get("paths.data")) / "your_dataset.csv"
    # df = pd.read_csv(data_path)
    
    # For demonstration, create sample data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'patient_id': range(1, n_samples + 1),
        'age': np.random.normal(65, 15, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'blood_pressure_systolic': np.random.normal(120, 20, n_samples),
        'blood_pressure_diastolic': np.random.normal(80, 10, n_samples),
        'heart_rate': np.random.normal(75, 15, n_samples),
        'temperature': np.random.normal(98.6, 1, n_samples),
        'diagnosis': np.random.choice(['normal', 'hypertension', 'diabetes', 'heart_disease'], 
                                    n_samples, p=[0.6, 0.2, 0.15, 0.05])
    })
    
    return df

def basic_info(df):
    """Display basic dataset information."""
    print("=== BASIC DATASET INFORMATION ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

def statistical_summary(df):
    """Display statistical summary."""
    print("\n=== STATISTICAL SUMMARY ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print("Numerical columns summary:")
    print(df.describe())
    
    print("\nCategorical columns summary:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

def analyze_outliers(df):
    """Analyze outliers in numerical columns."""
    print("\n=== OUTLIER ANALYSIS ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    outliers_summary = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_summary[col] = len(outliers)
    
    for col, count in outliers_summary.items():
        percentage = (count / len(df)) * 100
        print(f"{col}: {count} outliers ({percentage:.2f}%)")
    
    return outliers_summary

def plot_distributions(df):
    """Plot distributions of features."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Plot numerical distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Plot categorical distributions
    if len(categorical_cols) > 0:
        fig, axes = plt.subplots(1, len(categorical_cols), figsize=(15, 5))
        if len(categorical_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(categorical_cols):
            value_counts = df[col].value_counts()
            axes[i].bar(value_counts.index, value_counts.values)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()

def correlation_analysis(df):
    """Analyze correlations between numerical features."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

def target_analysis(df, target_col='diagnosis'):
    """Analyze target variable and its relationships with features."""
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataset")
        return
    
    print(f"\n=== TARGET VARIABLE ANALYSIS ===")
    print(f"Target variable '{target_col}' distribution:")
    target_dist = df[target_col].value_counts()
    print(target_dist)
    
    # Plot target distribution
    plt.figure(figsize=(10, 6))
    target_dist.plot(kind='bar')
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_exploration_results(df, outliers_summary):
    """Save exploration results to file."""
    exploration_summary = {
        'dataset_shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'outliers': outliers_summary,
        'data_types': df.dtypes.to_dict(),
        'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    results_path = Path(config.get("paths.results")) / "data_exploration_summary.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(exploration_summary, f, indent=2, default=str)
    
    print(f"\nExploration summary saved to: {results_path}")

def main():
    """Main exploration function."""
    logger.info("Starting data exploration")
    
    # Load data
    df = load_data()
    
    # Basic information
    basic_info(df)
    
    # Statistical summary
    statistical_summary(df)
    
    # Outlier analysis
    outliers_summary = analyze_outliers(df)
    
    # Plot distributions
    plot_distributions(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Target analysis
    target_analysis(df)
    
    # Save results
    save_exploration_results(df, outliers_summary)
    
    logger.info("Data exploration completed")

if __name__ == "__main__":
    main() 