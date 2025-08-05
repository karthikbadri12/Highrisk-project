"""
Real healthcare dataset loader for the high-risk AI project.
This module provides access to various online healthcare datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import os
from typing import Dict, List, Optional, Tuple
import json

from ..utils import config, logger

class RealDatasetLoader:
    """
    Loader for real healthcare datasets from online sources.
    """
    
    def __init__(self):
        self.data_path = Path(config.get("paths.data"))
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def download_physionet_dataset(self, dataset_name: str, target_dir: str = None) -> str:
        """
        Download dataset from PhysioNet.
        
        Args:
            dataset_name: Name of the PhysioNet dataset
            target_dir: Directory to save the dataset
            
        Returns:
            Path to downloaded dataset
        """
        if target_dir is None:
            target_dir = self.data_path / dataset_name
        else:
            target_dir = Path(target_dir)
            
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # PhysioNet datasets require approval, so we'll provide instructions
        logger.info(f"To download {dataset_name} from PhysioNet:")
        logger.info("1. Go to https://physionet.org/")
        logger.info("2. Search for the dataset")
        logger.info("3. Request access (free but requires approval)")
        logger.info("4. Download and extract to: " + str(target_dir))
        
        return str(target_dir)
    
    def load_mimic_sample(self, mimic_path: str) -> pd.DataFrame:
        """
        Load sample data from MIMIC-III/IV (if available).
        
        Args:
            mimic_path: Path to MIMIC data directory
            
        Returns:
            DataFrame with clinical notes
        """
        mimic_path = Path(mimic_path)
        
        if not mimic_path.exists():
            logger.warning(f"MIMIC data not found at {mimic_path}")
            logger.info("To use MIMIC data:")
            logger.info("1. Request access at https://mimic.mit.edu/")
            logger.info("2. Download and extract to the specified path")
            return pd.DataFrame()
        
        # Example structure for MIMIC data
        # This would need to be adapted based on actual MIMIC structure
        try:
            # Look for NOTEEVENTS table (MIMIC-III) or notes table (MIMIC-IV)
            notes_file = mimic_path / "NOTEEVENTS.csv"
            if not notes_file.exists():
                notes_file = mimic_path / "notes.csv"
            
            if notes_file.exists():
                df = pd.read_csv(notes_file)
                logger.info(f"Loaded {len(df)} notes from MIMIC")
                return df
            else:
                logger.warning("No notes file found in MIMIC directory")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading MIMIC data: {e}")
            return pd.DataFrame()
    
    def download_padchest_sample(self) -> pd.DataFrame:
        """
        Download sample data from PadChest (if available).
        
        Returns:
            DataFrame with chest X-ray reports
        """
        # PadChest requires registration
        logger.info("To access PadChest dataset:")
        logger.info("1. Go to https://padchest.um.es/")
        logger.info("2. Register for access")
        logger.info("3. Download the dataset")
        
        # Return sample structure
        sample_data = {
            'image_id': ['sample_1', 'sample_2', 'sample_3'],
            'report': [
                'Normal chest X-ray. No evidence of pneumonia or other abnormalities.',
                'Bilateral infiltrates consistent with pneumonia. No pneumothorax.',
                'Cardiomegaly with pulmonary congestion. No acute findings.'
            ],
            'findings': ['normal', 'pneumonia', 'cardiomegaly']
        }
        
        return pd.DataFrame(sample_data)
    
    def download_mednli(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Download MedNLI dataset for medical natural language inference.
        
        Returns:
            Tuple of (train, dev, test) DataFrames
        """
        mednli_url = "https://github.com/jgc128/mednli/raw/master/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0.zip"
        
        try:
            # Download MedNLI
            target_file = self.data_path / "mednli.zip"
            if not target_file.exists():
                logger.info("Downloading MedNLI dataset...")
                response = requests.get(mednli_url)
                with open(target_file, 'wb') as f:
                    f.write(response.content)
            
            # Extract
            extract_dir = self.data_path / "mednli"
            if not extract_dir.exists():
                with zipfile.ZipFile(target_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            
            # Load data
            train_file = extract_dir / "mli_train_v1.jsonl"
            dev_file = extract_dir / "mli_dev_v1.jsonl"
            test_file = extract_dir / "mli_test_v1.jsonl"
            
            train_data = []
            dev_data = []
            test_data = []
            
            for file_path, data_list in [(train_file, train_data), (dev_file, dev_data), (test_file, test_data)]:
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        for line in f:
                            data_list.append(json.loads(line))
            
            train_df = pd.DataFrame(train_data) if train_data else pd.DataFrame()
            dev_df = pd.DataFrame(dev_data) if dev_data else pd.DataFrame()
            test_df = pd.DataFrame(test_data) if test_data else pd.DataFrame()
            
            logger.info(f"Loaded MedNLI: {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test")
            
            return train_df, dev_df, test_df
            
        except Exception as e:
            logger.error(f"Error downloading MedNLI: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def create_clinical_summarization_dataset(self, source_data: pd.DataFrame, 
                                           task: str = "summarization") -> Dict[str, pd.DataFrame]:
        """
        Create a clinical summarization dataset from real data.
        
        Args:
            source_data: Source clinical data
            task: Task type ('summarization', 'classification', etc.)
            
        Returns:
            Dictionary with train/val/test splits
        """
        if source_data.empty:
            logger.warning("No source data provided, using synthetic data")
            return self._create_synthetic_dataset()
        
        # Clean and preprocess the data
        processed_data = self._preprocess_clinical_data(source_data)
        
        # Split the data
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        
        n_samples = len(processed_data)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_df = processed_data[:train_end]
        val_df = processed_data[train_end:val_end]
        test_df = processed_data[val_end:]
        
        logger.info(f"Created dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def _preprocess_clinical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess clinical data for summarization task.
        
        Args:
            df: Raw clinical data
            
        Returns:
            Preprocessed DataFrame
        """
        # This would be customized based on the actual dataset structure
        processed_df = df.copy()
        
        # Example preprocessing steps
        if 'text' in processed_df.columns:
            processed_df['clinical_note'] = processed_df['text']
        if 'summary' not in processed_df.columns:
            processed_df['summary'] = processed_df['clinical_note'].apply(
                lambda x: self._generate_simple_summary(x)
            )
        
        # Clean text
        processed_df['clinical_note_clean'] = processed_df['clinical_note'].apply(
            lambda x: self._clean_text(x) if isinstance(x, str) else ""
        )
        processed_df['summary_clean'] = processed_df['summary'].apply(
            lambda x: self._clean_text(x) if isinstance(x, str) else ""
        )
        
        # Add metadata
        processed_df['note_word_count'] = processed_df['clinical_note_clean'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        processed_df['summary_word_count'] = processed_df['summary_clean'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        
        return processed_df
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        import re
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', '', text)
        return text.strip()
    
    def _generate_simple_summary(self, text: str) -> str:
        """Generate a simple summary for demonstration."""
        if not isinstance(text, str) or len(text) < 50:
            return "No summary available."
        
        # Simple extractive summarization
        sentences = text.split('.')
        if len(sentences) > 1:
            return sentences[0] + "."
        else:
            return text[:100] + "..." if len(text) > 100 else text
    
    def _create_synthetic_dataset(self) -> Dict[str, pd.DataFrame]:
        """Create synthetic dataset as fallback."""
        from .clinical_data import create_sample_dataset
        return create_sample_dataset()

def get_available_datasets() -> Dict[str, Dict]:
    """
    Get information about available healthcare datasets.
    
    Returns:
        Dictionary with dataset information
    """
    datasets = {
        "MIMIC-III": {
            "description": "Medical Information Mart for Intensive Care III",
            "url": "https://mimic.mit.edu/",
            "access": "Requires approval",
            "content": "Clinical notes, discharge summaries, lab results",
            "size": "40,000+ patients, 2M+ notes",
            "best_for": "Clinical note summarization, medical text analysis"
        },
        "MIMIC-IV": {
            "description": "Updated version of MIMIC-III",
            "url": "https://mimic.mit.edu/",
            "access": "Requires approval",
            "content": "Clinical data, more recent than MIMIC-III",
            "size": "Larger than MIMIC-III",
            "best_for": "Clinical research, time series analysis"
        },
        "PadChest": {
            "description": "Chest X-ray images with reports",
            "url": "https://padchest.um.es/",
            "access": "Free with registration",
            "content": "Chest X-rays with radiology reports",
            "size": "160,000+ images",
            "best_for": "Radiology report generation, medical VQA"
        },
        "MedNLI": {
            "description": "Medical Natural Language Inference",
            "url": "https://github.com/jgc128/mednli",
            "access": "Free",
            "content": "Medical text pairs with inference labels",
            "size": "11,000+ sentence pairs",
            "best_for": "Medical text understanding, NLI tasks"
        },
        "PhysioNet": {
            "description": "Collection of physiological datasets",
            "url": "https://physionet.org/",
            "access": "Free with registration",
            "content": "Various clinical datasets",
            "size": "Multiple datasets",
            "best_for": "Time series analysis, clinical prediction"
        }
    }
    
    return datasets

def print_dataset_info():
    """Print information about available datasets."""
    datasets = get_available_datasets()
    
    print("🏥 AVAILABLE HEALTHCARE DATASETS")
    print("=" * 50)
    
    for name, info in datasets.items():
        print(f"\n📊 {name}")
        print(f"   Description: {info['description']}")
        print(f"   URL: {info['url']}")
        print(f"   Access: {info['access']}")
        print(f"   Content: {info['content']}")
        print(f"   Size: {info['size']}")
        print(f"   Best for: {info['best_for']}")

if __name__ == "__main__":
    print_dataset_info() 