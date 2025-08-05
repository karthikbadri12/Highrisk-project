"""
Clinical data loading and preprocessing for the few-shot summarization project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import re

from ..utils import config, logger

class ClinicalDataLoader:
    """Load and preprocess clinical note data for summarization."""
    
    def __init__(self):
        self.data_path = Path(config.get("paths.data"))
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def generate_sample_clinical_notes(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate sample clinical notes for demonstration.
        In a real project, you would load actual clinical data here.
        """
        logger.info(f"Generating {n_samples} sample clinical notes")
        
        # Sample patient demographics
        ages = np.random.normal(65, 15, n_samples).astype(int)
        genders = np.random.choice(['M', 'F'], n_samples)
        
        # Sample medical conditions and their associated notes
        conditions = [
            "hypertension", "diabetes", "heart_failure", "pneumonia", 
            "sepsis", "stroke", "kidney_disease", "cancer"
        ]
        
        # Generate clinical notes
        notes = []
        summaries = []
        
        for i in range(n_samples):
            condition = np.random.choice(conditions)
            age = ages[i]
            gender = genders[i]
            
            # Generate detailed clinical note
            note = self._generate_clinical_note(condition, age, gender)
            summary = self._generate_summary(condition, age, gender)
            
            notes.append(note)
            summaries.append(summary)
        
        # Create dataframe
        df = pd.DataFrame({
            'patient_id': range(1, n_samples + 1),
            'age': ages,
            'gender': genders,
            'condition': np.random.choice(conditions, n_samples),
            'clinical_note': notes,
            'summary': summaries,
            'note_length': [len(note) for note in notes],
            'summary_length': [len(summary) for summary in summaries]
        })
        
        return df
    
    def _generate_clinical_note(self, condition: str, age: int, gender: str) -> str:
        """Generate a realistic clinical note based on condition."""
        
        note_templates = {
            "hypertension": f"""
ADMISSION NOTE
Patient: {age} year old {gender}
Chief Complaint: Elevated blood pressure readings
History of Present Illness: Patient presents with consistently elevated blood pressure readings over the past 3 months. Home monitoring shows systolic pressures ranging from 140-180 mmHg and diastolic pressures of 90-110 mmHg. Patient reports occasional headaches and fatigue but no chest pain or shortness of breath.

Physical Examination:
- Blood Pressure: 165/95 mmHg (elevated)
- Heart Rate: 78 bpm, regular rhythm
- Temperature: 98.6°F
- Respiratory Rate: 16/min
- Oxygen Saturation: 98% on room air

Cardiovascular: Regular rate and rhythm, no murmurs, gallops, or rubs
Respiratory: Clear to auscultation bilaterally
Abdomen: Soft, non-tender, non-distended
Extremities: No edema, pulses 2+ throughout

Assessment: Essential hypertension, uncontrolled
Plan: Initiate antihypertensive therapy with ACE inhibitor, lifestyle modifications including sodium restriction and regular exercise, follow-up in 2 weeks for blood pressure monitoring.
""",
            
            "diabetes": f"""
ADMISSION NOTE
Patient: {age} year old {gender}
Chief Complaint: Poorly controlled diabetes mellitus
History of Present Illness: Patient with known Type 2 diabetes mellitus for 8 years, presenting with persistently elevated blood glucose levels. Home glucose monitoring shows fasting levels of 180-220 mg/dL and postprandial levels of 250-300 mg/dL. Patient reports increased thirst, frequent urination, and fatigue.

Physical Examination:
- Blood Pressure: 145/88 mmHg
- Heart Rate: 82 bpm, regular rhythm
- Temperature: 98.4°F
- Respiratory Rate: 18/min
- Oxygen Saturation: 97% on room air

Cardiovascular: Regular rate and rhythm, no murmurs
Respiratory: Clear to auscultation bilaterally
Abdomen: Soft, non-tender
Extremities: No edema, pulses 2+ throughout

Laboratory Results:
- Hemoglobin A1c: 9.2% (elevated)
- Fasting Glucose: 195 mg/dL
- Creatinine: 1.1 mg/dL
- Estimated GFR: 78 mL/min/1.73m²

Assessment: Type 2 diabetes mellitus, poorly controlled
Plan: Adjust insulin regimen, dietary consultation, diabetes education, close monitoring of blood glucose levels, follow-up in 1 week.
""",
            
            "pneumonia": f"""
ADMISSION NOTE
Patient: {age} year old {gender}
Chief Complaint: Fever, cough, and shortness of breath
History of Present Illness: Patient presents with 5-day history of productive cough with yellow sputum, fever up to 101.5°F, and increasing shortness of breath. Symptoms began after upper respiratory infection. Patient reports fatigue and decreased appetite.

Physical Examination:
- Blood Pressure: 135/85 mmHg
- Heart Rate: 95 bpm, regular rhythm
- Temperature: 100.8°F
- Respiratory Rate: 22/min
- Oxygen Saturation: 92% on room air

Cardiovascular: Regular rate and rhythm, no murmurs
Respiratory: Decreased breath sounds in right lower lobe, coarse rhonchi bilaterally
Abdomen: Soft, non-tender, non-distended
Extremities: No edema, pulses 2+ throughout

Chest X-ray: Right lower lobe infiltrate consistent with pneumonia
Laboratory Results:
- White Blood Cell Count: 14,500/μL
- Hemoglobin: 13.2 g/dL
- Platelets: 250,000/μL

Assessment: Community-acquired pneumonia, right lower lobe
Plan: Initiate antibiotic therapy with ceftriaxone and azithromycin, oxygen therapy as needed, chest physiotherapy, follow-up chest X-ray in 48 hours.
"""
        }
        
        if condition in note_templates:
            return note_templates[condition].strip()
        else:
            # Generic template for other conditions
            return f"""
ADMISSION NOTE
Patient: {age} year old {gender}
Chief Complaint: [Condition-related symptoms]
History of Present Illness: Patient presents with symptoms related to {condition}.

Physical Examination:
- Blood Pressure: [BP reading]
- Heart Rate: [HR] bpm
- Temperature: [Temp]°F
- Respiratory Rate: [RR]/min
- Oxygen Saturation: [O2 sat]%

Assessment: {condition.replace('_', ' ').title()}
Plan: [Treatment plan]
""".strip()
    
    def _generate_summary(self, condition: str, age: int, gender: str) -> str:
        """Generate a concise summary of the clinical note."""
        
        summary_templates = {
            "hypertension": f"{age} year old {gender} with essential hypertension, BP 165/95 mmHg. Plan: ACE inhibitor therapy and lifestyle modifications.",
            "diabetes": f"{age} year old {gender} with poorly controlled Type 2 diabetes, HbA1c 9.2%. Plan: Insulin adjustment and diabetes education.",
            "pneumonia": f"{age} year old {gender} with right lower lobe pneumonia. Plan: Antibiotic therapy with ceftriaxone and azithromycin."
        }
        
        if condition in summary_templates:
            return summary_templates[condition]
        else:
            return f"{age} year old {gender} with {condition.replace('_', ' ')}. Plan: [Treatment plan]."
    
    def preprocess_notes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess clinical notes for model input."""
        logger.info("Preprocessing clinical notes")
        
        # Clean text
        df['clinical_note_clean'] = df['clinical_note'].apply(self._clean_text)
        df['summary_clean'] = df['summary'].apply(self._clean_text)
        
        # Add metadata
        df['note_word_count'] = df['clinical_note_clean'].apply(lambda x: len(x.split()))
        df['summary_word_count'] = df['summary_clean'].apply(lambda x: len(x.split()))
        
        # Filter out very short or very long notes
        df = df[
            (df['note_word_count'] >= 50) & 
            (df['note_word_count'] <= 1000) &
            (df['summary_word_count'] >= 10) &
            (df['summary_word_count'] <= 100)
        ]
        
        logger.info(f"Preprocessed {len(df)} notes")
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', '', text)
        return text.strip()
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7, 
                   val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets."""
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split indices
        n_samples = len(df_shuffled)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split data
        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def save_data(self, data_dict: Dict[str, pd.DataFrame], filename: str = "clinical_notes"):
        """Save processed data to files."""
        data_path = self.data_path / filename
        data_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, df in data_dict.items():
            file_path = data_path / f"{split_name}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {split_name} data to {file_path}")
    
    def load_data(self, filename: str = "clinical_notes") -> Dict[str, pd.DataFrame]:
        """Load processed data from files."""
        data_path = self.data_path / filename
        
        data_dict = {}
        for split in ['train', 'val', 'test']:
            file_path = data_path / f"{split}.csv"
            if file_path.exists():
                data_dict[split] = pd.read_csv(file_path)
                logger.info(f"Loaded {split} data: {len(data_dict[split])} samples")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return data_dict

def create_sample_dataset():
    """Create and save a sample clinical notes dataset."""
    loader = ClinicalDataLoader()
    
    # Generate sample data
    df = loader.generate_sample_clinical_notes(n_samples=200)
    
    # Preprocess
    df = loader.preprocess_notes(df)
    
    # Split data
    data_splits = loader.split_data(df)
    
    # Save data
    loader.save_data(data_splits)
    
    return data_splits

if __name__ == "__main__":
    # Create sample dataset
    data_splits = create_sample_dataset()
    print(f"Created dataset with {len(data_splits['train'])} train, {len(data_splits['val'])} val, {len(data_splits['test'])} test samples") 