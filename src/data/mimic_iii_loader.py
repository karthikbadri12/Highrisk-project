"""
MIMIC-III data loader for the high-risk AI healthcare project.
This module processes MIMIC-III data to create clinical summarization tasks.
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
from datetime import datetime, timedelta

from ..utils import config, logger

class MIMICIIILoader:
    """
    Loader for MIMIC-III data to create clinical summarization tasks.
    """
    
    def __init__(self, mimic_path: str = "mimiciii_data"):
        self.mimic_path = Path(mimic_path)
        self.data_cache = {}
        
    def load_compressed_csv(self, filename: str) -> pd.DataFrame:
        """
        Load a compressed CSV file from MIMIC-III.
        
        Args:
            filename: Name of the CSV file (with .csv.gz extension)
            
        Returns:
            DataFrame with the loaded data
        """
        file_path = self.mimic_path / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        try:
            logger.info(f"Loading {filename}...")
            df = pd.read_csv(file_path, compression='gzip', low_memory=False)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def load_patient_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load core patient data from MIMIC-III.
        
        Returns:
            Dictionary with loaded DataFrames
        """
        data = {}
        
        # Load core tables
        core_tables = [
            'PATIENTS.csv.gz',
            'ADMISSIONS.csv.gz',
            'ICUSTAYS.csv.gz',
            'DIAGNOSES_ICD.csv.gz',
            'PROCEDURES_ICD.csv.gz',
            'PRESCRIPTIONS.csv.gz'
        ]
        
        for table in core_tables:
            df = self.load_compressed_csv(table)
            if not df.empty:
                data[table.replace('.csv.gz', '')] = df
        
        return data
    
    def create_clinical_summaries_from_mimic(self, num_samples: int = 200) -> pd.DataFrame:
        """
        Create clinical summaries from MIMIC-III data.
        Since we don't have NOTEEVENTS, we'll create synthetic clinical notes
        based on real patient data.
        
        Args:
            num_samples: Number of samples to create
            
        Returns:
            DataFrame with clinical notes and summaries
        """
        logger.info("Creating clinical summaries from MIMIC-III data...")
        
        # Load patient data
        patient_data = self.load_patient_data()
        
        if not patient_data:
            logger.error("No patient data loaded")
            return pd.DataFrame()
        
        # Get patients and admissions
        patients_df = patient_data.get('PATIENTS', pd.DataFrame())
        admissions_df = patient_data.get('ADMISSIONS', pd.DataFrame())
        diagnoses_df = patient_data.get('DIAGNOSES_ICD', pd.DataFrame())
        procedures_df = patient_data.get('PROCEDURES_ICD', pd.DataFrame())
        prescriptions_df = patient_data.get('PRESCRIPTIONS', pd.DataFrame())
        
        if patients_df.empty or admissions_df.empty:
            logger.error("Missing core patient or admission data")
            return pd.DataFrame()
        
        # Calculate age from DOB and admission time
        logger.info("Calculating patient ages...")
        
        # Handle MIMIC-III date shifting (dates are shifted for privacy)
        # We'll use a simpler approach to avoid overflow issues
        try:
            patients_df['DOB'] = pd.to_datetime(patients_df['DOB'])
            admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
            
            # Merge patient and admission data
            patient_admissions = pd.merge(
                patients_df, 
                admissions_df, 
                on='SUBJECT_ID', 
                how='inner'
            )
            
            # Calculate age at admission (handle potential overflow)
            try:
                patient_admissions['AGE'] = (
                    patient_admissions['ADMITTIME'] - patient_admissions['DOB']
                ).dt.total_seconds() / (365.25 * 24 * 3600)
                
                # Filter out unrealistic ages
                patient_admissions = patient_admissions[
                    (patient_admissions['AGE'] >= 0) & 
                    (patient_admissions['AGE'] <= 120)
                ]
            except OverflowError:
                # If overflow occurs, use a simplified age calculation
                logger.warning("Overflow in age calculation, using simplified approach")
                # Use year difference as approximation
                patient_admissions['AGE'] = (
                    patient_admissions['ADMITTIME'].dt.year - 
                    patient_admissions['DOB'].dt.year
                )
                # Filter out unrealistic ages
                patient_admissions = patient_admissions[
                    (patient_admissions['AGE'] >= 0) & 
                    (patient_admissions['AGE'] <= 120)
                ]
                
        except Exception as e:
            logger.warning(f"Error in age calculation: {e}, using default age")
            # If all else fails, use a default age range
            patient_admissions = pd.merge(
                patients_df, 
                admissions_df, 
                on='SUBJECT_ID', 
                how='inner'
            )
            # Assign random ages between 18-90 for demonstration
            patient_admissions['AGE'] = np.random.randint(18, 91, size=len(patient_admissions))
        
        logger.info(f"Available patient-admission pairs: {len(patient_admissions)}")
        
        # Sample patients for our dataset
        sample_patients = patient_admissions.sample(
            n=min(num_samples, len(patient_admissions)), 
            random_state=42
        )
        
        clinical_notes = []
        
        for idx, patient in sample_patients.iterrows():
            # Get patient-specific data
            subject_id = patient['SUBJECT_ID']
            hadm_id = patient['HADM_ID']
            
            # Get diagnoses for this admission
            patient_diagnoses = diagnoses_df[
                (diagnoses_df['SUBJECT_ID'] == subject_id) & 
                (diagnoses_df['HADM_ID'] == hadm_id)
            ]
            
            # Get procedures for this admission
            patient_procedures = procedures_df[
                (procedures_df['SUBJECT_ID'] == subject_id) & 
                (procedures_df['HADM_ID'] == hadm_id)
            ]
            
            # Get prescriptions for this admission
            patient_prescriptions = prescriptions_df[
                (prescriptions_df['SUBJECT_ID'] == subject_id) & 
                (prescriptions_df['HADM_ID'] == hadm_id)
            ]
            
            # Create clinical note
            clinical_note = self._generate_clinical_note(
                patient, patient_diagnoses, patient_procedures, patient_prescriptions
            )
            
            # Create summary
            summary = self._generate_summary(
                patient, patient_diagnoses, patient_procedures
            )
            
            clinical_notes.append({
                'patient_id': subject_id,
                'admission_id': hadm_id,
                'age': int(patient['AGE']),
                'gender': patient['GENDER'],
                'admission_type': patient['ADMISSION_TYPE'],
                'diagnosis': patient['DIAGNOSIS'],
                'clinical_note': clinical_note,
                'summary': summary,
                'note_length': len(clinical_note),
                'summary_length': len(summary),
                'clinical_note_clean': self._clean_text(clinical_note),
                'summary_clean': self._clean_text(summary),
                'note_word_count': len(clinical_note.split()),
                'summary_word_count': len(summary.split())
            })
        
        df = pd.DataFrame(clinical_notes)
        logger.info(f"Created {len(df)} clinical notes from MIMIC-III data")
        
        return df
    
    def _generate_clinical_note(self, patient: pd.Series, diagnoses: pd.DataFrame, 
                              procedures: pd.DataFrame, prescriptions: pd.DataFrame) -> str:
        """
        Generate a synthetic clinical note based on real patient data.
        
        Args:
            patient: Patient information
            diagnoses: Patient diagnoses
            procedures: Patient procedures
            prescriptions: Patient prescriptions
            
        Returns:
            Generated clinical note
        """
        # Patient demographics
        age = patient['AGE']
        gender = patient['GENDER']
        admission_type = patient['ADMISSION_TYPE']
        diagnosis = patient['DIAGNOSIS']
        
        # Build clinical note
        note_parts = []
        
        # Chief complaint and history
        note_parts.append(f"CHIEF COMPLAINT: {diagnosis}")
        note_parts.append(f"HISTORY OF PRESENT ILLNESS: {age}-year-old {gender.lower()} patient admitted for {diagnosis.lower()}.")
        
        # Admission details
        note_parts.append(f"ADMISSION TYPE: {admission_type}")
        
        # Diagnoses
        if not diagnoses.empty:
            primary_diagnoses = diagnoses[diagnoses['SEQ_NUM'] <= 3]['ICD9_CODE'].astype(str).tolist()
            if primary_diagnoses:
                note_parts.append(f"PRIMARY DIAGNOSES: {', '.join(primary_diagnoses[:3])}")
        
        # Procedures
        if not procedures.empty:
            proc_codes = procedures['ICD9_CODE'].astype(str).unique()[:3]
            if len(proc_codes) > 0:
                note_parts.append(f"PROCEDURES PERFORMED: {', '.join(proc_codes)}")
        
        # Medications
        if not prescriptions.empty:
            meds = prescriptions['DRUG'].astype(str).unique()[:5]
            if len(meds) > 0:
                note_parts.append(f"CURRENT MEDICATIONS: {', '.join(meds)}")
        
        # Physical examination
        note_parts.append("PHYSICAL EXAMINATION: Patient appears alert and oriented. Vital signs stable.")
        
        # Assessment and plan
        note_parts.append(f"ASSESSMENT: {diagnosis}")
        note_parts.append("PLAN: Continue current treatment plan. Monitor patient progress.")
        
        return "\n".join(note_parts)
    
    def _generate_summary(self, patient: pd.Series, diagnoses: pd.DataFrame, 
                         procedures: pd.DataFrame) -> str:
        """
        Generate a summary based on patient data.
        
        Args:
            patient: Patient information
            diagnoses: Patient diagnoses
            procedures: Patient procedures
            
        Returns:
            Generated summary
        """
        age = patient['AGE']
        gender = patient['GENDER']
        diagnosis = patient['DIAGNOSIS']
        
        summary_parts = []
        summary_parts.append(f"{age}-year-old {gender.lower()} with {diagnosis.lower()}")
        
        # Add primary diagnosis
        if not diagnoses.empty:
            primary_dx = diagnoses[diagnoses['SEQ_NUM'] == 1]
            if not primary_dx.empty:
                summary_parts.append(f"Primary diagnosis: {str(primary_dx.iloc[0]['ICD9_CODE'])}")
        
        # Add procedures if any
        if not procedures.empty:
            proc_count = len(procedures)
            summary_parts.append(f"Underwent {proc_count} procedure(s)")
        
        return ". ".join(summary_parts) + "."
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', '', text)
        return text.strip()
    
    def create_train_val_test_splits(self, df: pd.DataFrame, 
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Create train/validation/test splits from the dataset.
        
        Args:
            df: Input DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        if df.empty:
            logger.error("Empty DataFrame provided")
            return {}
        
        # Shuffle the data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split indices
        n_samples = len(df_shuffled)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split the data
        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]
        
        logger.info(f"Created splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def save_splits_to_csv(self, splits: Dict[str, pd.DataFrame], 
                          output_dir: str = "data/clinical_notes") -> None:
        """
        Save train/val/test splits to CSV files.
        
        Args:
            splits: Dictionary with train/val/test DataFrames
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, df in splits.items():
            output_file = output_path / f"{split_name}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {split_name} split to {output_file}")

def create_mimic_iii_dataset(num_samples: int = 200) -> Dict[str, pd.DataFrame]:
    """
    Create a complete MIMIC-III based dataset for clinical summarization.
    
    Args:
        num_samples: Number of samples to create
        
    Returns:
        Dictionary with train/val/test splits
    """
    loader = MIMICIIILoader()
    
    # Create clinical notes from MIMIC-III data
    clinical_notes_df = loader.create_clinical_summaries_from_mimic(num_samples)
    
    if clinical_notes_df.empty:
        logger.error("Failed to create clinical notes from MIMIC-III")
        return {}
    
    # Create splits
    splits = loader.create_train_val_test_splits(clinical_notes_df)
    
    # Save to CSV
    loader.save_splits_to_csv(splits)
    
    return splits

if __name__ == "__main__":
    # Test the MIMIC-III loader
    splits = create_mimic_iii_dataset(100)
    print(f"Created dataset with {len(splits.get('train', pd.DataFrame()))} training samples") 