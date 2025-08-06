#!/usr/bin/env python3
"""
Script to integrate MIMIC-III data into the high-risk AI healthcare project.
This demonstrates how to use real clinical data for the summarization task.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.mimic_iii_loader import create_mimic_iii_dataset, MIMICIIILoader
from src.models.few_shot_summarizer import FewShotClinicalSummarizer
from src.evaluation.hallucination_detector import HallucinationDetector
from src.utils import config, logger

def analyze_mimic_data():
    """Analyze the available MIMIC-III data."""
    
    print("🔍 ANALYZING MIMIC-III DATA")
    print("=" * 50)
    
    loader = MIMICIIILoader()
    
    # Load and analyze patient data
    patient_data = loader.load_patient_data()
    
    print(f"\n📊 Available Tables: {len(patient_data)}")
    for table_name, df in patient_data.items():
        print(f"   {table_name}: {len(df)} rows")
    
    # Analyze patients and admissions
    if 'PATIENTS' in patient_data and 'ADMISSIONS' in patient_data:
        patients_df = patient_data['PATIENTS']
        admissions_df = patient_data['ADMISSIONS']
        
        print(f"\n👥 Patient Demographics:")
        print(f"   Total patients: {len(patients_df)}")
        print(f"   Total admissions: {len(admissions_df)}")
        print(f"   Gender distribution:")
        print(patients_df['GENDER'].value_counts())
        
        # Calculate age if possible
        try:
            patients_df['DOB'] = pd.to_datetime(patients_df['DOB'])
            admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
            
            # Merge and calculate age for a sample
            sample_merge = pd.merge(
                patients_df.head(1000), 
                admissions_df.head(1000), 
                on='SUBJECT_ID', 
                how='inner'
            )
            
            if not sample_merge.empty:
                sample_merge['AGE'] = (
                    sample_merge['ADMITTIME'] - sample_merge['DOB']
                ).dt.total_seconds() / (365.25 * 24 * 3600)
                
                valid_ages = sample_merge[
                    (sample_merge['AGE'] >= 0) & 
                    (sample_merge['AGE'] <= 120)
                ]['AGE']
                
                if len(valid_ages) > 0:
                    print(f"   Sample average age: {valid_ages.mean():.1f} years")
        except Exception as e:
            print(f"   Age calculation not available: {e}")
        
        print(f"\n🏥 Admission Types:")
        print(admissions_df['ADMISSION_TYPE'].value_counts())
        
        print(f"\nTop Diagnoses:")
        if 'DIAGNOSES_ICD' in patient_data:
            diagnoses_df = patient_data['DIAGNOSES_ICD']
            top_diagnoses = diagnoses_df['ICD9_CODE'].value_counts().head(10)
            for code, count in top_diagnoses.items():
                print(f"   {code}: {count} occurrences")

def create_mimic_dataset():
    """Create a dataset from MIMIC-III data."""
    
    print("\n🔄 CREATING MIMIC-III DATASET")
    print("=" * 50)
    
    # Create dataset with 200 samples
    splits = create_mimic_iii_dataset(num_samples=200)
    
    if not splits:
        print("Failed to create MIMIC-III dataset")
        return None
    
    print(f"Created dataset successfully!")
    print(f"   Training samples: {len(splits['train'])}")
    print(f"   Validation samples: {len(splits['val'])}")
    print(f"   Test samples: {len(splits['test'])}")
    
    # Show sample data
    print(f"\nSample Clinical Note:")
    sample_note = splits['train'].iloc[0]
    print(f"   Patient ID: {sample_note['patient_id']}")
    print(f"   Age: {sample_note['age']}, Gender: {sample_note['gender']}")
    print(f"   Diagnosis: {sample_note['diagnosis']}")
    print(f"   Note length: {sample_note['note_length']} characters")
    print(f"   Summary length: {sample_note['summary_length']} characters")
    
    return splits

def run_mimic_experiment(splits):
    """Run the few-shot summarization experiment with MIMIC-III data."""
    
    print("\nRUNNING MIMIC-III EXPERIMENT")
    print("=" * 50)
    
    # Initialize models
    summarizer = FewShotClinicalSummarizer()
    hallucination_detector = HallucinationDetector()
    
    # Get test data
    test_data = splits['test']
    
    print(f"Testing on {len(test_data)} samples...")
    
    results = {
        'summaries': [],
        'metrics': {},
        'hallucination_scores': []
    }
    
    # Process first 10 samples for demonstration
    sample_size = min(10, len(test_data))
    
    for i in range(sample_size):
        sample = test_data.iloc[i]
        clinical_note = sample['clinical_note']
        true_summary = sample['summary']
        
        print(f"\nSample {i+1}/{sample_size}")
        print(f"   Note: {clinical_note[:100]}...")
        print(f"   True Summary: {true_summary}")
        
        # Generate summary
        try:
            predicted_summary = summarizer.summarize_note(clinical_note)
            print(f"   Predicted Summary: {predicted_summary}")
            
            # Evaluate hallucination
            hallucination_score = hallucination_detector.detect_hallucinations(
                clinical_note, predicted_summary
            )
            
            results['summaries'].append({
                'note': clinical_note,
                'true_summary': true_summary,
                'predicted_summary': predicted_summary,
                'hallucination_score': hallucination_score
            })
            
            results['hallucination_scores'].append(hallucination_score)
            
        except Exception as e:
            print(f"   Error processing sample: {e}")
    
    # Calculate metrics
    if results['summaries']:
        avg_hallucination = np.mean(results['hallucination_scores'])
        results['metrics'] = {
            'avg_hallucination_score': avg_hallucination,
            'samples_processed': len(results['summaries'])
        }
        
        print(f"\nResults:")
        print(f"   Average hallucination score: {avg_hallucination:.3f}")
        print(f"   Samples processed: {len(results['summaries'])}")
    
    return results

def compare_synthetic_vs_mimic():
    """Compare synthetic data vs MIMIC-III data."""
    
    print("\nCOMPARING SYNTHETIC VS MIMIC-III DATA")
    print("=" * 50)
    
    # Load synthetic data
    synthetic_train = pd.read_csv("data/clinical_notes/train.csv")
    synthetic_val = pd.read_csv("data/clinical_notes/val.csv")
    synthetic_test = pd.read_csv("data/clinical_notes/test.csv")
    
    # Load MIMIC-III data
    mimic_train = pd.read_csv("data/clinical_notes/train.csv")
    mimic_val = pd.read_csv("data/clinical_notes/val.csv")
    mimic_test = pd.read_csv("data/clinical_notes/test.csv")
    
    print("📊 Dataset Comparison:")
    print(f"   Synthetic - Train: {len(synthetic_train)}, Val: {len(synthetic_val)}, Test: {len(synthetic_test)}")
    print(f"   MIMIC-III - Train: {len(mimic_train)}, Val: {len(mimic_val)}, Test: {len(mimic_test)}")
    
    print("\n📝 Content Comparison:")
    
    # Compare note lengths
    synthetic_note_lengths = synthetic_train['note_length']
    mimic_note_lengths = mimic_train['note_length']
    
    print(f"   Synthetic note length - Mean: {synthetic_note_lengths.mean():.1f}, Std: {synthetic_note_lengths.std():.1f}")
    print(f"   MIMIC-III note length - Mean: {mimic_note_lengths.mean():.1f}, Std: {mimic_note_lengths.std():.1f}")
    
    # Compare summary lengths
    synthetic_summary_lengths = synthetic_train['summary_length']
    mimic_summary_lengths = mimic_train['summary_length']
    
    print(f"   Synthetic summary length - Mean: {synthetic_summary_lengths.mean():.1f}, Std: {synthetic_summary_lengths.std():.1f}")
    print(f"   MIMIC-III summary length - Mean: {mimic_summary_lengths.mean():.1f}, Std: {mimic_summary_lengths.std():.1f}")

def save_mimic_results(results, output_file="results/mimic_iii_results.json"):
    """Save MIMIC-III experiment results."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert results
    json_results = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Saved results to {output_path}")

def main():
    """Main function to run MIMIC-III integration."""
    
    logger.info("Starting MIMIC-III integration")
    
    try:
        # Analyze available MIMIC-III data
        analyze_mimic_data()
        
        # Create dataset from MIMIC-III
        splits = create_mimic_dataset()
        
        if splits is not None:
            # Run experiment with MIMIC-III data
            results = run_mimic_experiment(splits)
            
            # Save results
            save_mimic_results(results)
            
            # Compare with synthetic data
            compare_synthetic_vs_mimic()
        
        print("\n" + "=" * 50)
        print("MIMIC-III INTEGRATION COMPLETED!")
        print("=" * 50)
        print("\nNext Steps:")
        print("1. Review the generated clinical notes in data/clinical_notes/")
        print("2. Run the full experiment with: python run_experiment.py")
        print("3. Compare results between synthetic and MIMIC-III data")
        print("4. Update your ACM report with real data findings")
        
    except Exception as e:
        logger.error(f"Error in MIMIC-III integration: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    import numpy as np
    main() 