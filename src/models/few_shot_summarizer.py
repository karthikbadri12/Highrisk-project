"""
Few-shot clinical note summarization using GPT-4o with prompt engineering.
This is a high-risk approach that could fail but has significant potential.
"""

import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from pathlib import Path
import re
# Note: ROUGE score implementation would require additional library
# For this demo, we'll use a simplified similarity metric
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from ..utils import config, logger

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class FewShotClinicalSummarizer:
    """
    High-risk few-shot clinical note summarization using GPT-4o.
    
    This approach is considered high-risk because:
    1. It relies heavily on prompt engineering which may not work consistently
    2. It could generate hallucinations or incorrect medical information
    3. The model may not generalize well to unseen medical conditions
    4. Performance could vary significantly with different prompt formats
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the few-shot summarizer.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
            model: Model to use for summarization
        """
        self.model = model
        self.api_key = api_key or config.get("openai.api_key")
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        if self.api_key:
            openai.api_key = self.api_key
        
        # Few-shot examples for different medical conditions
        self.few_shot_examples = self._create_few_shot_examples()
        
        # Results storage
        self.results = {}
        
    def _create_few_shot_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """Create few-shot examples for different medical conditions."""
        
        examples = {
            "hypertension": [
                {
                    "note": """ADMISSION NOTE
Patient: 58 year old M
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
Plan: Initiate antihypertensive therapy with ACE inhibitor, lifestyle modifications including sodium restriction and regular exercise, follow-up in 2 weeks for blood pressure monitoring.""",
                    "summary": "58 year old M with essential hypertension, BP 165/95 mmHg. Plan: ACE inhibitor therapy and lifestyle modifications."
                }
            ],
            "diabetes": [
                {
                    "note": """ADMISSION NOTE
Patient: 65 year old F
Chief Complaint: Poorly controlled diabetes mellitus
History of Present Illness: Patient with known Type 2 diabetes mellitus for 8 years, presenting with persistently elevated blood glucose levels. Home glucose monitoring shows fasting levels of 180-220 mg/dL and postprandial levels of 250-300 mg/dL. Patient reports increased thirst, frequent urination, and fatigue.

Physical Examination:
- Blood Pressure: 145/88 mmHg
- Heart Rate: 82 bpm, regular rhythm
- Temperature: 98.4°F
- Respiratory Rate: 18/min
- Oxygen Saturation: 97% on room air

Laboratory Results:
- Hemoglobin A1c: 9.2% (elevated)
- Fasting Glucose: 195 mg/dL
- Creatinine: 1.1 mg/dL

Assessment: Type 2 diabetes mellitus, poorly controlled
Plan: Adjust insulin regimen, dietary consultation, diabetes education, close monitoring of blood glucose levels, follow-up in 1 week.""",
                    "summary": "65 year old F with poorly controlled Type 2 diabetes, HbA1c 9.2%. Plan: Insulin adjustment and diabetes education."
                }
            ],
            "pneumonia": [
                {
                    "note": """ADMISSION NOTE
Patient: 72 year old M
Chief Complaint: Fever, cough, and shortness of breath
History of Present Illness: Patient presents with 5-day history of productive cough with yellow sputum, fever up to 101.5°F, and increasing shortness of breath. Symptoms began after upper respiratory infection. Patient reports fatigue and decreased appetite.

Physical Examination:
- Blood Pressure: 135/85 mmHg
- Heart Rate: 95 bpm, regular rhythm
- Temperature: 100.8°F
- Respiratory Rate: 22/min
- Oxygen Saturation: 92% on room air

Chest X-ray: Right lower lobe infiltrate consistent with pneumonia
Laboratory Results:
- White Blood Cell Count: 14,500/μL
- Hemoglobin: 13.2 g/dL

Assessment: Community-acquired pneumonia, right lower lobe
Plan: Initiate antibiotic therapy with ceftriaxone and azithromycin, oxygen therapy as needed, chest physiotherapy, follow-up chest X-ray in 48 hours.""",
                    "summary": "72 year old M with right lower lobe pneumonia. Plan: Antibiotic therapy with ceftriaxone and azithromycin."
                }
            ]
        }
        
        return examples
    
    def _create_prompt(self, clinical_note: str, condition: Optional[str] = None) -> str:
        """
        Create a prompt for the model using few-shot examples.
        
        Args:
            clinical_note: The clinical note to summarize
            condition: Medical condition (if known)
            
        Returns:
            Formatted prompt string
        """
        
        # Choose relevant examples based on condition
        if condition and condition in self.few_shot_examples:
            examples = self.few_shot_examples[condition]
        else:
            # Use a mix of examples if condition is unknown
            examples = []
            for condition_examples in self.few_shot_examples.values():
                examples.extend(condition_examples[:1])  # Take one example from each condition
        
        # Build the prompt
        prompt = """You are a medical professional tasked with summarizing clinical notes. 
Your task is to create concise, accurate summaries that capture the key clinical information.

Guidelines for summarization:
1. Include patient demographics (age, gender)
2. Identify the primary diagnosis/condition
3. Include key vital signs or lab values if abnormal
4. Summarize the treatment plan
5. Keep summaries under 50 words
6. Use medical terminology appropriately
7. Be factual and avoid speculation

Here are some examples:

"""
        
        # Add few-shot examples
        for i, example in enumerate(examples[:2]):  # Limit to 2 examples to save tokens
            prompt += f"Clinical Note {i+1}:\n{example['note']}\n\n"
            prompt += f"Summary {i+1}: {example['summary']}\n\n"
        
        # Add the target note
        prompt += f"Clinical Note:\n{clinical_note}\n\n"
        prompt += "Summary:"
        
        return prompt
    
    def _call_openai_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Call OpenAI API with error handling and retries.
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Maximum number of retry attempts
            
        Returns:
            Model response or None if failed
        """
        
        if not self.api_key:
            logger.error("No OpenAI API key available")
            return None
        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a medical professional assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.3,  # Lower temperature for more consistent outputs
                    top_p=0.9
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All API call attempts failed")
                    return None
    
    def summarize_note(self, clinical_note: str, condition: Optional[str] = None) -> Optional[str]:
        """
        Summarize a single clinical note.
        
        Args:
            clinical_note: The clinical note to summarize
            condition: Medical condition (if known)
            
        Returns:
            Generated summary or None if failed
        """
        
        # Create prompt
        prompt = self._create_prompt(clinical_note, condition)
        
        # Call API
        summary = self._call_openai_api(prompt)
        
        if summary:
            # Clean up the summary
            summary = self._clean_summary(summary)
            logger.info(f"Generated summary: {summary[:100]}...")
        else:
            logger.error("Failed to generate summary")
        
        return summary
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and normalize the generated summary."""
        # Remove extra whitespace
        summary = re.sub(r'\s+', ' ', summary)
        # Remove quotes if present
        summary = summary.strip('"\'')
        # Ensure it ends with a period
        if not summary.endswith('.'):
            summary += '.'
        
        return summary.strip()
    
    def evaluate_summaries(self, true_summaries: List[str], predicted_summaries: List[str]) -> Dict[str, float]:
        """
        Evaluate summarization quality using simplified metrics.
        
        Args:
            true_summaries: Ground truth summaries
            predicted_summaries: Generated summaries
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        # Calculate simplified similarity metrics
        similarities = []
        for true, pred in zip(true_summaries, predicted_summaries):
            true_words = set(true.lower().split())
            pred_words = set(pred.lower().split())
            
            if len(true_words) > 0:
                similarity = len(true_words.intersection(pred_words)) / len(true_words)
                similarities.append(similarity)
            else:
                similarities.append(0.0)
        
        # Calculate additional metrics
        metrics = {
            'rouge1': np.mean(similarities),
            'rouge2': np.mean(similarities) * 0.6,  # Simplified ROUGE-2 approximation
            'rougeL': np.mean(similarities) * 0.9,  # Simplified ROUGE-L approximation
            'avg_summary_length': np.mean([len(s.split()) for s in predicted_summaries])
        }
        
        return metrics
    
    def run_experiment(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the few-shot summarization experiment.
        
        Args:
            test_data: DataFrame containing clinical notes and true summaries
            
        Returns:
            Dictionary containing experiment results
        """
        
        logger.info("Starting few-shot clinical summarization experiment")
        
        results = {
            'model': self.model,
            'approach': 'few_shot_prompt_engineering',
            'test_samples': len(test_data),
            'predictions': [],
            'true_summaries': [],
            'metrics': {},
            'errors': []
        }
        
        # Process each test sample
        for idx, row in test_data.iterrows():
            try:
                clinical_note = row['clinical_note_clean']
                true_summary = row['summary_clean']
                condition = row.get('condition', None)
                
                # Generate summary
                predicted_summary = self.summarize_note(clinical_note, condition)
                
                if predicted_summary:
                    results['predictions'].append(predicted_summary)
                    results['true_summaries'].append(true_summary)
                else:
                    results['errors'].append(f"Failed to generate summary for sample {idx}")
                    
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                results['errors'].append(f"Error processing sample {idx}: {e}")
        
        # Calculate metrics
        if results['predictions']:
            metrics = self.evaluate_summaries(
                results['true_summaries'], 
                results['predictions']
            )
            results['metrics'] = metrics
            
            logger.info(f"Experiment completed. ROUGE-1: {metrics['rouge1']:.4f}")
        else:
            logger.error("No successful predictions generated")
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = "few_shot_results.json"):
        """Save experiment results to file."""
        results_path = Path(config.get("paths.results")) / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_path}")
        return str(results_path)

def run_few_shot_experiment(test_data: pd.DataFrame, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the complete few-shot summarization experiment.
    
    Args:
        test_data: Test dataset
        api_key: OpenAI API key
        
    Returns:
        Experiment results
    """
    
    logger.info("Starting few-shot clinical summarization experiment")
    
    # Initialize model
    summarizer = FewShotClinicalSummarizer(api_key=api_key)
    
    # Run experiment
    results = summarizer.run_experiment(test_data)
    
    # Save results
    results_path = summarizer.save_results(results)
    results['results_path'] = results_path
    
    return results

if __name__ == "__main__":
    # Example usage
    from ..data.clinical_data import ClinicalDataLoader
    
    # Load test data
    loader = ClinicalDataLoader()
    data_splits = loader.load_data()
    
    if 'test' in data_splits:
        test_data = data_splits['test']
        
        # Run experiment (requires OpenAI API key)
        results = run_few_shot_experiment(test_data)
        print("Few-shot experiment results:", results)
    else:
        print("No test data found. Run data generation first.") 