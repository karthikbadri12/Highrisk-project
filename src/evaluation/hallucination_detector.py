"""
Hallucination detection for clinical note summarization.
This is a critical safety component for the high-risk few-shot approach.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter

from ..utils import config, logger

class HallucinationDetector:
    """
    Detect potential hallucinations in clinical note summaries.
    
    This is a high-risk component because:
    1. Hallucination detection is inherently difficult
    2. False positives could reject valid summaries
    3. False negatives could miss dangerous hallucinations
    4. The detection methods themselves may not be reliable
    """
    
    def __init__(self):
        """Initialize the hallucination detector."""
        self.medical_terms = self._load_medical_terms()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Load spaCy model for medical text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _load_medical_terms(self) -> set:
        """Load a set of common medical terms for verification."""
        medical_terms = {
            # Vital signs
            'blood pressure', 'heart rate', 'temperature', 'respiratory rate', 'oxygen saturation',
            'systolic', 'diastolic', 'bpm', 'f', 'c', 'mmhg', 'spo2',
            
            # Common medical conditions
            'hypertension', 'diabetes', 'pneumonia', 'sepsis', 'stroke', 'heart failure',
            'kidney disease', 'cancer', 'infection', 'fever', 'cough', 'shortness of breath',
            
            # Medications
            'ace inhibitor', 'insulin', 'antibiotic', 'ceftriaxone', 'azithromycin',
            'antihypertensive', 'diuretic', 'beta blocker',
            
            # Medical procedures
            'chest x-ray', 'laboratory', 'physical examination', 'assessment', 'plan',
            'follow-up', 'monitoring', 'therapy', 'treatment',
            
            # Units and measurements
            'mg/dl', 'mmol/l', 'mg', 'ml', 'l', 'kg', 'cm', 'inches', 'years', 'months',
            'elevated', 'normal', 'abnormal', 'high', 'low', 'positive', 'negative'
        }
        
        return medical_terms
    
    def detect_hallucinations(self, clinical_note: str, summary: str) -> Dict[str, Any]:
        """
        Detect potential hallucinations in a summary compared to the source note.
        
        Args:
            clinical_note: Original clinical note
            summary: Generated summary
            
        Returns:
            Dictionary containing hallucination detection results
        """
        
        results = {
            'summary': summary,
            'note_length': len(clinical_note.split()),
            'summary_length': len(summary.split()),
            'hallucination_score': 0.0,
            'warnings': [],
            'suggestions': []
        }
        
        # Check 1: Factual consistency
        factual_check = self._check_factual_consistency(clinical_note, summary)
        results.update(factual_check)
        
        # Check 2: Medical term verification
        term_check = self._verify_medical_terms(summary)
        results.update(term_check)
        
        # Check 3: Semantic similarity
        similarity_check = self._check_semantic_similarity(clinical_note, summary)
        results.update(similarity_check)
        
        # Check 4: Logical consistency
        logic_check = self._check_logical_consistency(clinical_note, summary)
        results.update(logic_check)
        
        # Calculate overall hallucination score
        results['hallucination_score'] = self._calculate_hallucination_score(results)
        
        return results
    
    def _check_factual_consistency(self, note: str, summary: str) -> Dict[str, Any]:
        """Check if facts in summary are consistent with the source note."""
        
        # Extract key facts from note
        note_facts = self._extract_facts(note)
        summary_facts = self._extract_facts(summary)
        
        # Check for contradictions
        contradictions = []
        missing_facts = []
        
        for fact_type, note_value in note_facts.items():
            if fact_type in summary_facts:
                summary_value = summary_facts[fact_type]
                if note_value != summary_value:
                    contradictions.append({
                        'fact_type': fact_type,
                        'note_value': note_value,
                        'summary_value': summary_value
                    })
            else:
                missing_facts.append(fact_type)
        
        return {
            'factual_contradictions': contradictions,
            'missing_facts': missing_facts,
            'factual_consistency_score': 1.0 - (len(contradictions) / max(len(note_facts), 1))
        }
    
    def _extract_facts(self, text: str) -> Dict[str, str]:
        """Extract key facts from clinical text."""
        facts = {}
        
        # Extract age and gender
        age_match = re.search(r'(\d+)\s*year\s*old\s*([MF])', text, re.IGNORECASE)
        if age_match:
            facts['age'] = age_match.group(1)
            facts['gender'] = age_match.group(2)
        
        # Extract blood pressure
        bp_match = re.search(r'(\d+)/(\d+)\s*mmhg', text, re.IGNORECASE)
        if bp_match:
            facts['blood_pressure'] = f"{bp_match.group(1)}/{bp_match.group(2)}"
        
        # Extract heart rate
        hr_match = re.search(r'(\d+)\s*bpm', text, re.IGNORECASE)
        if hr_match:
            facts['heart_rate'] = hr_match.group(1)
        
        # Extract temperature
        temp_match = re.search(r'(\d+\.?\d*)\s*[fF]', text)
        if temp_match:
            facts['temperature'] = temp_match.group(1)
        
        # Extract diagnosis
        diagnosis_patterns = [
            r'Assessment:\s*([^\.]+)',
            r'Diagnosis:\s*([^\.]+)',
            r'with\s+([^,\.]+)',
        ]
        
        for pattern in diagnosis_patterns:
            diagnosis_match = re.search(pattern, text, re.IGNORECASE)
            if diagnosis_match:
                facts['diagnosis'] = diagnosis_match.group(1).strip()
                break
        
        return facts
    
    def _verify_medical_terms(self, summary: str) -> Dict[str, Any]:
        """Verify that medical terms in summary are valid."""
        
        # Extract medical terms from summary
        words = summary.lower().split()
        medical_terms_found = [word for word in words if word in self.medical_terms]
        
        # Check for potentially incorrect medical terms
        suspicious_terms = []
        for word in words:
            if word not in self.medical_terms and len(word) > 3:
                # Check if it looks like a medical term but isn't in our list
                if any(char.isupper() for char in word) or word.endswith(('itis', 'osis', 'emia')):
                    suspicious_terms.append(word)
        
        return {
            'medical_terms_found': medical_terms_found,
            'suspicious_terms': suspicious_terms,
            'medical_term_confidence': len(medical_terms_found) / max(len(words), 1)
        }
    
    def _check_semantic_similarity(self, note: str, summary: str) -> Dict[str, Any]:
        """Check semantic similarity between note and summary."""
        
        try:
            # Vectorize texts
            vectors = self.vectorizer.fit_transform([note, summary])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return {
                'semantic_similarity': float(similarity),
                'similarity_adequate': similarity > 0.3  # Threshold for adequacy
            }
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return {
                'semantic_similarity': 0.0,
                'similarity_adequate': False
            }
    
    def _check_logical_consistency(self, note: str, summary: str) -> Dict[str, Any]:
        """Check for logical inconsistencies in the summary."""
        
        warnings = []
        
        # Check for contradictory statements
        contradictions = [
            ('normal', 'abnormal'),
            ('elevated', 'normal'),
            ('high', 'low'),
            ('positive', 'negative')
        ]
        
        note_lower = note.lower()
        summary_lower = summary.lower()
        
        for term1, term2 in contradictions:
            if term1 in note_lower and term2 in summary_lower:
                warnings.append(f"Potential contradiction: {term1} vs {term2}")
        
        # Check for missing critical information
        critical_info = ['assessment', 'plan', 'diagnosis']
        missing_critical = []
        
        for info in critical_info:
            if info in note_lower and info not in summary_lower:
                missing_critical.append(info)
        
        if missing_critical:
            warnings.append(f"Missing critical information: {', '.join(missing_critical)}")
        
        return {
            'logical_warnings': warnings,
            'logical_consistency_score': 1.0 - (len(warnings) / 5.0)  # Normalize to 0-1
        }
    
    def _calculate_hallucination_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall hallucination score."""
        
        # Weight different factors
        weights = {
            'factual_consistency_score': 0.4,
            'medical_term_confidence': 0.2,
            'semantic_similarity': 0.3,
            'logical_consistency_score': 0.1
        }
        
        score = 0.0
        for factor, weight in weights.items():
            if factor in results:
                score += results[factor] * weight
        
        return min(max(score, 0.0), 1.0)  # Clamp to 0-1
    
    def evaluate_summaries(self, clinical_notes: List[str], summaries: List[str]) -> Dict[str, Any]:
        """
        Evaluate multiple summaries for hallucinations.
        
        Args:
            clinical_notes: List of original clinical notes
            summaries: List of generated summaries
            
        Returns:
            Dictionary containing evaluation results
        """
        
        logger.info(f"Evaluating {len(summaries)} summaries for hallucinations")
        
        results = {
            'total_summaries': len(summaries),
            'evaluations': [],
            'hallucination_scores': [],
            'warnings': [],
            'high_risk_summaries': []
        }
        
        for i, (note, summary) in enumerate(zip(clinical_notes, summaries)):
            evaluation = self.detect_hallucinations(note, summary)
            results['evaluations'].append(evaluation)
            results['hallucination_scores'].append(evaluation['hallucination_score'])
            
            # Collect warnings
            results['warnings'].extend(evaluation.get('warnings', []))
            
            # Flag high-risk summaries
            if evaluation['hallucination_score'] < 0.7:
                results['high_risk_summaries'].append({
                    'index': i,
                    'score': evaluation['hallucination_score'],
                    'summary': summary[:100] + '...' if len(summary) > 100 else summary
                })
        
        # Calculate aggregate statistics
        if results['hallucination_scores']:
            results['mean_hallucination_score'] = np.mean(results['hallucination_scores'])
            results['std_hallucination_score'] = np.std(results['hallucination_scores'])
            results['high_risk_percentage'] = len(results['high_risk_summaries']) / len(summaries) * 100
        else:
            results['mean_hallucination_score'] = 0.0
            results['std_hallucination_score'] = 0.0
            results['high_risk_percentage'] = 0.0
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str = "hallucination_evaluation.json"):
        """Save hallucination evaluation results."""
        results_path = Path(config.get("paths.results")) / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Hallucination evaluation results saved to: {results_path}")
        return str(results_path)

def run_hallucination_evaluation(clinical_notes: List[str], summaries: List[str]) -> Dict[str, Any]:
    """
    Run hallucination evaluation on generated summaries.
    
    Args:
        clinical_notes: List of original clinical notes
        summaries: List of generated summaries
        
    Returns:
        Evaluation results
    """
    
    logger.info("Starting hallucination evaluation")
    
    # Initialize detector
    detector = HallucinationDetector()
    
    # Run evaluation
    results = detector.evaluate_summaries(clinical_notes, summaries)
    
    # Save results
    results_path = detector.save_evaluation_results(results)
    results['results_path'] = results_path
    
    logger.info(f"Hallucination evaluation completed. Mean score: {results['mean_hallucination_score']:.4f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_notes = [
        "58 year old M with essential hypertension, BP 165/95 mmHg. Plan: ACE inhibitor therapy.",
        "65 year old F with poorly controlled Type 2 diabetes, HbA1c 9.2%. Plan: Insulin adjustment."
    ]
    
    sample_summaries = [
        "58 year old M with essential hypertension, BP 165/95 mmHg. Plan: ACE inhibitor therapy.",
        "65 year old F with diabetes, HbA1c 9.2%. Plan: Insulin adjustment and dietary changes."
    ]
    
    results = run_hallucination_evaluation(sample_notes, sample_summaries)
    print("Hallucination evaluation results:", results) 