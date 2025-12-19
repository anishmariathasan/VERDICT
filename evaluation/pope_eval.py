"""POPE (Polling-based Object Probing Evaluation) for LVLMs.

Adapted for medical imaging to evaluate hallucination rates by probing
the model about the existence of specific medical findings.
"""

import logging
import random
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class POPEEvaluator:
    """
    Evaluator for POPE benchmark.
    
    Generates questions about the presence/absence of findings and
    evaluates the model's responses (Yes/No).
    
    Attributes:
        model: The LVLM to evaluate.
        findings_list: List of possible medical findings.
    """
    
    def __init__(
        self,
        model: Any,
        findings_list: Optional[List[str]] = None,
    ) -> None:
        """
        Initialise POPE evaluator.
        
        Args:
            model: The model wrapper.
            findings_list: List of findings to probe.
        """
        self.model = model
        self.findings_list = findings_list or [
            "cardiomegaly", "pleural effusion", "pneumonia", "pneumothorax",
            "atelectasis", "consolidation", "edema", "nodule"
        ]
        
    def generate_questions(
        self,
        ground_truth_findings: List[str],
        num_questions: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Generate POPE questions for a sample.
        
        Args:
            ground_truth_findings: List of findings present in the image.
            num_questions: Number of questions to generate.
        
        Returns:
            List of question dictionaries (question, answer, type).
        """
        questions = []
        
        # Positive questions (findings present)
        for finding in ground_truth_findings:
            if finding in self.findings_list:
                questions.append({
                    "question": f"Is there {finding} in the image?",
                    "answer": "yes",
                    "type": "positive"
                })
                
        # Negative questions (findings absent)
        absent_findings = [f for f in self.findings_list if f not in ground_truth_findings]
        random.shuffle(absent_findings)
        
        for finding in absent_findings:
            questions.append({
                "question": f"Is there {finding} in the image?",
                "answer": "no",
                "type": "negative"
            })
            
        # Sample to desired number
        if len(questions) > num_questions:
            questions = random.sample(questions, num_questions)
            
        return questions
    
    def evaluate(
        self,
        image: torch.Tensor,
        questions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Run evaluation on a single image.
        
        Args:
            image: Input image.
            questions: List of questions.
        
        Returns:
            Dictionary of accuracy metrics.
        """
        correct = 0
        total = len(questions)
        
        for q in questions:
            # Generate answer using model
            # This assumes model.generate supports VQA prompt
            response = self.model.generate(
                image=image,
                prompt=f"Question: {q['question']} Answer:",
                max_new_tokens=10
            )["generated_text"].lower()
            
            # Check correctness (simple yes/no check)
            pred = "yes" if "yes" in response else "no"
            if pred == q["answer"]:
                correct += 1
                
        return {
            "pope_accuracy": correct / total if total > 0 else 0.0
        }
