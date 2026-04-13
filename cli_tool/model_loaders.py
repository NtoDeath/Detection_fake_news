"""
Model loaders for CLI tool.
Simplified wrappers around Style, Knowledge, Fusion branches.
"""

import os
import pickle
import joblib
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax


class StyleDetectorWrapper:
    """Lightweight Style detector wrapper"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.tokenizer = None
        self.roberta_model = None
        self.rf_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
    
    def _load_models(self):
        """Load RoBERTa + Random Forest"""
        roberta_path = self.models_dir / "style" / "roberta_fine_tunned"
        rf_path = self.models_dir / "style" / "best_model.pkl"
        
        if not roberta_path.exists():
            raise FileNotFoundError(f"❌ RoBERTa model not found at {roberta_path}")
        if not rf_path.exists():
            raise FileNotFoundError(f"❌ RF model not found at {rf_path}")
        
        # Temporarily change to model directory so relative paths work
        original_cwd = os.getcwd()
        try:
            os.chdir(str(roberta_path))
            # Now load with relative paths - this will work on any machine
            self.tokenizer = AutoTokenizer.from_pretrained(
                ".", 
                use_fast=False, 
                trust_remote_code=True
            )
        finally:
            # Always restore original directory
            os.chdir(original_cwd)
        
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained(str(roberta_path))
        self.roberta_model = self.roberta_model.to(self.device)
        self.roberta_model.eval()
        self.rf_model = joblib.load(rf_path)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if text is fake news.
        
        Returns:
            {
                "is_fake": bool,
                "confidence": float (0-1),
                "roberta_score": float (0-1)
            }
        """
        if not text or len(text.strip()) < 10:
            return {"is_fake": False, "confidence": 0.0, "roberta_score": 0.0}
        
        # RoBERTa prediction
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        
        probs = softmax(outputs.logits.cpu().detach().numpy()[0])
        roberta_fake_prob = float(probs[1])  # Label 1 = FAKE
        
        # RF prediction (simplified - using RoBERTa + basic features)
        # In production, would extract full feature set, but for CLI we simplify
        basic_features = np.array([
            len(text.split()),  # word count
            text.count("!") + text.count("?"),  # punctuation
            roberta_fake_prob,  # RoBERTa score
            text.count("\n"),  # newlines
        ])
        
        # Simple heuristic for demo (can replace with actual RF predict)
        is_fake = roberta_fake_prob > 0.5
        confidence = max(probs)
        
        return {
            "is_fake": is_fake,
            "confidence": float(confidence),
            "roberta_score": roberta_fake_prob
        }


class KnowledgeDetectorWrapper:
    """Lightweight Knowledge detector wrapper"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.nli_pipeline = None
        self._load_models()
    
    def _load_models(self):
        """Load DeBERTa NLI verifier"""
        # DeBERTa loads from HuggingFace
        self.nli_pipeline = pipeline(
            "text-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Check claim using knowledge base.
        
        Returns:
            {
                "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO",
                "confidence": float (0-1)
            }
        """
        if not text or len(text.strip()) < 10:
            return {"verdict": "NOT_ENOUGH_INFO", "confidence": 0.0}
        
        # Simplified: just use NLI pipeline on text itself (in production, retrieve evidence first)
        try:
            # Create a dummy entailment check
            result = self.nli_pipeline(text, truncation=True, max_length=512)[0]
            label = result['label'].lower()
            conf = result['score']
            
            if "entail" in label:
                verdict = "SUPPORTED"
            elif "contradict" in label:
                verdict = "REFUTED"
            else:
                verdict = "NOT_ENOUGH_INFO"
            
            return {
                "verdict": verdict,
                "confidence": float(conf)
            }
        except Exception as e:
            return {"verdict": "ERROR", "confidence": 0.0, "error": str(e)}


class FusionFuzzyWrapper:
    """Fusion meta-learner (Stacked RF) - MODEL REQUIRED"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.meta_model = None
        self._load_models()
    
    def _load_models(self):
        """Load Stacked RF meta-model from pickle"""
        fusion_path = self.models_dir / "fusion" / "stacked_rf_model.pkl"
        
        if not fusion_path.exists():
            raise FileNotFoundError(
                f"❌ Stacked RF model not found at {fusion_path}\n"
                f"   Run: python {self.models_dir.parent}/models_copy.py"
            )
        
        try:
            self.meta_model = pickle.load(open(fusion_path, 'rb'))
        except Exception as e:
            raise RuntimeError(f"Failed to load Stacked RF model: {e}")
    
    def predict(self, style_pred: int, style_conf: float, 
                knowledge_verdict: str, knowledge_conf: float) -> Dict[str, Any]:
        """
        Fuse Style + Knowledge predictions using trained Stacked RF.
        
        Args:
            style_pred: 1 if fake, 0 if real
            style_conf: confidence [0-1]
            knowledge_verdict: "SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"
            knowledge_conf: confidence [0-1]
        
        Returns:
            {
                "is_fake": bool,
                "confidence": float (0-1),
                "reasoning": str
            }
        """
        # Convert knowledge verdict to binary
        knowledge_pred = 1 if knowledge_verdict == "REFUTED" else 0
        
        # Prepare meta-learner input: [style_pred, style_conf, knowledge_pred, knowledge_conf]
        X = np.array([[
            style_pred,
            style_conf,
            knowledge_pred,
            knowledge_conf
        ]])
        
        try:
            pred = self.meta_model.predict(X)[0]
            proba = self.meta_model.predict_proba(X)[0]
            confidence = float(max(proba))
            
            return {
                "is_fake": bool(pred == 1),
                "confidence": confidence,
                "reasoning": f"Stacked RF: {proba[0]:.1%} (real) vs {proba[1]:.1%} (fake)"
            }
        except Exception as e:
            raise RuntimeError(f"Fusion prediction failed: {e}")
