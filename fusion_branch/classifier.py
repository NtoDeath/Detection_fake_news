"""Unified classifier interface for ensemble fake news detection.

Provides consistent prediction API across model types:
- Classical ML: Random Forest, XGBoost
- Deep learning: RoBERTa, fine-tuned BERT
- Knowledge-based: Claim verification + evidence matching

Design Pattern
--------------
Adapter pattern: wraps scikit-learn or transformers models
to provide uniform .predict() interface.

Usage
-----
```python
# With scikit-learn model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
classifier = Classifier(rf)
predictions = classifier.predict(X_test)

# With transformers model
roberta_model = AutoModelForSequenceClassification.from_pretrained(...)
classifier = Classifier(roberta_model)
predictions = classifier.predict(test_sentences)
```

Dependencies
------------
- Supports any model implementing .predict(X) method
- Compatible with scikit-learn, transformers, custom models
"""

from typing import Any, Optional


class Classifier:
    """Unified inference interface for fake news decision models.
    
    Wraps trained models (ML/DL) with consistent prediction interface.
    Enables model-agnostic ensemble orchestration.
    
    Attributes
    ----------
    model : object
        Trained classifier implementing .predict(X) method.
        Supported: scikit-learn, transformers, custom models.
    
    Notes
    -----
    This class is intentionally minimal (adapter pattern).
    Actual prediction logic delegated to wrapped model.
    """
    
    def __init__(self, model: Any) -> None:
        """Initialize classifier with trained model.
        
        Parameters
        ----------
        model : object
            Trained classifier or predictor object.
            Must implement .predict(X) method.
        
        Notes
        -----
        No validation of model type (duck typing).
        Errors occur at prediction time if model incompatible.
        """
        self.model: Any = model

    def predict(self, input_data: Any) -> Any:
        """Generate predictions using wrapped model.
        
        Parameters
        ----------
        input_data : array-like or tensor
            Input features or text to classify.
            Format depends on wrapped model type.
        
        Returns
        -------
        array-like or tensor
            Model predictions (class labels or probabilities).
            Output format matches wrapped model.
        
        Notes
        -----
        Directly delegates to model.predict(input_data).
        Raises error if model lacks .predict() method.
        
        Examples
        --------
        >>> clf = Classifier(trained_rf_model)
        >>> predictions = clf.predict(X_test)
        >>> probabilities = clf.model.predict_proba(X_test)
        """
        # Delegate prediction to wrapped model
        # Supports scikit-learn: returns class labels (0/1)
        # or transformers: returns logits/probabilities (depends on config)
        return self.model.predict(input_data)