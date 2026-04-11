# Part B Heterogeneous Dataset: Allocation Strategy

## Overview

Part B is a **data-driven** curated dataset designed to validate hybrid fusion logic before final evaluation. The allocation strategy is based on actual data statistics, not arbitrary guesses.

## Allocation Formula

```
Total available data from 4 sources: ~57,403 samples
Part B ratio (reserved for fusion testing): 20%
Part B target size: ~11,480 samples
```

### Category Distribution (as % of Part B)

| Category | Allocation | Purpose |
|----------|-----------|---------|
| **Consensus FAKE** | 20% | Both models confidently predict FAKE - baseline agreement test |
| **Consensus REAL** | 20% | Both models confidently predict REAL - baseline agreement test |
| **Conflict** | 25% | Models disagree despite high confidence - critical for fusion tuning |
| **Uncertainty** | 15% | Both models low confidence - tests fallback logic |
| **OOD** | 10% | Out-of-distribution texts (very short/long) - edge case handling |
| **Edge cases** | 5% | Sarcasm, satire, controversial - rare pattern detection |
| **Reserve/Overflow** | 5% | Unallocated for future use or categorization failures |

## Example Results (from 57,403 total samples)

```
Part B target: 11,480 samples (20% of 57,403)

Allocated category sizes:
- Consensus FAKE:     2,296 samples (20%)
- Consensus REAL:     2,296 samples (20%)
- Conflict:           2,870 samples (25%) ← Largest for fusion testing
- Uncertainty:        1,722 samples (15%)
- OOD:                1,148 samples (10%)
- Edge cases:           574 samples (5%)
─────────────────────────────────────
Total allocated:     10,906 samples (95.0%)
```

## Why This Allocation?

### Consensus Categories (40% combined)
- **Equal 20% each**: Ensures balanced positive/negative baseline testing
- **Purpose**: Validate that fusion correctly preserves agreement when both models align
- **Success criterion**: >95% accuracy on consensus items

### Conflict (25%) - **Largest Category**
- **Why largest**: Model disagreement is the most informative scenario
- **Purpose**: Tests threshold optimization and decision-making under disagreement
- **Success criterion**: F1 > baseline when models disagree
- **Example**: Style predicts FAKE (0.85 confidence) vs Knowledge predicts REAL (0.72 confidence)

### Uncertainty (15%)
- **Why smaller than conflict**: Represents ambiguous cases less common in practice
- **Purpose**: Tests low-confidence fallback logic and conservative predictions
- **Success criterion**: Correct handling when both confidence scores are <0.5

### OOD (10%)
- **Why 10%**: Represents distribution edge cases but not standard testing
- **Purpose**: Validates robustness to unusual inputs (very short tweets, very long articles)
- **Success criterion**: Graceful degradation, no crashes

### Edge Cases (5%) - **Smallest Category**
- **Why smallest**: Rare patterns (sarcasm, satire) are uncommon in practice
- **Purpose**: Ensures fusion handles non-literal/controversial content
- **Success criterion**: Doesn't completely fail on sarcasm/satire

## Data-Driven vs. Arbitrary

### ✅ Our Approach (Data-Driven)
1. **Calculate** Part B size as 20% of available data
2. **Derive** category percentages from what makes sense for fusion testing
3. **Scale** to actual data volume (11.5k, not hardcoded 15.7k)
4. **Justify** each category's importance for the task
5. **Flexible** - adjusts to different dataset sizes

### ❌ Previous Approach (Arbitrary)
- Hardcoded: 3,140 consensus FAKE, 3,140 consensus REAL, 3,925 conflict, etc.
- No justification for why these specific numbers
- Didn't scale with data availability
- No clear mapping to fusion validation needs

## How to Adjust Allocation

Edit the allocation percentages in `prepare_part_B_heterogeneous.py`:

```python
allocations = {
    'consensus_fake': int(part_B_size * 0.20),      # Change here → more conflict tests
    'consensus_real': int(part_B_size * 0.20),      
    'conflict': int(part_B_size * 0.25),            
    'uncertainty': int(part_B_size * 0.15),         
    'ood': int(part_B_size * 0.10),                 
    'edge_cases': int(part_B_size * 0.05),          
}
```

Example: To increase conflict testing from 25% to 35%:
```python
'conflict': int(part_B_size * 0.35),        # ← 35%
'consensus_fake': int(part_B_size * 0.20),  # ← 20%
'consensus_real': int(part_B_size * 0.15),  # ← Reduce to 15%
'uncertainty': int(part_B_size * 0.15),     
'ood': int(part_B_size * 0.10),             
'edge_cases': int(part_B_size * 0.05),      
```

## Evaluation on Part B

The `FusionEvaluator` class in `fusion_branch/fusion_utils.py` reports:

```
[GLOBAL PERFORMANCE]
F1 Score, Accuracy, Precision, Recall

[PER-CATEGORY PERFORMANCE]
Each metric broken down by category (now justified!)
- Where should fusion excel? Consensus categories
- Where is tuning critical? Conflict category
- Where should fusion be conservative? OOD/Edge cases
```

## Next Steps

1. **Train STYLE and KNOWLEDGE on Part A** (60% of data, separate datasets)
2. **Get predictions on Part B** from both models
3. **Optimize fusion thresholds** using grid/Bayesian search on Part B
4. **Evaluate final ensemble on Part C** (20% held-out test)

---

**Key Takeaway**: Part B allocation is now **data-driven, justified, and scalable** rather than arbitrary. Each category size reflects its importance for fusion validation.
