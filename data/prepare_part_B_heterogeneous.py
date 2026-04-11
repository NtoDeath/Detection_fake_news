"""
Part A & B Dataset Split Construction

Implements 8-phase strategy to create disjoint Part A (80%) and Part B (20%) datasets
from heterogeneous sources with normalized labels and consistent filtering.

## Phase Strategy:

1. **Inputs & Données Brutes**
   - Load dataset.csv (LIAR+Twitter+UoVictoria, already normalized by data_extraction.py)
   - Load groundtruth.csv (ClaimBuster claims with Verdict)
   - Load train.jsonl (FEVER evidence-based claims)

2. **Normalisation & Filtrage**
   - Normalize groundtruth: Filter Verdict!=0, map -1→0, 1→1
   - Normalize FEVER: Filter label!='NOT_ENOUGH_INFO', map REFUTES→0, SUPPORTS→1
   - dataset.csv already normalized (0=FAKE from data_extraction.py)

3. **Combine & Statistics**
   - Merge all 3 sources with source/dataset markers
   - Calculate statistics (rows, class ratios, source distributions)

4. **Stratified 80/20 Split**
   - Single split on unified+normalized data to ensure Part A/B disjoint
   - Stratify on label (0/1 ratio preserved)

5. **Extract Part A** (80% in original formats for training branches)
   - dataset_partA.csv: STYLE-based rows in original format [text, label]
   - groundtruth_partA.csv: KNOWLEDGE-based (Groundtruth) rows with original columns
   - train_partA.jsonl: KNOWLEDGE-based (FEVER) rows in original JSONL format

6. **Save Part B** (20% unified normalized CSV for fusion validation)
   - part_B_validation.csv: Unified+normalized with [text, label, source]

7. **Summary Report**
   - Statistics by source and label
   - Verification of no row overlap
   - Data integrity checks

8. **Integration Notes**
   - Update feature_extraction.py to use dataset_partA.csv
   - Update train_claim_detector.py to use groundtruth_partA.csv
   - Update evaluate_pipeline.py to use train_partA.jsonl
   - Update unified_main.ipynb Phase 0 to call this script
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def phase_1_load_inputs(base_path='.'):
    """Phase 1: Load raw inputs from 3 sources."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: LOAD INPUTS & DONNÉES BRUTES")
    logger.info("="*70)
    
    base = Path(base_path)
    
    # Load dataset.csv (already normalized by data_extraction.py)
    logger.info("\n1.1 Loading dataset.csv (LIAR+Twitter+UoVictoria)...")
    try:
        dataset_df = pd.read_csv(base / 'dataset.csv')
        logger.info(f"     ✓ Loaded {len(dataset_df)} rows")
        logger.info(f"     Columns: {list(dataset_df.columns)}")
        logger.info(f"     Label distribution: {dataset_df.get('label', pd.Series()).value_counts().to_dict()}")
    except FileNotFoundError:
        logger.error("     ✗ dataset.csv not found. Run data_extraction.py first.")
        dataset_df = pd.DataFrame()
    
    # Load groundtruth.csv
    logger.info("\n1.2 Loading groundtruth.csv (ClaimBuster)...")
    try:
        groundtruth_df = pd.read_csv(Path(base / '..' / 'knowledge_branch' / 'groundtruth.csv'))
        logger.info(f"     ✓ Loaded {len(groundtruth_df)} rows")
        logger.info(f"     Columns: {list(groundtruth_df.columns)}")
        if 'Verdict' in groundtruth_df.columns:
            logger.info(f"     Verdict distribution: {groundtruth_df['Verdict'].value_counts().to_dict()}")
    except FileNotFoundError:
        logger.error("     ✗ groundtruth.csv not found.")
        groundtruth_df = pd.DataFrame()
    
    # Load train.jsonl (FEVER)
    logger.info("\n1.3 Loading train.jsonl (FEVER)...")
    try:
        fever_df = pd.read_json(Path(base / 'knowledge_based' / 'train.jsonl'), lines=True)
        logger.info(f"     ✓ Loaded {len(fever_df)} rows")
        logger.info(f"     Columns: {list(fever_df.columns)}")
        if 'label' in fever_df.columns:
            logger.info(f"     Label distribution: {fever_df['label'].value_counts().to_dict()}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"     ✗ train.jsonl not found or invalid: {e}")
        fever_df = pd.DataFrame()
    
    return dataset_df, groundtruth_df, fever_df


def normalize_groundtruth(df):
    """
    Phase 2.2: Normalize groundtruth dataset.
    
    - Filter: Keep only Verdict != 0 (exclude uncertain entries)
    - Map: -1 → 0 (FAKE/REFUTED), 1 → 1 (TRUE/SUPPORTED)
    """
    logger.info("\n2.2 Normalizing groundtruth.csv...")
    
    if df.empty:
        logger.warning("     Groundtruth dataframe is empty, skipping.")
        return pd.DataFrame()
    
    # Check required columns
    if 'Verdict' not in df.columns or 'Text' not in df.columns:
        logger.error(f"     Missing required columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    initial_count = len(df)
    
    # Filter: Remove Verdict == 0 (uncertain entries)
    df_valid = df[df['Verdict'].isin([-1, 1])].copy()
    filtered_count = initial_count - len(df_valid)
    logger.info(f"     Filtered out {filtered_count} uncertain entries (Verdict==0)")
    
    # Normalize labels: -1 → 0 (FAKE), 1 → 1 (TRUE)
    df_valid['label'] = df_valid['Verdict'].apply(lambda x: 0 if x == -1 else 1)
    df_valid['text'] = df_valid['Text'].fillna('')
    
    # Create output with required columns
    result = df_valid[['text', 'label', 'Verdict', 'Text']].copy()
    result['source'] = 'GROUNDTRUTH'
    
    logger.info(f"     ✓ Output: {len(result)} rows")
    logger.info(f"     Label dist: {result['label'].value_counts().to_dict()}")
    
    return result


def normalize_fever(df):
    """
    Phase 2.3: Normalize FEVER dataset.
    
    - Filter: Keep only label in ['SUPPORTS', 'REFUTES'] (exclude NOT_ENOUGH_INFO)
    - Map: 'REFUTES' → 0 (FAKE), 'SUPPORTS' → 1 (TRUE)
    """
    logger.info("\n2.3 Normalizing FEVER (train.jsonl)...")
    
    if df.empty:
        logger.warning("     FEVER dataframe is empty, skipping.")
        return pd.DataFrame()
    
    # Check required columns
    if 'label' not in df.columns or 'claim' not in df.columns:
        logger.error(f"     Missing required columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    initial_count = len(df)
    
    # Filter: Remove NOT_ENOUGH_INFO labels
    df_valid = df[df['label'].isin(['SUPPORTS', 'REFUTES'])].copy()
    filtered_count = initial_count - len(df_valid)
    logger.info(f"     Filtered out {filtered_count} NOT_ENOUGH_INFO entries")
    
    # Normalize labels: REFUTES → 0 (FAKE), SUPPORTS → 1 (TRUE)
    df_valid['label_binary'] = df_valid['label'].apply(lambda x: 0 if x == 'REFUTES' else 1)
    df_valid['text'] = df_valid['claim'].fillna('')
    
    # Create output with required columns
    result = df_valid[['text', 'label_binary', 'label', 'claim']].copy()
    result['label'] = result['label_binary']
    result['source'] = 'FEVER'
    
    logger.info(f"     ✓ Output: {len(result)} rows")
    logger.info(f"     Label dist: {result['label'].value_counts().to_dict()}")
    
    return result


def phase_2_normalize_and_filter(dataset_df, groundtruth_df, fever_df):
    """Phase 2: Normalize and filter all 3 sources."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: NORMALISATION & FILTRAGE")
    logger.info("="*70)
    
    # Dataset.csv is already normalized by data_extraction.py
    logger.info("\n2.1 dataset.csv already normalized (0=FAKE from data_extraction.py)")
    logger.info(f"     Keeping {len(dataset_df)} rows")
    
    dataset_df['source'] = 'DATA'
    dataset_normalized = dataset_df[['text', 'label', 'source']].copy()
    
    # Normalize groundtruth and FEVER
    groundtruth_normalized = normalize_groundtruth(groundtruth_df)
    fever_normalized = normalize_fever(fever_df)
    
    return dataset_normalized, groundtruth_normalized, fever_normalized


def phase_3_combine_and_statistics(dataset_normalized, groundtruth_normalized, fever_normalized):
    """Phase 3: Combine all 3 sources and compute statistics."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: COMBINE & STATISTICS")
    logger.info("="*70)
    
    # Combine all sources
    logger.info("\n3.1 Combining all sources...")
    
    dfs_to_combine = []
    for name, df in [('DATA', dataset_normalized), ('GROUNDTRUTH', groundtruth_normalized), ('FEVER', fever_normalized)]:
        if not df.empty:
            dfs_to_combine.append(df)
            logger.info(f"     {name:15}: {len(df):6} rows")
    
    if not dfs_to_combine:
        logger.error("     No valid data sources found!")
        return pd.DataFrame()
    
    df_all = pd.concat(dfs_to_combine, ignore_index=True)
    
    # Remove exact text duplicates within each source separately, but allow across sources
    logger.info(f"\n3.2 Removing duplicates...")
    initial_count = len(df_all)
    df_all = df_all.drop_duplicates(subset=['text', 'source'])
    dedup_count = initial_count - len(df_all)
    logger.info(f"     Removed {dedup_count} duplicate rows (same text, same source)")
    
    # Compute statistics
    logger.info(f"\n3.3 Statistics:")
    logger.info(f"     Total unique rows: {len(df_all)}")
    logger.info(f"     Label distribution (0=FAKE, 1=TRUE):")
    for label in [0, 1]:
        count = len(df_all[df_all['label'] == label])
        pct = (count / len(df_all) * 100) if len(df_all) > 0 else 0
        logger.info(f"       {label}: {count:6} ({pct:5.1f}%)")
    
    logger.info(f"     Source distribution:")
    for source in df_all['source'].unique():
        count = len(df_all[df_all['source'] == source])
        pct = (count / len(df_all) * 100)
        logger.info(f"       {source:15}: {count:6} ({pct:5.1f}%)")
    
    return df_all


def phase_4_stratified_split(df_all, test_size=0.20, random_state=42):
    """Phase 4: Stratified 80/20 split."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: STRATIFIED 80/20 SPLIT")
    logger.info("="*70)
    
    logger.info(f"\n4.1 Splitting with stratification (test_size={test_size})...")
    
    # Stratified split to preserve label ratio
    part_A, part_B = train_test_split(
        df_all,
        test_size=test_size,
        stratify=df_all['label'],
        random_state=random_state
    )
    
    logger.info(f"\n4.2 Split results:")
    logger.info(f"     Part A (80%): {len(part_A)} rows")
    logger.info(f"     Part B (20%): {len(part_B)} rows")
    
    # Verify label distributions
    logger.info(f"\n4.3 Label distribution:")
    for part_name, part_df in [('Part A', part_A), ('Part B', part_B)]:
        logger.info(f"     {part_name}:")
        for label in [0, 1]:
            count = len(part_df[part_df['label'] == label])
            pct = (count / len(part_df) * 100) if len(part_df) > 0 else 0
            logger.info(f"       {label}: {count:6} ({pct:5.1f}%)")
    
    # Verify no overlap
    logger.info(f"\n4.4 Overlap check:")
    overlap = pd.merge(
        part_A[['text', 'label']],
        part_B[['text', 'label']],
        on=['text', 'label'],
        how='inner'
    )
    if len(overlap) == 0:
        logger.info(f"     ✓ No overlap between Part A and Part B (disjoint guaranteed)")
    else:
        logger.warning(f"     ✗ WARNING: {len(overlap)} overlapping rows found!")
    
    return part_A, part_B


def phase_5_extract_part_a(part_A, base_path='.', original_data_base='.'):
    """Phase 5: Extract Part A in 3 original formats for training branches."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 5: EXTRACT PART A (80% in original formats)")
    logger.info("="*70)
    
    base = Path(base_path)
    splits_dir = base / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    # Load original data sources to get full row context
    logger.info("\n5.1 Loading original data sources for context...")
    original_data = Path(original_data_base)
    
    dataset_orig = pd.read_csv(original_data / 'dataset.csv')
    groundtruth_orig = pd.read_csv(original_data / '..' / 'knowledge_branch' / 'groundtruth.csv')
    fever_orig = pd.read_json(original_data / 'knowledge_based' / 'train.jsonl', lines=True)
    
    # Extract dataset_partA.csv (STYLE branch)
    logger.info("\n5.2 Extracting dataset_partA.csv for STYLE branch...")
    dataset_rows = part_A[part_A['source'] == 'DATA'].copy()
    
    # Get original columns from dataset.csv
    dataset_partA = dataset_orig[dataset_orig['text'].isin(dataset_rows['text'])].copy()
    dataset_partA = dataset_partA.drop_duplicates(subset=['text'])
    dataset_partA_path = splits_dir / 'dataset_partA.csv'
    dataset_partA.to_csv(dataset_partA_path, index=False)
    logger.info(f"     ✓ Saved {len(dataset_partA)} rows to {dataset_partA_path.name}")
    
    # Extract groundtruth_partA.csv (KNOWLEDGE branch - claim detection)
    logger.info("\n5.3 Extracting groundtruth_partA.csv for KNOWLEDGE branch...")
    groundtruth_rows = part_A[part_A['source'] == 'GROUNDTRUTH'].copy()
    
    # Get original columns from groundtruth.csv (keep all original columns)
    groundtruth_partA = groundtruth_orig[groundtruth_orig['Text'].isin(groundtruth_rows['text'])].copy()
    groundtruth_partA = groundtruth_partA.drop_duplicates(subset=['Text'])
    groundtruth_partA_path = splits_dir / 'groundtruth_partA.csv'
    groundtruth_partA.to_csv(groundtruth_partA_path, index=False)
    logger.info(f"     ✓ Saved {len(groundtruth_partA)} rows to {groundtruth_partA_path.name}")
    
    # Extract train_partA.jsonl (KNOWLEDGE branch - verification)
    logger.info("\n5.4 Extracting train_partA.jsonl for KNOWLEDGE branch...")
    fever_rows = part_A[part_A['source'] == 'FEVER'].copy()
    
    # Get original rows from FEVER with all columns
    train_partA = fever_orig[fever_orig['claim'].isin(fever_rows['text'])].copy()
    train_partA = train_partA.drop_duplicates(subset=['claim'])
    train_partA_path = splits_dir / 'train_partA.jsonl'
    train_partA.to_json(train_partA_path, orient='records', lines=True)
    logger.info(f"     ✓ Saved {len(train_partA)} rows to {train_partA_path.name}")
    
    return dataset_partA_path, groundtruth_partA_path, train_partA_path


def phase_6_save_part_b(part_B, base_path='.'):
    """Phase 6: Save Part B as unified normalized CSV."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 6: SAVE PART B (20% unified normalized)")
    logger.info("="*70)
    
    base = Path(base_path)
    splits_dir = base / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    # Save Part B unified CSV
    logger.info("\n6.1 Saving part_B_validation.csv...")
    
    # Keep only essential columns for fusion validation
    part_B_csv = part_B[['text', 'label', 'source']].copy()
    part_B_csv = part_B_csv.drop_duplicates(subset=['text'])
    
    part_B_path = splits_dir / 'part_B_validation.csv'
    part_B_csv.to_csv(part_B_path, index=False)
    
    logger.info(f"     ✓ Saved {len(part_B_csv)} rows to {part_B_path.name}")
    logger.info(f"     Columns: {list(part_B_csv.columns)}")
    logger.info(f"     Label distribution: {part_B_csv['label'].value_counts().to_dict()}")
    logger.info(f"     Source distribution: {part_B_csv['source'].value_counts().to_dict()}")
    
    return part_B_path


def phase_7_summary_report(part_A, part_B, dataset_partA_path, groundtruth_partA_path, train_partA_path, part_B_path):
    """Phase 7: Generate summary report."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 7: SUMMARY REPORT")
    logger.info("="*70)
    
    logger.info("\n7.1 Output Files Generated:")
    logger.info(f"     Part A (STYLE branch)")
    logger.info(f"       ✓ {dataset_partA_path.name}")
    logger.info(f"     Part A (KNOWLEDGE branches)")
    logger.info(f"       ✓ {groundtruth_partA_path.name}")
    logger.info(f"       ✓ {train_partA_path.name}")
    logger.info(f"     Part B (Fusion Validation)")
    logger.info(f"       ✓ {part_B_path.name}")
    
    logger.info("\n7.2 Data Integrity Verification:")
    logger.info(f"     Part A rows: {len(part_A)}")
    logger.info(f"     Part B rows: {len(part_B)}")
    logger.info(f"     Total (should equal input): {len(part_A) + len(part_B)}")
    
    # Verify no text overlap (most important check)
    all_texts_A = set(part_A['text'].unique())
    all_texts_B = set(part_B['text'].unique())
    overlap = all_texts_A.intersection(all_texts_B)
    if len(overlap) == 0:
        logger.info(f"     ✓ No text overlap between Part A and B (disjoint guaranteed)")
    else:
        logger.warning(f"     ✗ WARNING: {len(overlap)} overlapping texts!")
    
    logger.info("\n7.3 Label Distribution:")
    logger.info(f"     Part A (80%):")
    for label in [0, 1]:
        count = len(part_A[part_A['label'] == label])
        pct = (count / len(part_A) * 100) if len(part_A) > 0 else 0
        logger.info(f"       {label}: {count:6} ({pct:5.1f}%)")
    
    logger.info(f"     Part B (20%):")
    for label in [0, 1]:
        count = len(part_B[part_B['label'] == label])
        pct = (count / len(part_B) * 100) if len(part_B) > 0 else 0
        logger.info(f"       {label}: {count:6} ({pct:5.1f}%)")
    
    logger.info("\n7.4 Source Distribution:")
    logger.info(f"     Part A:")
    for source in part_A['source'].unique():
        count = len(part_A[part_A['source'] == source])
        pct = (count / len(part_A) * 100)
        logger.info(f"       {source:15}: {count:6} ({pct:5.1f}%)")
    
    logger.info(f"     Part B:")
    for source in part_B['source'].unique():
        count = len(part_B[part_B['source'] == source])
        pct = (count / len(part_B) * 100)
        logger.info(f"       {source:15}: {count:6} ({pct:5.1f}%)")


def phase_8_save_metadata(part_A, part_B, base_path='.'):
    """Phase 8: Save metadata documentation."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 8: SAVE METADATA & DOCUMENTATION")
    logger.info("="*70)
    
    base = Path(base_path)
    splits_dir = base / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'single_80/20_stratified_split_on_unified_normalized_data',
        'part_A': {
            'size': len(part_A),
            'percentage': '80%',
            'purpose': 'Training split for Style + Knowledge branches',
            'files': [
                'dataset_partA.csv (Style branch)',
                'groundtruth_partA.csv (Knowledge - claim detection)',
                'train_partA.jsonl (Knowledge - verification)'
            ]
        },
        'part_B': {
            'size': len(part_B),
            'percentage': '20%',
            'purpose': 'Validation split for fusion logic testing',
            'files': ['part_B_validation.csv (unified, normalized)']
        },
        'label_convention': {
            '0': 'FAKE/FALSE/REFUTED',
            '1': 'TRUE/REAL/SUPPORTED'
        },
        'filtering_applied': {
            'groundtruth': 'Filtered Verdict==0 (uncertain) before sampling',
            'fever': 'Filtered label==NOT_ENOUGH_INFO before sampling',
            'dataset': 'Already filtered by data_extraction.py (no mostly-true)'
        },
        'sources': {
            'DATA': 'LIAR (train+valid) + Twitter + UoVictoria',
            'GROUNDTRUTH': 'ClaimBuster claims with verdict labels',
            'FEVER': 'Evidence-based claims (train.jsonl)'
        },
        'label_distribution': {
            'part_A': {
                '0': int((part_A['label'] == 0).sum()),
                '1': int((part_A['label'] == 1).sum())
            },
            'part_B': {
                '0': int((part_B['label'] == 0).sum()),
                '1': int((part_B['label'] == 1).sum())
            }
        },
        'verification': {
            'disjoint': 'No row overlap between Part A and B (stratified split)',
            'label_ratio_preserved': True
        }
    }
    
    metadata_path = splits_dir / 'part_B_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n8.1 Metadata saved to: {metadata_path.name}")
    logger.info(f"\n8.2 Metadata contents:")
    logger.info(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    return metadata_path


def main(base_path='.'):
    """Execute all 8 phases."""
    logger.info("\n\n")
    logger.info("█" * 70)
    logger.info("PART A & B DATASET SPLIT - HETEROGENEOUS SOURCES")
    logger.info("█" * 70)
    
    try:
        # Phase 1: Load inputs
        dataset_df, groundtruth_df, fever_df = phase_1_load_inputs(base_path)
        
        # Phase 2: Normalize and filter
        dataset_normalized, groundtruth_normalized, fever_normalized = phase_2_normalize_and_filter(
            dataset_df, groundtruth_df, fever_df
        )
        
        # Phase 3: Combine and statistics
        df_all = phase_3_combine_and_statistics(
            dataset_normalized, groundtruth_normalized, fever_normalized
        )
        
        if df_all.empty:
            logger.error("\n✗ FATAL: No valid data after combining sources!")
            return
        
        # Phase 4: Stratified split
        part_A, part_B = phase_4_stratified_split(df_all)
        
        # Phase 5: Extract Part A in original formats
        dataset_partA_path, groundtruth_partA_path, train_partA_path = phase_5_extract_part_a(part_A, base_path)
        
        # Phase 6: Save Part B unified
        part_B_path = phase_6_save_part_b(part_B, base_path)
        
        # Phase 7: Summary report
        phase_7_summary_report(part_A, part_B, dataset_partA_path, groundtruth_partA_path, train_partA_path, part_B_path)
        
        # Phase 8: Save metadata
        metadata_path = phase_8_save_metadata(part_A, part_B, base_path)
        
        logger.info("\n" + "="*70)
        logger.info("✓ SUCCESS: Dataset split complete!")
        logger.info("="*70)
        logger.info("\nNext steps (update script references):")
        logger.info("  1. style_branch/feature_extraction.py → use 'data/splits/dataset_partA.csv'")
        logger.info("  2. knowledge_branch/train_claim_detector.py → use 'knowledge_branch/splits/groundtruth_partA.csv'")
        logger.info("  3. knowledge_branch/evaluate_pipeline.py → use 'knowledge_branch/splits/train_partA.jsonl'")
        logger.info("  4. unified_main.ipynb → update Phase 0 order")
        logger.info("\n" + "="*70 + "\n")
        
    except Exception as e:
        logger.error(f"\n✗ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main(base_path='.')

