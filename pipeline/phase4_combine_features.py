"""
Phase 4: Feature Combination
=============================
- Create hashtag feature matrices
- Create combined (hashtag + TF-IDF) features
- Save all feature representations
"""

import pandas as pd
import numpy as np
import pickle
from scipy import sparse
from sklearn.preprocessing import normalize

print("="*80)
print("PHASE 4: FEATURE COMBINATION")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[Loading Data]")

train_df = pd.read_pickle('train_with_weights.pkl')
val_df = pd.read_pickle('val_with_weights.pkl')
test_df = pd.read_pickle('test_with_weights.pkl')

with open('hashtag_vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    vocab = vocab_data['vocab']

train_tfidf = sparse.load_npz('train_tfidf.npz')
val_tfidf = sparse.load_npz('val_tfidf.npz')
test_tfidf = sparse.load_npz('test_tfidf.npz')

print(f"✓ Loaded all data")

# ============================================================================
# Step 4.1: Create Hashtag Feature Matrices
# ============================================================================
print("\n[Step 4.1] Creating hashtag feature matrices...")

def create_hashtag_features(df, vocab):
    """Create hashtag feature matrix from weights"""
    
    n_tweets = len(df)
    n_hashtags = len(vocab)
    
    # Create sparse matrix
    rows, cols, data = [], [], []
    
    for idx, weights in enumerate(df['hashtag_weights']):
        for hashtag, weight in weights.items():
            if hashtag in vocab:
                col_idx = vocab[hashtag]
                rows.append(idx)
                cols.append(col_idx)
                data.append(weight)
    
    hashtag_matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_tweets, n_hashtags)
    )
    
    return hashtag_matrix

train_hashtag = create_hashtag_features(train_df, vocab)
val_hashtag = create_hashtag_features(val_df, vocab)
test_hashtag = create_hashtag_features(test_df, vocab)

print(f"✓ Hashtag features created")
print(f"  Train shape: {train_hashtag.shape}")
print(f"  Val shape: {val_hashtag.shape}")
print(f"  Test shape: {test_hashtag.shape}")
print(f"  Sparsity: {100 * test_hashtag.nnz / (test_hashtag.shape[0] * test_hashtag.shape[1]):.2f}%")

# ============================================================================
# Step 4.2: Create Combined Features
# ============================================================================
print("\n[Step 4.2] Creating combined features...")

def create_combined_features(hashtag_matrix, tfidf_matrix, hashtag_weight=0.5):
    """Combine hashtag and TF-IDF features"""
    
    # Normalize matrices
    hashtag_norm = normalize(hashtag_matrix, norm='l2')
    tfidf_norm = normalize(tfidf_matrix, norm='l2')
    
    # Weight and concatenate
    hashtag_weighted = hashtag_norm * hashtag_weight
    tfidf_weighted = tfidf_norm * (1 - hashtag_weight)
    
    combined = sparse.hstack([hashtag_weighted, tfidf_weighted])
    
    return combined

# Create combined features (50% hashtag, 50% text)
train_combined = create_combined_features(train_hashtag, train_tfidf, 0.5)
val_combined = create_combined_features(val_hashtag, val_tfidf, 0.5)
test_combined = create_combined_features(test_hashtag, test_tfidf, 0.5)

print(f"✓ Combined features created")
print(f"  Train shape: {train_combined.shape}")
print(f"  Val shape: {val_combined.shape}")
print(f"  Test shape: {test_combined.shape}")

# ============================================================================
# Step 4.3: Save All Features
# ============================================================================
print("\n[Step 4.3] Saving feature matrices...")

# Save hashtag features
sparse.save_npz('train_hashtag.npz', train_hashtag)
sparse.save_npz('val_hashtag.npz', val_hashtag)
sparse.save_npz('test_hashtag.npz', test_hashtag)

# Save combined features
sparse.save_npz('train_combined.npz', train_combined)
sparse.save_npz('val_combined.npz', val_combined)
sparse.save_npz('test_combined.npz', test_combined)

print(f"✓ All features saved")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("FEATURE SUMMARY")
print("="*80)
print(f"{'Feature Type':<20} {'Train Shape':<20} {'Val Shape':<20} {'Test Shape':<20}")
print("-"*80)
print(f"{'Hashtag-Only':<20} {str(train_hashtag.shape):<20} {str(val_hashtag.shape):<20} {str(test_hashtag.shape):<20}")
print(f"{'TF-IDF-Only':<20} {str(train_tfidf.shape):<20} {str(val_tfidf.shape):<20} {str(test_tfidf.shape):<20}")
print(f"{'Combined':<20} {str(train_combined.shape):<20} {str(val_combined.shape):<20} {str(test_combined.shape):<20}")
print("="*80)

print("\n" + "="*80)
print("PHASE 4 COMPLETE!")
print("="*80)
print("Files saved:")
print("  - train/val/test_hashtag.npz")
print("  - train/val/test_combined.npz")
print("="*80)
print("\n✓ Ready for Phase 5: Optimal K identification\n")
