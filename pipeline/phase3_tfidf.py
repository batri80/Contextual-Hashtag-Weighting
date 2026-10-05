"""
Phase 3: TF-IDF Text Features
==============================
- Compute TF-IDF vectors for tweet text
- Save vectorizer and matrices
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

print("="*80)
print("PHASE 3: TF-IDF TEXT FEATURES")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[Loading Data]")
train_df = pd.read_pickle('train_with_weights.pkl')
val_df = pd.read_pickle('val_with_weights.pkl')
test_df = pd.read_pickle('test_with_weights.pkl')

print(f"✓ Loaded: {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")

# ============================================================================
# Compute TF-IDF
# ============================================================================
print("\n[Computing TF-IDF]")

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,  # Top 5000 words
    min_df=5,           # Word must appear in at least 5 documents
    max_df=0.8,         # Word must appear in at most 80% of documents
    stop_words='english',
    ngram_range=(1, 2)  # Unigrams and bigrams
)

# Fit on train
print("  Fitting on train set...")
train_tfidf = tfidf.fit_transform(train_df['clean_text'])

print(f"✓ TF-IDF computed")
print(f"  Vocabulary size: {len(tfidf.vocabulary_):,}")
print(f"  Train shape: {train_tfidf.shape}")
print(f"  Sparsity: {100 * train_tfidf.nnz / (train_tfidf.shape[0] * train_tfidf.shape[1]):.2f}%")

# Transform val and test
print("\n  Transforming val and test sets...")
val_tfidf = tfidf.transform(val_df['clean_text'])
test_tfidf = tfidf.transform(test_df['clean_text'])

print(f"  Val shape: {val_tfidf.shape}")
print(f"  Test shape: {test_tfidf.shape}")

# ============================================================================
# Save TF-IDF
# ============================================================================
print("\n[Saving TF-IDF]")

# Save vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Save matrices
sparse.save_npz('train_tfidf.npz', train_tfidf)
sparse.save_npz('val_tfidf.npz', val_tfidf)
sparse.save_npz('test_tfidf.npz', test_tfidf)

print(f"✓ Saved:")
print(f"  - tfidf_vectorizer.pkl")
print(f"  - train/val/test_tfidf.npz")

# Show top features
print("\n[Top 20 TF-IDF Features]")
feature_names = tfidf.get_feature_names_out()
vocab_items = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])[:20]
for word, idx in vocab_items:
    print(f"  {word}")

print("\n" + "="*80)
print("PHASE 3 COMPLETE!")
print("="*80)
print("\n✓ Ready for Phase 4: Feature combination\n")
