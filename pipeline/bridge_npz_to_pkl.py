"""
Bridge Script: Convert Sparse NPZ to Dense PKL
==============================================
Converts your Phase 4 sparse matrices (.npz) to dense arrays (.pkl)
for compatibility with enhancement scripts.

Run this AFTER Phase 4 and BEFORE enhanced_phase6_flexible.py
"""

import numpy as np
from scipy import sparse
import pickle

print("="*80)
print("BRIDGE SCRIPT: NPZ → PKL CONVERSION")
print("="*80)

# ============================================================================
# Step 1: Load Sparse Matrices
# ============================================================================
print("\n[Step 1] Loading sparse matrices from Phase 4...")

try:
    test_hashtag_sparse = sparse.load_npz('test_hashtag.npz')
    test_tfidf_sparse = sparse.load_npz('test_tfidf.npz')
    test_combined_sparse = sparse.load_npz('test_combined.npz')
    
    print(f"✓ Loaded sparse matrices:")
    print(f"  test_hashtag.npz:  {test_hashtag_sparse.shape}")
    print(f"  test_tfidf.npz:    {test_tfidf_sparse.shape}")
    print(f"  test_combined.npz: {test_combined_sparse.shape}")
    
except FileNotFoundError as e:
    print(f"\n✗ ERROR: Could not find sparse matrix files!")
    print(f"  Make sure you've run Phase 4 first.")
    print(f"  Expected files: test_hashtag.npz, test_tfidf.npz, test_combined.npz")
    exit(1)

# ============================================================================
# Step 2: Convert to Dense Arrays
# ============================================================================
print("\n[Step 2] Converting to dense arrays...")

# Convert sparse to dense
test_hashtag_dense = test_hashtag_sparse.toarray()
test_tfidf_dense = test_tfidf_sparse.toarray()
test_combined_dense = test_combined_sparse.toarray()

print(f"✓ Converted to dense:")
print(f"  Hashtag:  {test_hashtag_dense.shape} ({test_hashtag_dense.nbytes / 1024 / 1024:.1f} MB)")
print(f"  TF-IDF:   {test_tfidf_dense.shape} ({test_tfidf_dense.nbytes / 1024 / 1024:.1f} MB)")
print(f"  Combined: {test_combined_dense.shape} ({test_combined_dense.nbytes / 1024 / 1024:.1f} MB)")

# ============================================================================
# Step 3: Save as Pickle Files
# ============================================================================
print("\n[Step 3] Saving as pickle files...")

with open('hashtag_embeddings.pkl', 'wb') as f:
    pickle.dump(test_hashtag_dense, f)
print(f"  ✓ Saved: hashtag_embeddings.pkl")

with open('tfidf_embeddings.pkl', 'wb') as f:
    pickle.dump(test_tfidf_dense, f)
print(f"  ✓ Saved: tfidf_embeddings.pkl")

with open('combined_embeddings.pkl', 'wb') as f:
    pickle.dump(test_combined_dense, f)
print(f"  ✓ Saved: combined_embeddings.pkl")

# ============================================================================
# Step 4: Verify Conversion
# ============================================================================
print("\n[Step 4] Verifying conversion...")

with open('hashtag_embeddings.pkl', 'rb') as f:
    verify_hashtag = pickle.load(f)

with open('tfidf_embeddings.pkl', 'rb') as f:
    verify_tfidf = pickle.load(f)

with open('combined_embeddings.pkl', 'rb') as f:
    verify_combined = pickle.load(f)

print(f"✓ Verification:")
print(f"  hashtag_embeddings.pkl:  {type(verify_hashtag)} shape {verify_hashtag.shape}")
print(f"  tfidf_embeddings.pkl:    {type(verify_tfidf)} shape {verify_tfidf.shape}")
print(f"  combined_embeddings.pkl: {type(verify_combined)} shape {verify_combined.shape}")

# Check they're numpy arrays
assert isinstance(verify_hashtag, np.ndarray), "Hashtag should be numpy array!"
assert isinstance(verify_tfidf, np.ndarray), "TF-IDF should be numpy array!"
assert isinstance(verify_combined, np.ndarray), "Combined should be numpy array!"

print(f"\n✓ All checks passed!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("CONVERSION COMPLETE!")
print("="*80)

print("\nFiles created:")
print("  ✓ hashtag_embeddings.pkl  (dense numpy array)")
print("  ✓ tfidf_embeddings.pkl    (dense numpy array)")
print("  ✓ combined_embeddings.pkl (dense numpy array)")

print("\n✅ You can now run the enhancement scripts:")
print("   python enhanced_phase6_flexible.py")
print("   python qualitative_analysis.py")
print("   python visualization_suite.py")
print("   python ablation_study.py")

print("\n💡 Note: Your original sparse .npz files are still intact.")
print("   The .pkl files are just copies in dense format for the enhancements.")

print("="*80)
