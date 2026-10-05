"""
QUICK FIX: Rebuild Vocabulary with Contiguous Indices
======================================================
The vocabulary was created with non-contiguous indices.
This script rebuilds it with proper 0, 1, 2, ... indices.

Run this BEFORE Phase 4.
"""

import pickle
from collections import Counter

print("="*80)
print("QUICK FIX: Rebuilding Vocabulary with Contiguous Indices")
print("="*80)

# Load the existing vocabulary data
print("\n[1/3] Loading existing vocabulary...")
with open('hashtag_vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)

old_vocab = vocab_data['vocab']
hashtag_freq = vocab_data['freq']
hashtag_prob = vocab_data['prob']

print(f"✓ Loaded vocabulary: {len(old_vocab):,} hashtags")
print(f"  Old index range: {min(old_vocab.values())} to {max(old_vocab.values())}")

# Rebuild with contiguous indices
print("\n[2/3] Rebuilding with contiguous indices...")

# Get hashtags that are in vocab (already filtered by frequency)
filtered_hashtags = [(h, hashtag_freq[h]) for h in old_vocab.keys()]

# Create NEW vocabulary with contiguous indices 0, 1, 2, ...
new_vocab = {h: i for i, (h, f) in enumerate(filtered_hashtags)}

print(f"✓ Rebuilt vocabulary: {len(new_vocab):,} hashtags")
print(f"  New index range: {min(new_vocab.values())} to {max(new_vocab.values())}")

# Verify
assert len(new_vocab) == len(old_vocab), "Vocabulary size mismatch!"
assert max(new_vocab.values()) == len(new_vocab) - 1, "Indices not contiguous!"
assert set(new_vocab.keys()) == set(old_vocab.keys()), "Hashtags don't match!"

print(f"✓ Verification passed!")

# Save the fixed vocabulary
print("\n[3/3] Saving fixed vocabulary...")
with open('hashtag_vocab.pkl', 'wb') as f:
    pickle.dump({
        'vocab': new_vocab,
        'freq': hashtag_freq,
        'prob': hashtag_prob
    }, f)

print("✓ Saved: hashtag_vocab.pkl")

print("\n" + "="*80)
print("FIX COMPLETE!")
print("="*80)
print("The vocabulary now has contiguous indices [0, 1, 2, ..., n-1]")
print("\nYou can now run Phase 4:")
print("  python phase4_combine_features.py")
print("="*80)
