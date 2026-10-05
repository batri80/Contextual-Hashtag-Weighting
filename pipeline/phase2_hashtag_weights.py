"""
Phase 2: Contextual Hashtag Weights (FULLY OPTIMIZED)
======================================================
ALL bottlenecks fixed:
- Fast temporal computation (O(n) sliding window)
- Fast weight computation (binary search for temporal lookups)
- Fixed progress counter
- Safe type conversion
"""

import pandas as pd
import numpy as np
import pickle
import time
import bisect
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 2: CONTEXTUAL HASHTAG WEIGHTS (FULLY OPTIMIZED)")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[Loading Data]")
train_df = pd.read_pickle('train_data.pkl')
val_df = pd.read_pickle('val_data.pkl')
test_df = pd.read_pickle('test_data.pkl')

print(f"✓ Loaded: {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")

# ============================================================================
# Step 2.1: Build Hashtag Vocabulary and Statistics
# ============================================================================
print("\n[Step 2.1] Building hashtag vocabulary...")

def build_hashtag_vocab(df, min_freq=10):
    """Build hashtag vocabulary and compute statistics"""
    
    # Collect all hashtags
    all_hashtags = []
    for hashtags in df['hashtags']:
        all_hashtags.extend(hashtags)
    
    # Count frequencies
    hashtag_freq = Counter(all_hashtags)
    
    # Filter: Keep hashtags with at least min_freq occurrences
    vocab = {h: i for i, (h, f) in enumerate(hashtag_freq.items()) if f >= min_freq}
    
    # Compute P(h_i)
    total_hashtag_count = sum(hashtag_freq.values())
    hashtag_prob = {h: f / total_hashtag_count for h, f in hashtag_freq.items() if h in vocab}
    
    return vocab, hashtag_freq, hashtag_prob

vocab, hashtag_freq, hashtag_prob = build_hashtag_vocab(train_df, min_freq=10)

print(f"✓ Vocabulary built")
print(f"  Total unique hashtags: {len(hashtag_freq):,}")
print(f"  Vocabulary size (freq≥10): {len(vocab):,}")

# Save vocabulary
with open('hashtag_vocab.pkl', 'wb') as f:
    pickle.dump({
        'vocab': vocab,
        'freq': hashtag_freq,
        'prob': hashtag_prob
    }, f)

print(f"\n  Top 10 hashtags:")
for h, f in list(hashtag_freq.most_common(10)):
    if h in vocab:
        print(f"    #{h}: {f:,}")

# ============================================================================
# Step 2.2: Compute PMI Matrix
# ============================================================================
print("\n[Step 2.2] Computing PMI matrix...")

def compute_pmi_matrix(df, vocab, hashtag_prob):
    """Compute Pointwise Mutual Information for all hashtag pairs"""
    
    # Co-occurrence matrix
    cooc = defaultdict(int)
    
    for hashtags in df['hashtags']:
        # Filter to vocab
        hashtags_in_vocab = [h for h in hashtags if h in vocab]
        
        # Count co-occurrences
        for i, h_i in enumerate(hashtags_in_vocab):
            for h_j in hashtags_in_vocab[i+1:]:
                pair = tuple(sorted([h_i, h_j]))
                cooc[pair] += 1
    
    print(f"  Co-occurring pairs: {len(cooc):,}")
    
    # Compute PMI
    total_posts = len(df)
    pmi_matrix = {}
    
    for (h_i, h_j), count_ij in cooc.items():
        p_ij = count_ij / total_posts
        p_i = hashtag_prob.get(h_i, 1e-10)
        p_j = hashtag_prob.get(h_j, 1e-10)
        
        pmi = np.log(p_ij / (p_i * p_j))
        pmi_matrix[(h_i, h_j)] = pmi
        pmi_matrix[(h_j, h_i)] = pmi  # Symmetric
    
    # Normalize PMI to [0, 1]
    pmi_values = list(pmi_matrix.values())
    pmi_min, pmi_max = min(pmi_values), max(pmi_values)
    
    pmi_normalized = {}
    for pair, pmi in pmi_matrix.items():
        pmi_norm = (pmi - pmi_min) / (pmi_max - pmi_min) if pmi_max > pmi_min else 0
        pmi_normalized[pair] = pmi_norm
    
    print(f"  PMI range: [{pmi_min:.3f}, {pmi_max:.3f}]")
    
    return pmi_normalized, pmi_min, pmi_max

pmi_matrix, pmi_min, pmi_max = compute_pmi_matrix(train_df, vocab, hashtag_prob)

print(f"✓ PMI matrix computed")

# Save PMI matrix
with open('pmi_matrix.pkl', 'wb') as f:
    pickle.dump({
        'pmi': pmi_matrix,
        'pmi_min': pmi_min,
        'pmi_max': pmi_max
    }, f)

# ============================================================================
# Step 2.3: Compute Hashtag Embeddings
# ============================================================================
print("\n[Step 2.3] Computing hashtag embeddings...")

print("  Loading SentenceTransformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_hashtag_embeddings(vocab, model, batch_size=128):
    """Compute embeddings for all hashtags in vocabulary"""
    
    hashtag_embeddings = {}
    hashtag_list = list(vocab.keys())
    
    for i in range(0, len(hashtag_list), batch_size):
        batch = hashtag_list[i:i+batch_size]
        
        # Convert hashtags to readable text
        batch_text = [h.replace('_', ' ') for h in batch]
        
        # Encode
        embeddings = model.encode(batch_text, show_progress_bar=False)
        
        # Store
        for h, emb in zip(batch, embeddings):
            hashtag_embeddings[h] = emb
    
    return hashtag_embeddings

hashtag_embeddings = compute_hashtag_embeddings(vocab, embedding_model)

print(f"✓ Hashtag embeddings computed")
print(f"  Vocabulary size: {len(hashtag_embeddings):,}")
print(f"  Embedding dim: {len(next(iter(hashtag_embeddings.values())))}")

# Save embeddings
with open('hashtag_embeddings.pkl', 'wb') as f:
    pickle.dump(hashtag_embeddings, f)

# ============================================================================
# Step 2.4: Compute Text Embeddings
# ============================================================================
print("\n[Step 2.4] Computing text embeddings...")

def compute_text_embeddings(df, model, batch_size=128):
    """Compute embeddings for all tweet texts"""
    
    texts = df['clean_text'].tolist()
    text_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        text_embeddings.append(embeddings)
        
        if (i // batch_size + 1) % 50 == 0:
            print(f"    Processed {i+len(batch):,}/{len(texts):,}")
    
    text_embeddings = np.vstack(text_embeddings)
    return text_embeddings

print("  Train set:")
train_text_embeddings = compute_text_embeddings(train_df, embedding_model)
train_df['text_embedding'] = list(train_text_embeddings)

print("  Val set:")
val_text_embeddings = compute_text_embeddings(val_df, embedding_model)
val_df['text_embedding'] = list(val_text_embeddings)

print("  Test set:")
test_text_embeddings = compute_text_embeddings(test_df, embedding_model)
test_df['text_embedding'] = list(test_text_embeddings)

print(f"✓ Text embeddings computed")

# ============================================================================
# Step 2.5: Compute Temporal Activity Data (OPTIMIZED!)
# ============================================================================
print("\n[Step 2.5] Computing temporal activity data (optimized)...")

def compute_temporal_data_optimized(df, vocab, T_window=90):
    """
    OPTIMIZED: Compute temporal activity data for all hashtags
    Uses O(n) sliding window instead of O(n²)
    """
    
    T_window_sec = T_window * 24 * 3600
    
    # Track hashtag history - KEEP SORTED for binary search later
    hashtag_history = defaultdict(list)
    
    print("  Building hashtag history...")
    for idx, row in df.iterrows():
        timestamp = row['created_at'].timestamp()
        for h in row['hashtags']:
            if h in vocab:
                hashtag_history[h].append(timestamp)
    
    # Sort all histories for binary search
    print("  Sorting histories for fast lookup...")
    for h in hashtag_history:
        hashtag_history[h].sort()
    
    print(f"  Hashtags with history: {len(hashtag_history):,}")
    
    # Compute max frequency using SLIDING WINDOW
    print("  Computing max frequencies (optimized)...")
    hashtag_max_freq = {}
    
    processed = 0
    total = len(vocab)
    
    for h in vocab:
        if h not in hashtag_history:
            hashtag_max_freq[h] = 1
            processed += 1
            continue
        
        times = hashtag_history[h]  # Already sorted
        n = len(times)
        
        if n == 0:
            hashtag_max_freq[h] = 1
            processed += 1
            continue
        
        # OPTIMIZED SLIDING WINDOW
        max_count = 0
        left = 0
        
        for right in range(n):
            t_end = times[right]
            t_start = t_end - T_window_sec
            
            # Shrink window from left
            while left < right and times[left] < t_start:
                left += 1
            
            # Count in window
            count = right - left + 1
            max_count = max(max_count, count)
        
        hashtag_max_freq[h] = max(max_count, 1)
        
        processed += 1
        if processed % 500 == 0:
            print(f"    Processed {processed:,}/{total:,} hashtags")
    
    print(f"  ✓ Max frequencies computed for {len(hashtag_max_freq):,} hashtags")
    
    # Return parameters
    tau_dormant_sec = 30 * 24 * 3600
    tau_decay_sec = 7 * 24 * 3600
    delta_T_sec = 7 * 24 * 3600
    
    return hashtag_history, hashtag_max_freq, tau_dormant_sec, tau_decay_sec, delta_T_sec

hashtag_history, hashtag_max_freq, tau_dormant_sec, tau_decay_sec, delta_T_sec = \
    compute_temporal_data_optimized(train_df, vocab)

print(f"✓ Temporal data computed")

# Save temporal data
with open('temporal_data.pkl', 'wb') as f:
    pickle.dump({
        'history': dict(hashtag_history),
        'max_freq': hashtag_max_freq,
        'tau_dormant': tau_dormant_sec,
        'tau_decay': tau_decay_sec,
        'delta_T': delta_T_sec
    }, f)

# ============================================================================
# Step 2.6: Compute Contextual Weights (OPTIMIZED!)
# ============================================================================
print("\n[Step 2.6] Computing contextual weights (optimized)...")

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def safe_numeric_convert(value, default=0):
    """Safely convert value to numeric, handling strings and NaN"""
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def compute_contextual_weights_fast(row, vocab, pmi_matrix, hashtag_embeddings, 
                                     hashtag_history, hashtag_max_freq,
                                     alpha1=0.33, alpha2=0.33, alpha3=0.33,
                                     beta1=0.5, beta2=0.5,
                                     tau_dormant_sec=30*24*3600, tau_decay_sec=7*24*3600,
                                     delta_T_sec=7*24*3600):
    """
    OPTIMIZED: Compute contextual weights with fast temporal lookups
    Uses binary search instead of list filtering
    """
    
    hashtags_in_post = [h for h in row['hashtags'] if h in vocab]
    
    if len(hashtags_in_post) == 0:
        return {}
    
    text_embedding = row['text_embedding']
    timestamp = row['created_at'].timestamp()
    
    # Post activity (temperature analog) - SAFE CONVERSION
    retweet_count = safe_numeric_convert(row.get('retweet_count', 0))
    likes_count = safe_numeric_convert(row.get('likes', 0))
    T_p = np.log(1 + retweet_count + likes_count)
    Gamma = T_p / 1.0
    
    energies = {}
    
    for h_i in hashtags_in_post:
        
        # Component 1: Hashtag Interaction Energy
        E_hashtag = 0.0
        
        if len(hashtags_in_post) > 1:
            V_sum = 0.0
            count = 0
            
            for h_j in hashtags_in_post:
                if h_i == h_j:
                    continue
                
                V_pmi = pmi_matrix.get((h_i, h_j), 0.0)
                
                if h_i in hashtag_embeddings and h_j in hashtag_embeddings:
                    sim = cosine_similarity(hashtag_embeddings[h_i], hashtag_embeddings[h_j])
                    V_sem = (sim + 1) / 2
                else:
                    V_sem = 0.0
                
                V_ij = beta1 * V_pmi + beta2 * V_sem
                V_sum += V_ij
                count += 1
            
            if count > 0:
                V_avg = V_sum / count
                E_hashtag = -alpha1 * V_avg
        
        # Component 2: Text-Hashtag Coupling Energy
        E_text = 0.0
        
        if h_i in hashtag_embeddings:
            sim = cosine_similarity(hashtag_embeddings[h_i], text_embedding)
            sim_norm = (sim + 1) / 2
            E_text = -alpha2 * sim_norm
        
        # Component 3: Temporal Activity Energy (OPTIMIZED with BINARY SEARCH)
        E_temporal = 0.0
        
        if h_i in hashtag_history and len(hashtag_history[h_i]) > 0:
            times = hashtag_history[h_i]  # Already sorted
            
            # Binary search for last occurrence before current timestamp
            idx = bisect.bisect_left(times, timestamp)
            
            if idx > 0:
                t_last = times[idx - 1]
                delta_t = timestamp - t_last
                
                # LIFECYCLE EQUIVALENCE CHECK
                if delta_t > tau_dormant_sec:
                    A_i = 0.0
                else:
                    # Count recent occurrences using binary search
                    recent_start_time = timestamp - delta_T_sec
                    start_idx = bisect.bisect_left(times, recent_start_time)
                    f_recent = idx - start_idx
                    
                    f_max = hashtag_max_freq.get(h_i, 1)
                    rho_i = f_recent / f_max
                    
                    decay_factor = np.exp(-delta_t / tau_decay_sec)
                    A_i = rho_i * decay_factor
                
                E_temporal = -alpha3 * A_i * Gamma
        
        # Total energy
        E_total = E_hashtag + E_text + E_temporal
        energies[h_i] = E_total
    
    # Boltzmann distribution
    exp_neg_E = {h: np.exp(-E) for h, E in energies.items()}
    Z = sum(exp_neg_E.values())
    
    weights = {h: exp_val / Z for h, exp_val in exp_neg_E.items()}
    
    return weights

def compute_all_weights_fast(df, vocab, pmi_matrix, hashtag_embeddings,
                              hashtag_history, hashtag_max_freq,
                              alpha1=0.33, alpha2=0.33, alpha3=0.33):
    """Compute contextual weights for all tweets - FAST VERSION"""
    
    print(f"  Processing {len(df):,} tweets...")
    start_time = time.time()
    
    all_weights = []
    
    # FIXED: Use enumerate to get sequential counter
    for counter, (idx, row) in enumerate(df.iterrows(), 1):
        weights = compute_contextual_weights_fast(
            row, vocab, pmi_matrix, hashtag_embeddings,
            hashtag_history, hashtag_max_freq,
            alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
            tau_dormant_sec=tau_dormant_sec, tau_decay_sec=tau_decay_sec,
            delta_T_sec=delta_T_sec
        )
        all_weights.append(weights)
        
        if counter % 1000 == 0:
            elapsed = time.time() - start_time
            rate = counter / elapsed
            remaining = (len(df) - counter) / rate
            print(f"    [{counter:>7,}/{len(df):,}] {rate:>6.1f} tweets/sec | "
                  f"Elapsed: {elapsed/60:>5.1f}m | Remaining: {remaining/60:>5.1f}m")
    
    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.2f}s ({len(df)/elapsed:.1f} tweets/sec)")
    
    return all_weights

# Compute weights for all splits
print("  Train set:")
train_weights = compute_all_weights_fast(train_df, vocab, pmi_matrix, hashtag_embeddings,
                                         hashtag_history, hashtag_max_freq)
train_df['hashtag_weights'] = train_weights

print("  Val set:")
val_weights = compute_all_weights_fast(val_df, vocab, pmi_matrix, hashtag_embeddings,
                                       hashtag_history, hashtag_max_freq)
val_df['hashtag_weights'] = val_weights

print("  Test set:")
test_weights = compute_all_weights_fast(test_df, vocab, pmi_matrix, hashtag_embeddings,
                                        hashtag_history, hashtag_max_freq)
test_df['hashtag_weights'] = test_weights

print(f"✓ Contextual weights computed")

# Save
train_df.to_pickle('train_with_weights.pkl')
val_df.to_pickle('val_with_weights.pkl')
test_df.to_pickle('test_with_weights.pkl')

print("\n" + "="*80)
print("PHASE 2 COMPLETE!")
print("="*80)
print("Files saved:")
print("  - hashtag_vocab.pkl")
print("  - pmi_matrix.pkl")
print("  - hashtag_embeddings.pkl")
print("  - temporal_data.pkl")
print("  - train/val/test_with_weights.pkl")
print("="*80)
print("\n✓ Ready for Phase 3: TF-IDF computation\n")