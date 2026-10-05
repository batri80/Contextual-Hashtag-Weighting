"""
Quick BERT Comparison at K=20
==============================
Compares Hashtag-Only vs BERT baseline at K=20
Uses existing BERT embeddings if available
"""

import numpy as np
import pickle
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_rel, wilcoxon
import time

print("="*80)
print("BERT vs Hashtag-Only Comparison at K=20")
print("="*80)

K = 20
N_RUNS = 5  # Multiple runs for statistical comparison

# ============================================================================
# Load Hashtag Features
# ============================================================================
print(f"\n[Step 1] Loading Hashtag-Only features...")

try:
    test_hashtag = sparse.load_npz('test_hashtag.npz')
    print(f"✓ Loaded: {test_hashtag.shape}")
except FileNotFoundError:
    print("✗ ERROR: test_hashtag.npz not found!")
    print("  Run Phase 4 first")
    exit(1)

# ============================================================================
# Cluster Hashtag Features (Multiple Runs)
# ============================================================================
print(f"\n[Step 2] Clustering Hashtag-Only (K={K}, {N_RUNS} runs)...")

hashtag_results = []

for run in range(N_RUNS):
    print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
    
    start = time.time()
    kmeans = KMeans(n_clusters=K, random_state=42 + run, n_init=10)
    labels = kmeans.fit_predict(test_hashtag)
    elapsed = time.time() - start
    
    sil = silhouette_score(test_hashtag, labels, sample_size=10000)
    
    hashtag_results.append({
        'silhouette': sil,
        'time': elapsed,
        'labels': labels
    })
    
    print(f"Sil={sil:.4f}, Time={elapsed:.2f}s")

hashtag_avg_sil = np.mean([r['silhouette'] for r in hashtag_results])
hashtag_std_sil = np.std([r['silhouette'] for r in hashtag_results])
hashtag_avg_time = np.mean([r['time'] for r in hashtag_results])

print(f"\n  Summary:")
print(f"    Avg Silhouette: {hashtag_avg_sil:.4f} ± {hashtag_std_sil:.4f}")
print(f"    Avg Time: {hashtag_avg_time:.2f}s")

# ============================================================================
# Load or Indicate BERT Features Needed
# ============================================================================
print(f"\n[Step 3] Loading BERT features...")

try:
    bert_features = np.load('test_bert_features.npy')
    print(f"✓ Loaded existing BERT embeddings: {bert_features.shape}")
    has_bert = True
    
except FileNotFoundError:
    print("✗ BERT features not found (test_bert_features.npy)")
    print("\n" + "="*80)
    print("YOU NEED TO GENERATE BERT EMBEDDINGS FIRST")
    print("="*80)
    print("\nOption 1: Run Phase 9 with K=20")
    print("  1. Edit phase9_bert.py")
    print("     Change line ~19 to: K_OPTIMAL = 20")
    print("  2. Run: python phase9_bert.py")
    print("     (Takes ~30-60 minutes)")
    print("\nOption 2: Generate just embeddings (faster)")
    print("  Use the script below to generate embeddings only")
    print("="*80)
    
    has_bert = False

# ============================================================================
# Cluster BERT Features (if available)
# ============================================================================
if has_bert:
    print(f"\n[Step 4] Clustering BERT (K={K}, {N_RUNS} runs)...")
    
    bert_results = []
    
    for run in range(N_RUNS):
        print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
        
        start = time.time()
        kmeans = KMeans(n_clusters=K, random_state=42 + run, n_init=10)
        labels = kmeans.fit_predict(bert_features)
        elapsed = time.time() - start
        
        sil = silhouette_score(bert_features, labels, sample_size=10000)
        
        bert_results.append({
            'silhouette': sil,
            'time': elapsed,
            'labels': labels
        })
        
        print(f"Sil={sil:.4f}, Time={elapsed:.2f}s")
    
    bert_avg_sil = np.mean([r['silhouette'] for r in bert_results])
    bert_std_sil = np.std([r['silhouette'] for r in bert_results])
    bert_avg_time = np.mean([r['time'] for r in bert_results])
    
    print(f"\n  Summary:")
    print(f"    Avg Silhouette: {bert_avg_sil:.4f} ± {bert_std_sil:.4f}")
    print(f"    Avg Time: {bert_avg_time:.2f}s")
    
    # ========================================================================
    # Statistical Comparison
    # ========================================================================
    print(f"\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)
    
    # Extract scores
    hashtag_scores = [r['silhouette'] for r in hashtag_results]
    bert_scores = [r['silhouette'] for r in bert_results]
    
    # Paired t-test
    t_stat, p_value_t = ttest_rel(hashtag_scores, bert_scores)
    
    # Wilcoxon test
    w_stat, p_value_w = wilcoxon(hashtag_scores, bert_scores)
    
    # Cohen's d effect size
    mean_diff = hashtag_avg_sil - bert_avg_sil
    pooled_std = np.sqrt((hashtag_std_sil**2 + bert_std_sil**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    print(f"\nHashtag-Only (Ours): {hashtag_avg_sil:.4f} ± {hashtag_std_sil:.4f}")
    print(f"BERT Baseline:       {bert_avg_sil:.4f} ± {bert_std_sil:.4f}")
    
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    sig = '***' if p_value_t < 0.001 else '**' if p_value_t < 0.01 else '*' if p_value_t < 0.05 else 'ns'
    print(f"  p-value:     {p_value_t:.6f} {sig}")
    
    print(f"\nWilcoxon test:")
    print(f"  W-statistic: {w_stat:.4f}")
    sig_w = '***' if p_value_w < 0.001 else '**' if p_value_w < 0.01 else '*' if p_value_w < 0.05 else 'ns'
    print(f"  p-value:     {p_value_w:.6f} {sig_w}")
    
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
    effect = 'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
    print(f"  Interpretation: {effect}")
    
    # ========================================================================
    # Results Summary
    # ========================================================================
    print(f"\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    improvement = ((hashtag_avg_sil - bert_avg_sil) / bert_avg_sil) * 100
    speedup = bert_avg_time / hashtag_avg_time
    
    print(f"\nPerformance:")
    print(f"  Improvement: {improvement:+.2f}%")
    
    print(f"\nEfficiency:")
    print(f"  Speedup: {speedup:.1f}x faster")
    
    print(f"\nStatistical Significance:")
    if p_value_t < 0.05:
        if hashtag_avg_sil > bert_avg_sil:
            print(f"  ✅ Hashtag-Only is SIGNIFICANTLY BETTER than BERT (p={p_value_t:.4f})")
        else:
            print(f"  ⚠️  BERT is SIGNIFICANTLY BETTER than Hashtag-Only (p={p_value_t:.4f})")
    else:
        print(f"  ✗ No significant difference (p={p_value_t:.4f})")
    
    # ========================================================================
    # Save Comparison Results
    # ========================================================================
    print(f"\n[Step 5] Saving comparison results...")
    
    comparison_results = {
        'K': K,
        'hashtag': {
            'avg_silhouette': hashtag_avg_sil,
            'std_silhouette': hashtag_std_sil,
            'avg_time': hashtag_avg_time,
            'scores': hashtag_scores,
            'results': hashtag_results
        },
        'bert': {
            'avg_silhouette': bert_avg_sil,
            'std_silhouette': bert_std_sil,
            'avg_time': bert_avg_time,
            'scores': bert_scores,
            'results': bert_results
        },
        'statistics': {
            't_statistic': t_stat,
            'p_value_t': p_value_t,
            'w_statistic': w_stat,
            'p_value_w': p_value_w,
            'cohens_d': cohens_d,
            'significant': p_value_t < 0.05
        }
    }
    
    with open('bert_comparison_k20.pkl', 'wb') as f:
        pickle.dump(comparison_results, f)
    
    print(f"✓ Saved: bert_comparison_k20.pkl")
    
    # ========================================================================
    # For Your Paper
    # ========================================================================
    print(f"\n" + "="*80)
    print("FOR YOUR PAPER")
    print("="*80)
    
    print(f"\nTable: Baseline Comparison at K=20")
    print(f"{'Method':<25} {'Silhouette':<15} {'Time (s)':<12} {'p-value'}")
    print(f"-"*80)
    print(f"{'Hashtag-Only (Ours)':<25} {hashtag_avg_sil:.4f} ± {hashtag_std_sil:.4f}    {hashtag_avg_time:>6.2f}      -")
    print(f"{'BERT Baseline':<25} {bert_avg_sil:.4f} ± {bert_std_sil:.4f}    {bert_avg_time:>6.2f}      {p_value_t:.4f}{sig}")
    
    print(f"\nKey Statement for Paper:")
    if p_value_t < 0.05 and hashtag_avg_sil > bert_avg_sil:
        print(f'  "Our contextual hashtag weighting method (Silhouette={hashtag_avg_sil:.4f})')
        print(f'   significantly outperforms the BERT baseline (Silhouette={bert_avg_sil:.4f},')
        print(f'   p={p_value_t:.4f}) while being {speedup:.1f}x more computationally efficient."')
    elif p_value_t < 0.05:
        print(f'  "BERT baseline achieves higher Silhouette score ({bert_avg_sil:.4f} vs')
        print(f'   {hashtag_avg_sil:.4f}, p={p_value_t:.4f}), though at {speedup:.1f}x higher')
        print(f'   computational cost."')
    else:
        print(f'  "Our method achieves comparable performance to BERT (Silhouette={hashtag_avg_sil:.4f}')
        print(f'   vs {bert_avg_sil:.4f}, p={p_value_t:.4f}) while being {speedup:.1f}x faster."')

print(f"\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
