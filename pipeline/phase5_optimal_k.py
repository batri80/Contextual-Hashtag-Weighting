"""
Phase 5: Optimal K Identification
==================================
- Test K values from 2 to 20
- Compute multiple metrics: Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz
- Generate visualizations
- Recommend optimal K
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import time

print("="*80)
print("PHASE 5: OPTIMAL K IDENTIFICATION")
print("="*80)

# ============================================================================
# Load Features
# ============================================================================
print("\n[Loading Features]")

test_hashtag = sparse.load_npz('test_hashtag.npz')
test_tfidf = sparse.load_npz('test_tfidf.npz')
test_combined = sparse.load_npz('test_combined.npz')

print(f"✓ Loaded feature matrices")

# ============================================================================
# Define K Range
# ============================================================================
k_range = range(2, 21)  # Test K from 2 to 20

print(f"\nTesting K values: {list(k_range)}")

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_k_range(features, k_range, method_name="Features"):
    """Evaluate clustering quality for different values of K"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*80}")
    
    metrics = {
        'k_values': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'time': []
    }
    
    for k in k_range:
        print(f"\nK={k:2d}: ", end="", flush=True)
        start_time = time.time()
        
        # Run K-means
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42
        )
        
        labels = kmeans.fit_predict(features)
        elapsed = time.time() - start_time
        
        # Compute metrics
        inertia = kmeans.inertia_
        
        # Use sampling for large datasets
        sample_size = min(10000, len(labels))
        silhouette = silhouette_score(features, labels, sample_size=sample_size)
        
        # Convert to dense for these metrics
        features_dense = features.toarray() if sparse.issparse(features) else features
        davies_bouldin = davies_bouldin_score(features_dense, labels)
        calinski = calinski_harabasz_score(features_dense, labels)
        
        # Store metrics
        metrics['k_values'].append(k)
        metrics['inertia'].append(inertia)
        metrics['silhouette'].append(silhouette)
        metrics['davies_bouldin'].append(davies_bouldin)
        metrics['calinski_harabasz'].append(calinski)
        metrics['time'].append(elapsed)
        
        print(f"Sil={silhouette:.4f}, DB={davies_bouldin:.4f}, CH={calinski:.1f}, Time={elapsed:.1f}s")
    
    return metrics

# ============================================================================
# Evaluate All Three Feature Types
# ============================================================================

print("\n" + "="*80)
print("STARTING OPTIMAL K EVALUATION")
print("="*80)

# 1. Hashtag-only
metrics_hashtag = evaluate_k_range(test_hashtag, k_range, "Hashtag-Only (Contextual)")

# 2. TF-IDF-only  
metrics_tfidf = evaluate_k_range(test_tfidf, k_range, "TF-IDF-Only")

# 3. Combined
metrics_combined = evaluate_k_range(test_combined, k_range, "Combined (Hashtag + TF-IDF)")

# ============================================================================
# Plot Results
# ============================================================================
print("\n[Generating Visualizations]")

def plot_optimal_k_metrics(metrics, method_name):
    """Plot all metrics for optimal K selection"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Optimal K Analysis - {method_name}', fontsize=16, fontweight='bold')
    
    k_values = metrics['k_values']
    
    # 1. Elbow method (Inertia)
    ax1 = axes[0, 0]
    ax1.plot(k_values, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Silhouette Score
    ax2 = axes[0, 1]
    ax2.plot(k_values, metrics['silhouette'], 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    optimal_k_sil = k_values[np.argmax(metrics['silhouette'])]
    ax2.axhline(y=max(metrics['silhouette']), color='r', linestyle='--', alpha=0.5)
    ax2.text(optimal_k_sil, max(metrics['silhouette']), 
             f'  K={optimal_k_sil}', fontsize=10, color='red', fontweight='bold')
    
    # 3. Davies-Bouldin Index
    ax3 = axes[1, 0]
    ax3.plot(k_values, metrics['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax3.set_ylabel('Davies-Bouldin Index', fontsize=12)
    ax3.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    optimal_k_db = k_values[np.argmin(metrics['davies_bouldin'])]
    ax3.axhline(y=min(metrics['davies_bouldin']), color='r', linestyle='--', alpha=0.5)
    ax3.text(optimal_k_db, min(metrics['davies_bouldin']), 
             f'  K={optimal_k_db}', fontsize=10, color='red', fontweight='bold')
    
    # 4. Calinski-Harabasz Index
    ax4 = axes[1, 1]
    ax4.plot(k_values, metrics['calinski_harabasz'], 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax4.set_ylabel('Calinski-Harabasz Score', fontsize=12)
    ax4.set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    optimal_k_ch = k_values[np.argmax(metrics['calinski_harabasz'])]
    ax4.axhline(y=max(metrics['calinski_harabasz']), color='r', linestyle='--', alpha=0.5)
    ax4.text(optimal_k_ch, max(metrics['calinski_harabasz']), 
             f'  K={optimal_k_ch}', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    filename = f'optimal_k_{method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    return optimal_k_sil, optimal_k_db, optimal_k_ch

# Generate plots for all methods
optimal_k_sil_h, optimal_k_db_h, optimal_k_ch_h = plot_optimal_k_metrics(metrics_hashtag, "Hashtag-Only")
optimal_k_sil_t, optimal_k_db_t, optimal_k_ch_t = plot_optimal_k_metrics(metrics_tfidf, "TF-IDF-Only")
optimal_k_sil_c, optimal_k_db_c, optimal_k_ch_c = plot_optimal_k_metrics(metrics_combined, "Combined")

# ============================================================================
# Save Metrics
# ============================================================================
print("\n[Saving Metrics]")

with open('optimal_k_metrics.pkl', 'wb') as f:
    pickle.dump({
        'hashtag': metrics_hashtag,
        'tfidf': metrics_tfidf,
        'combined': metrics_combined
    }, f)

print("✓ Metrics saved: optimal_k_metrics.pkl")

# ============================================================================
# Recommendations
# ============================================================================
print("\n" + "="*80)
print("OPTIMAL K RECOMMENDATIONS")
print("="*80)

results_df = pd.DataFrame({
    'Method': ['Hashtag-Only', 'TF-IDF-Only', 'Combined'],
    'Silhouette': [optimal_k_sil_h, optimal_k_sil_t, optimal_k_sil_c],
    'Davies-Bouldin': [optimal_k_db_h, optimal_k_db_t, optimal_k_db_c],
    'Calinski-Harabasz': [optimal_k_ch_h, optimal_k_ch_t, optimal_k_ch_c]
})

print(results_df.to_string(index=False))

# Consensus K
from collections import Counter
all_k = [optimal_k_sil_h, optimal_k_db_h, optimal_k_ch_h,
         optimal_k_sil_t, optimal_k_db_t, optimal_k_ch_t,
         optimal_k_sil_c, optimal_k_db_c, optimal_k_ch_c]
k_counts = Counter(all_k)
consensus_k = k_counts.most_common(1)[0][0]

print("\n" + "="*80)
print(f"🎯 CONSENSUS OPTIMAL K: {consensus_k}")
print(f"   (appears {k_counts[consensus_k]} times across all metrics)")
print("="*80)

# Save consensus K
with open('optimal_k.txt', 'w') as f:
    f.write(f"{consensus_k}\n")

print(f"\n✓ Consensus K saved to: optimal_k.txt")

print("\n" + "="*80)
print("PHASE 5 COMPLETE!")
print("="*80)
print("\n✓ Ready for Phase 6: Clustering experiments\n")
