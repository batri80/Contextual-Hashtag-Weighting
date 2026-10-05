"""
Phase 6: Clustering Experiments
================================
- Run K-means clustering with all three feature types
- Multiple runs with different random seeds
- Compute internal metrics
- Save results
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import time

print("="*80)
print("PHASE 6: CLUSTERING EXPERIMENTS")
print("="*80)

# ============================================================================
# Load Features and K
# ============================================================================
print("\n[Loading Data]")

test_hashtag = sparse.load_npz('test_hashtag.npz')
test_tfidf = sparse.load_npz('test_tfidf.npz')
test_combined = sparse.load_npz('test_combined.npz')

# Load optimal K
with open('optimal_k.txt', 'r') as f:
    K_OPTIMAL = int(f.read().strip())

print(f"✓ Loaded features")
print(f"✓ Optimal K: {K_OPTIMAL}")

# ============================================================================
# Clustering Function
# ============================================================================

def run_clustering_experiment(features, n_clusters, method_name, n_runs=5):
    """
    Run clustering experiment with multiple random initializations
    
    Returns: results dict with labels, metrics, and timing
    """
    
    print(f"\n{'='*80}")
    print(f"Clustering: {method_name} (K={n_clusters})")
    print(f"{'='*80}")
    
    results = {
        'method': method_name,
        'n_clusters': n_clusters,
        'runs': []
    }
    
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}...", end=" ", flush=True)
        
        run_result = {}
        start_time = time.time()
        
        # Run K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42 + run
        )
        
        labels = kmeans.fit_predict(features)
        elapsed = time.time() - start_time
        
        run_result['labels'] = labels
        run_result['time'] = elapsed
        run_result['inertia'] = kmeans.inertia_
        run_result['centers'] = kmeans.cluster_centers_
        
        # Silhouette score
        sample_size = min(10000, len(labels))
        run_result['silhouette'] = silhouette_score(
            features, labels, sample_size=sample_size
        )
        
        print(f"Time={elapsed:.2f}s, Silhouette={run_result['silhouette']:.4f}")
        
        results['runs'].append(run_result)
    
    # Compute averages
    results['avg_time'] = np.mean([r['time'] for r in results['runs']])
    results['std_time'] = np.std([r['time'] for r in results['runs']])
    results['avg_silhouette'] = np.mean([r['silhouette'] for r in results['runs']])
    results['std_silhouette'] = np.std([r['silhouette'] for r in results['runs']])
    results['avg_inertia'] = np.mean([r['inertia'] for r in results['runs']])
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {method_name}")
    print(f"{'='*80}")
    print(f"Avg Time:       {results['avg_time']:.2f} ± {results['std_time']:.2f} seconds")
    print(f"Avg Silhouette: {results['avg_silhouette']:.4f} ± {results['std_silhouette']:.4f}")
    print(f"Avg Inertia:    {results['avg_inertia']:.2f}")
    
    return results

# ============================================================================
# Run Experiments for All Three Methods
# ============================================================================

print("\n" + "="*80)
print("RUNNING CLUSTERING EXPERIMENTS")
print("="*80)

# Experiment 1: Hashtag-Only (Contextual Weights)
results_hashtag = run_clustering_experiment(
    test_hashtag, 
    K_OPTIMAL, 
    "Hashtag-Only (Contextual)",
    n_runs=5
)

# Experiment 2: TF-IDF-Only  
results_tfidf = run_clustering_experiment(
    test_tfidf,
    K_OPTIMAL,
    "TF-IDF-Only",
    n_runs=5
)

# Experiment 3: Combined (Hashtag + TF-IDF)
results_combined = run_clustering_experiment(
    test_combined,
    K_OPTIMAL,
    "Combined (Hashtag + TF-IDF)",
    n_runs=5
)

# ============================================================================
# Save Results
# ============================================================================
print("\n[Saving Results]")

with open('clustering_results.pkl', 'wb') as f:
    pickle.dump({
        'hashtag': results_hashtag,
        'tfidf': results_tfidf,
        'combined': results_combined,
        'K': K_OPTIMAL
    }, f)

print("✓ Results saved: clustering_results.pkl")

# ============================================================================
# Quick Comparison
# ============================================================================
print("\n" + "="*80)
print("QUICK COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Method': ['Hashtag-Only', 'TF-IDF-Only', 'Combined'],
    'Silhouette': [
        results_hashtag['avg_silhouette'],
        results_tfidf['avg_silhouette'],
        results_combined['avg_silhouette']
    ],
    'Std': [
        results_hashtag['std_silhouette'],
        results_tfidf['std_silhouette'],
        results_combined['std_silhouette']
    ],
    'Time (s)': [
        results_hashtag['avg_time'],
        results_tfidf['avg_time'],
        results_combined['avg_time']
    ]
})

print(comparison_df.to_string(index=False))

# Find best method
best_idx = comparison_df['Silhouette'].idxmax()
best_method = comparison_df.loc[best_idx, 'Method']

print("\n" + "="*80)
print(f"🏆 Best Method: {best_method}")
print(f"   Silhouette: {comparison_df.loc[best_idx, 'Silhouette']:.4f}")
print("="*80)

print("\n" + "="*80)
print("PHASE 6 COMPLETE!")
print("="*80)
print("\n✓ Ready for Phase 7: Performance analysis\n")
