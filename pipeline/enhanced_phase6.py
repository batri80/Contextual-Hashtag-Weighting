"""
Enhanced Phase 6: Comprehensive Clustering Analysis
====================================================
- Test multiple K values (K=2, 5, 10, 15, 20)
- Compare multiple clustering algorithms (K-Means, Hierarchical, GMM)
- Evaluate with multiple metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Analyze cluster characteristics
- Save comprehensive results
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
import warnings
import time
warnings.filterwarnings('ignore')

print("="*80)
print("ENHANCED PHASE 6: COMPREHENSIVE CLUSTERING ANALYSIS")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================
K_VALUES = [2, 5, 10, 15, 20]  # Test multiple K values
ALGORITHMS = {
    'KMeans': lambda k: KMeans(n_clusters=k, random_state=42, n_init=10),
}

# ============================================================================
# Step 1: Load Features
# ============================================================================
print("\n[Step 1] Loading embeddings...")

with open('hashtag_embeddings.pkl', 'rb') as f:
    hashtag_emb = pickle.load(f)

with open('tfidf_embeddings.pkl', 'rb') as f:
    tfidf_emb = pickle.load(f)

with open('combined_embeddings.pkl', 'rb') as f:
    combined_emb = pickle.load(f)

# Load test data for analysis
test_df = pd.read_pickle('test_data.pkl')

print(f"✓ Loaded embeddings:")
print(f"  Hashtag: {hashtag_emb.shape}")
print(f"  TF-IDF: {tfidf_emb.shape}")
print(f"  Combined: {combined_emb.shape}")
print(f"  Test data: {len(test_df)} tweets")

feature_sets = {
    'Hashtag-Only': hashtag_emb,
    'TF-IDF-Only': tfidf_emb,
    'Combined': combined_emb
}

# ============================================================================
# Step 2: Run Comprehensive Experiments
# ============================================================================
print("\n[Step 2] Running clustering experiments...")
print(f"Testing K values: {K_VALUES}")
print(f"Algorithms: {list(ALGORITHMS.keys())}")

results = []

for feat_name, features in feature_sets.items():
    print(f"\n{'='*80}")
    print(f"Feature Set: {feat_name}")
    print(f"{'='*80}")
    
    for k in K_VALUES:
        print(f"\n  K = {k}:")
        
        for algo_name, algo_func in ALGORITHMS.items():
            try:
                start_time = time.time()
                
                # Fit clustering
                clusterer = algo_func(k)
                
                if algo_name == 'GMM':
                    labels = clusterer.fit_predict(features)
                else:
                    labels = clusterer.fit_predict(features)
                
                elapsed = time.time() - start_time
                
                # Calculate metrics
                sil = silhouette_score(features, labels, sample_size=min(10000, len(labels)))
                db = davies_bouldin_score(features, labels)
                ch = calinski_harabasz_score(features, labels)
                
                # Cluster size statistics
                cluster_sizes = Counter(labels)
                sizes = list(cluster_sizes.values())
                min_size = min(sizes)
                max_size = max(sizes)
                mean_size = np.mean(sizes)
                std_size = np.std(sizes)
                
                # Calculate balance (coefficient of variation)
                balance = std_size / mean_size if mean_size > 0 else 0
                
                result = {
                    'Features': feat_name,
                    'Algorithm': algo_name,
                    'K': k,
                    'Silhouette': sil,
                    'Davies-Bouldin': db,
                    'Calinski-Harabasz': ch,
                    'Min_Size': min_size,
                    'Max_Size': max_size,
                    'Mean_Size': mean_size,
                    'Std_Size': std_size,
                    'Balance': balance,
                    'Time': elapsed,
                    'Labels': labels
                }
                
                results.append(result)
                
                print(f"    {algo_name:20s} | Sil={sil:.4f} | DB={db:.4f} | CH={ch:>8.1f} | Balance={balance:.3f} | {elapsed:.2f}s")
                
            except Exception as e:
                print(f"    {algo_name:20s} | FAILED: {str(e)[:60]}")

print("\n" + "="*80)
print("All experiments completed!")
print("="*80)

# ============================================================================
# Step 3: Analyze Results
# ============================================================================
print("\n[Step 3] Analyzing results...")

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Labels'} for r in results])

# Save full results
with open('enhanced_clustering_results.pkl', 'wb') as f:
    pickle.dump(results, f)

results_df.to_csv('enhanced_clustering_results.csv', index=False)

print(f"✓ Results saved:")
print(f"  - enhanced_clustering_results.pkl (with cluster labels)")
print(f"  - enhanced_clustering_results.csv (metrics only)")

# ============================================================================
# Step 4: Find Best Configurations
# ============================================================================
print("\n[Step 4] Identifying best configurations...")

# Best configurations per feature set
print("\n" + "="*80)
print("BEST CONFIGURATIONS PER FEATURE SET (by Silhouette)")
print("="*80)

for feat_name in feature_sets.keys():
    feat_results = results_df[results_df['Features'] == feat_name]
    best = feat_results.nlargest(5, 'Silhouette')[['Algorithm', 'K', 'Silhouette', 'Davies-Bouldin', 'Balance']]
    print(f"\n{feat_name}:")
    print(best.to_string(index=False))

# Best K per feature set
print("\n" + "="*80)
print("OPTIMAL K PER FEATURE SET")
print("="*80)

for feat_name in feature_sets.keys():
    feat_results = results_df[results_df['Features'] == feat_name]
    
    # Average metrics across algorithms for each K
    k_summary = feat_results.groupby('K').agg({
        'Silhouette': 'mean',
        'Davies-Bouldin': 'mean',
        'Calinski-Harabasz': 'mean',
        'Balance': 'mean'
    }).reset_index()
    
    # Find optimal K by each metric
    opt_sil_k = k_summary.loc[k_summary['Silhouette'].idxmax(), 'K']
    opt_db_k = k_summary.loc[k_summary['Davies-Bouldin'].idxmin(), 'K']
    opt_ch_k = k_summary.loc[k_summary['Calinski-Harabasz'].idxmax(), 'K']
    
    print(f"\n{feat_name}:")
    print(f"  Optimal K (Silhouette):        {int(opt_sil_k)}")
    print(f"  Optimal K (Davies-Bouldin):    {int(opt_db_k)}")
    print(f"  Optimal K (Calinski-Harabasz): {int(opt_ch_k)}")

# ============================================================================
# Step 5: Generate Visualizations
# ============================================================================
print("\n[Step 5] Generating visualizations...")

# 5.1: Silhouette vs K curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feat_name in enumerate(feature_sets.keys()):
    ax = axes[idx]
    feat_results = results_df[results_df['Features'] == feat_name]
    
    for algo_name in ALGORITHMS.keys():
        algo_results = feat_results[feat_results['Algorithm'] == algo_name]
        if len(algo_results) > 0:
            ax.plot(algo_results['K'], algo_results['Silhouette'], 
                   marker='o', label=algo_name, linewidth=2)
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title(f'{feat_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)

plt.tight_layout()
plt.savefig('silhouette_vs_k.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: silhouette_vs_k.png")
plt.close()

# 5.2: Davies-Bouldin vs K curves (lower is better)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feat_name in enumerate(feature_sets.keys()):
    ax = axes[idx]
    feat_results = results_df[results_df['Features'] == feat_name]
    
    for algo_name in ALGORITHMS.keys():
        algo_results = feat_results[feat_results['Algorithm'] == algo_name]
        if len(algo_results) > 0:
            ax.plot(algo_results['K'], algo_results['Davies-Bouldin'], 
                   marker='o', label=algo_name, linewidth=2)
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index', fontsize=12)
    ax.set_title(f'{feat_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)

plt.tight_layout()
plt.savefig('davies_bouldin_vs_k.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: davies_bouldin_vs_k.png")
plt.close()

# 5.3: Heatmap of Silhouette scores
pivot_data = {}
for feat_name in feature_sets.keys():
    feat_results = results_df[results_df['Features'] == feat_name]
    pivot = feat_results.pivot_table(values='Silhouette', 
                                     index='Algorithm', 
                                     columns='K', 
                                     aggfunc='mean')
    pivot_data[feat_name] = pivot

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feat_name in enumerate(feature_sets.keys()):
    ax = axes[idx]
    sns.heatmap(pivot_data[feat_name], annot=True, fmt='.4f', cmap='YlGnBu', 
                ax=ax, cbar_kws={'label': 'Silhouette Score'})
    ax.set_title(f'{feat_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)

plt.tight_layout()
plt.savefig('silhouette_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: silhouette_heatmap.png")
plt.close()

# 5.4: Cluster balance analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feat_name in enumerate(feature_sets.keys()):
    ax = axes[idx]
    feat_results = results_df[results_df['Features'] == feat_name]
    
    for algo_name in ALGORITHMS.keys():
        algo_results = feat_results[feat_results['Algorithm'] == algo_name]
        if len(algo_results) > 0:
            ax.plot(algo_results['K'], algo_results['Balance'], 
                   marker='o', label=algo_name, linewidth=2)
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Balance (Coefficient of Variation)', fontsize=12)
    ax.set_title(f'{feat_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Threshold')

plt.tight_layout()
plt.savefig('cluster_balance.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: cluster_balance.png")
plt.close()

# ============================================================================
# Step 6: Summary Report
# ============================================================================
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

print("\n📊 Overall Best Configurations:")
best_overall = results_df.nlargest(10, 'Silhouette')[['Features', 'Algorithm', 'K', 'Silhouette', 'Davies-Bouldin', 'Balance']]
print(best_overall.to_string(index=False))

print("\n🎯 Recommended Configurations:")
for feat_name in feature_sets.keys():
    feat_results = results_df[results_df['Features'] == feat_name]
    best = feat_results.nlargest(1, 'Silhouette').iloc[0]
    print(f"\n{feat_name}:")
    print(f"  Algorithm: {best['Algorithm']}")
    print(f"  K: {int(best['K'])}")
    print(f"  Silhouette: {best['Silhouette']:.4f}")
    print(f"  Davies-Bouldin: {best['Davies-Bouldin']:.4f}")
    print(f"  Balance: {best['Balance']:.3f}")

print("\n" + "="*80)
print("✓ ENHANCED PHASE 6 COMPLETE!")
print("="*80)
print("\nFiles generated:")
print("  - enhanced_clustering_results.pkl")
print("  - enhanced_clustering_results.csv")
print("  - silhouette_vs_k.png")
print("  - davies_bouldin_vs_k.png")
print("  - silhouette_heatmap.png")
print("  - cluster_balance.png")
