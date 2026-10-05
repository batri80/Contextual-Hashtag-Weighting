"""
Qualitative Cluster Analysis - Election 2020
=============================================
Extracts and analyzes top hashtags for each cluster.
Works with Election 2020 dataset results.
"""

import numpy as np
import pandas as pd
from scipy import sparse
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUALITATIVE CLUSTER ANALYSIS - ELECTION 2020")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================
FEATURE_TYPE = 'Hashtag-Only'
K_VALUE = 20
ALGORITHM = 'KMeans'

print(f"\nConfiguration:")
print(f"  Feature Type: {FEATURE_TYPE}")
print(f"  K: {K_VALUE}")
print(f"  Algorithm: {ALGORITHM}")

# ============================================================================
# Step 1: Load Data and Results
# ============================================================================
print(f"\n[Step 1] Loading data and clustering results...")

# Try to load enhanced clustering results
try:
    with open('enhanced_clustering_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    print(f"✓ Loaded enhanced clustering results")
except FileNotFoundError:
    print(f"✗ enhanced_clustering_results.pkl not found")
    print(f"  Looking for alternative result files...")
    
    # Try to find the results by loading directly from embeddings and re-clustering
    print(f"\n  Will load from raw data and extract cluster labels...")
    all_results = None

# Load test data
try:
    with open('test_with_weights.pkl', 'rb') as f:
        test_df = pickle.load(f)
    print(f"✓ Loaded test data: {len(test_df)} tweets")
except FileNotFoundError:
    print(f"✗ ERROR: test_with_weights.pkl not found!")
    print(f"  This file is required. Please ensure you've run Phase 2.")
    exit(1)

# Load hashtag embeddings
try:
    hashtag_embeddings = sparse.load_npz('test_hashtag.npz')
    print(f"✓ Loaded hashtag embeddings: {hashtag_embeddings.shape}")
except FileNotFoundError:
    print(f"✗ ERROR: test_hashtag.npz not found!")
    print(f"  Run Phase 4 first to generate feature matrices.")
    exit(1)

# ============================================================================
# Step 2: Get or Generate Cluster Labels
# ============================================================================
print(f"\n[Step 2] Extracting cluster labels for K={K_VALUE}...")

if all_results is not None:
    # Find the specific result we want
    target_result = None
    for result in all_results:
        if (result['Features'] == FEATURE_TYPE and 
            result['K'] == K_VALUE and 
            result['Algorithm'] == ALGORITHM):
            target_result = result
            break
    
    if target_result is not None:
        labels = target_result['Labels']
        silhouette = target_result['Silhouette']
        print(f"✓ Found labels from enhanced results")
        print(f"  Silhouette Score: {silhouette:.4f}")
    else:
        print(f"✗ Could not find specific configuration in results")
        print(f"  Re-clustering with K={K_VALUE}...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=K_VALUE, random_state=42, n_init=10)
        labels = kmeans.fit_predict(hashtag_embeddings)
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(hashtag_embeddings, labels, sample_size=10000)
        print(f"✓ Re-clustered data")
        print(f"  Silhouette Score: {silhouette:.4f}")
else:
    # No results file, re-cluster
    print(f"  Re-clustering data with K={K_VALUE}...")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K_VALUE, random_state=42, n_init=10)
    labels = kmeans.fit_predict(hashtag_embeddings)
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(hashtag_embeddings, labels, sample_size=10000)
    print(f"✓ Generated cluster labels")
    print(f"  Silhouette Score: {silhouette:.4f}")

# Add labels to dataframe
test_df['cluster'] = labels

# ============================================================================
# Step 3: Analyze Clusters
# ============================================================================
print(f"\n[Step 3] Analyzing cluster characteristics...")

cluster_stats = []

for cluster_id in range(K_VALUE):
    cluster_tweets = test_df[test_df['cluster'] == cluster_id]
    
    # Collect all hashtags in this cluster
    all_hashtags = []
    for hashtags in cluster_tweets['hashtags']:
        if isinstance(hashtags, list):
            all_hashtags.extend([h.lower() for h in hashtags])
        elif isinstance(hashtags, str):
            all_hashtags.extend([h.lower() for h in hashtags.split(',')])
    
    # Count hashtags
    hashtag_counts = Counter(all_hashtags)
    top_hashtags = hashtag_counts.most_common(20)
    
    stats = {
        'cluster_id': cluster_id,
        'size': len(cluster_tweets),
        'percentage': (len(cluster_tweets) / len(test_df)) * 100,
        'unique_hashtags': len(hashtag_counts),
        'top_hashtags': top_hashtags,
        'total_hashtag_occurrences': sum(hashtag_counts.values()),
        'avg_hashtags_per_tweet': sum(hashtag_counts.values()) / len(cluster_tweets) if len(cluster_tweets) > 0 else 0
    }
    
    cluster_stats.append(stats)

print(f"✓ Analyzed {K_VALUE} clusters")

# ============================================================================
# Step 4: Display Results
# ============================================================================
print(f"\n{'='*80}")
print(f"CLUSTER ANALYSIS RESULTS (K={K_VALUE})")
print(f"{'='*80}")

# Sort clusters by size (descending)
cluster_stats_sorted = sorted(cluster_stats, key=lambda x: x['size'], reverse=True)

for stats in cluster_stats_sorted:
    print(f"\n{'─'*80}")
    print(f"Cluster {stats['cluster_id']} (Size: {stats['size']:,} tweets, {stats['percentage']:.2f}%)")
    print(f"{'─'*80}")
    
    print(f"\nTop 20 Hashtags:")
    for i, (hashtag, count) in enumerate(stats['top_hashtags'], 1):
        percentage = (count / stats['total_hashtag_occurrences']) * 100
        print(f"  {i:2d}. #{hashtag:<30s} ({count:5,} occurrences, {percentage:5.2f}%)")
    
    print(f"\nCluster Statistics:")
    print(f"  Unique hashtags: {stats['unique_hashtags']:,}")
    print(f"  Avg hashtags/tweet: {stats['avg_hashtags_per_tweet']:.2f}")

# ============================================================================
# Step 5: Cluster Interpretations
# ============================================================================
print(f"\n{'='*80}")
print(f"SUGGESTED CLUSTER INTERPRETATIONS")
print(f"{'='*80}")

print(f"\nBased on top hashtags, clusters may represent:")

for stats in cluster_stats_sorted[:10]:  # Show top 10 largest
    top_5_hashtags = [h for h, _ in stats['top_hashtags'][:5]]
    print(f"\nCluster {stats['cluster_id']} ({stats['percentage']:.1f}%):")
    print(f"  Top tags: {', '.join(['#'+h for h in top_5_hashtags])}")
    print(f"  → [Manual interpretation needed based on domain knowledge]")

# ============================================================================
# Step 6: Save Results
# ============================================================================
print(f"\n[Step 6] Saving results...")

# Save detailed cluster information
cluster_info = {
    'K': K_VALUE,
    'feature_type': FEATURE_TYPE,
    'algorithm': ALGORITHM,
    'silhouette': silhouette,
    'cluster_stats': cluster_stats,
    'labels': labels
}

with open(f'cluster_hashtags_k{K_VALUE}_election2020.pkl', 'wb') as f:
    pickle.dump(cluster_info, f)

# Create CSV summary
summary_data = []
for stats in cluster_stats:
    top_5 = ', '.join([f"#{h}" for h, _ in stats['top_hashtags'][:5]])
    summary_data.append({
        'Cluster_ID': stats['cluster_id'],
        'Size': stats['size'],
        'Percentage': f"{stats['percentage']:.2f}%",
        'Unique_Hashtags': stats['unique_hashtags'],
        'Top_5_Hashtags': top_5
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Size', ascending=False)
summary_df.to_csv(f'cluster_stats_k{K_VALUE}_election2020.csv', index=False)

print(f"✓ Results saved:")
print(f"  - cluster_hashtags_k{K_VALUE}_election2020.pkl")
print(f"  - cluster_stats_k{K_VALUE}_election2020.csv")

# ============================================================================
# Step 7: Cluster Balance Analysis
# ============================================================================
print(f"\n[Step 7] Cluster balance analysis...")

sizes = [s['size'] for s in cluster_stats]
mean_size = np.mean(sizes)
std_size = np.std(sizes)
cv = std_size / mean_size if mean_size > 0 else 0

print(f"\nCluster Size Distribution:")
print(f"  Mean: {mean_size:.1f} tweets")
print(f"  Std Dev: {std_size:.1f}")
print(f"  Coefficient of Variation: {cv:.3f}")
print(f"  Min Size: {min(sizes):,} tweets")
print(f"  Max Size: {max(sizes):,} tweets")

if cv < 0.5:
    print(f"  → Well-balanced clusters")
elif cv < 1.0:
    print(f"  → Moderately balanced clusters")
else:
    print(f"  → Imbalanced clusters (some very large, some very small)")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*80}")
print(f"QUALITATIVE ANALYSIS COMPLETE!")
print(f"{'='*80}")

print(f"\nKey Findings:")
print(f"  • {K_VALUE} clusters identified")
print(f"  • Silhouette Score: {silhouette:.4f}")
print(f"  • Largest cluster: {max(sizes):,} tweets ({max(sizes)/len(test_df)*100:.1f}%)")
print(f"  • Smallest cluster: {min(sizes):,} tweets ({min(sizes)/len(test_df)*100:.1f}%)")
print(f"  • Cluster balance (CV): {cv:.3f}")

print(f"\nNext Steps:")
print(f"  1. Review top hashtags for each cluster")
print(f"  2. Assign meaningful topic labels")
print(f"  3. Analyze temporal evolution (if timestamps available)")
print(f"  4. Compare with COVID-19 cluster themes")

print(f"\n{'='*80}")