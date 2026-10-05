"""
Ablation Study - Fixed Version for Election 2020
=================================================
Tests different combinations of Hashtag and TF-IDF features to find optimal weighting.

Fixed: Concatenates features instead of adding (handles different dimensions).
"""

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ABLATION STUDY - FEATURE WEIGHT OPTIMIZATION")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================
K = 20
ALPHA_VALUES = [1.0, 0.75, 0.5, 0.25, 0.0]  # Weight for hashtag features
N_RUNS = 3  # Runs per configuration for stability

print(f"\nConfiguration:")
print(f"  K: {K}")
print(f"  Alpha values: {ALPHA_VALUES}")
print(f"  (Alpha=1.0: 100% Hashtag, Alpha=0.0: 100% TF-IDF)")
print(f"  Runs per configuration: {N_RUNS}")

# ============================================================================
# Step 1: Load Features
# ============================================================================
print(f"\n[Step 1] Loading features...")

# Load hashtag features
try:
    hashtag_features = sparse.load_npz('test_hashtag.npz')
    print(f"✓ Hashtag features: {hashtag_features.shape}")
except FileNotFoundError:
    print(f"✗ ERROR: test_hashtag.npz not found!")
    exit(1)

# Load TF-IDF features
try:
    tfidf_features = sparse.load_npz('test_tfidf.npz')
    print(f"✓ TF-IDF features: {tfidf_features.shape}")
except FileNotFoundError:
    print(f"✗ ERROR: test_tfidf.npz not found!")
    exit(1)

n_samples = hashtag_features.shape[0]
n_hashtag_features = hashtag_features.shape[1]
n_tfidf_features = tfidf_features.shape[1]

print(f"\n  Total samples: {n_samples:,}")
print(f"  Hashtag dimensions: {n_hashtag_features:,}")
print(f"  TF-IDF dimensions: {n_tfidf_features:,}")

# ============================================================================
# Step 2: Normalize Features
# ============================================================================
print(f"\n[Step 2] Normalizing features...")

from sklearn.preprocessing import normalize

hashtag_normalized = normalize(hashtag_features, norm='l2', axis=1)
tfidf_normalized = normalize(tfidf_features, norm='l2', axis=1)

print(f"✓ Features normalized (L2 norm)")

# ============================================================================
# Step 3: Run Ablation Study
# ============================================================================
print(f"\n[Step 3] Running ablation study...")
print(f"{'='*80}")

results = []

for alpha in ALPHA_VALUES:
    beta = 1.0 - alpha
    
    print(f"\n[Alpha = {alpha:.2f}] ({alpha*100:.0f}% Hashtag + {beta*100:.0f}% TF-IDF)")
    print(f"{'-'*80}")
    
    run_scores = []
    
    for run in range(N_RUNS):
        # Create weighted combination by concatenation
        # Scale each part by its weight
        hashtag_weighted = hashtag_normalized * alpha
        tfidf_weighted = tfidf_normalized * beta
        
        # Concatenate (not add - different dimensions!)
        combined = sparse.hstack([hashtag_weighted, tfidf_weighted])
        
        # Cluster
        kmeans = KMeans(n_clusters=K, random_state=42 + run, n_init=10)
        labels = kmeans.fit_predict(combined)
        
        # Evaluate
        silhouette = silhouette_score(combined, labels, sample_size=10000)
        run_scores.append(silhouette)
        
        print(f"  Run {run+1}/{N_RUNS}: Silhouette = {silhouette:.4f}")
    
    # Statistics
    avg_score = np.mean(run_scores)
    std_score = np.std(run_scores)
    
    results.append({
        'Alpha': alpha,
        'Hashtag_Weight': f'{alpha*100:.0f}%',
        'TFIDF_Weight': f'{beta*100:.0f}%',
        'Avg_Silhouette': avg_score,
        'Std_Silhouette': std_score,
        'Scores': run_scores
    })
    
    print(f"\n  Summary: {avg_score:.4f} ± {std_score:.4f}")

# ============================================================================
# Step 4: Results Summary
# ============================================================================
print(f"\n{'='*80}")
print("ABLATION STUDY RESULTS")
print(f"{'='*80}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Avg_Silhouette', ascending=False)

print(f"\n{'Configuration':<30} {'Silhouette':<20} {'Rank'}")
print(f"{'-'*70}")

for idx, row in results_df.iterrows():
    config = f"{row['Hashtag_Weight']} Hashtag + {row['TFIDF_Weight']} TF-IDF"
    score = f"{row['Avg_Silhouette']:.4f} ± {row['Std_Silhouette']:.4f}"
    rank = "🏆 BEST" if idx == results_df.index[0] else f"  #{results_df.index.tolist().index(idx) + 1}"
    print(f"{config:<30} {score:<20} {rank}")

# ============================================================================
# Step 5: Analysis
# ============================================================================
print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}")

best_config = results_df.iloc[0]
worst_config = results_df.iloc[-1]

print(f"\nBest Configuration:")
print(f"  {best_config['Hashtag_Weight']} Hashtag + {best_config['TFIDF_Weight']} TF-IDF")
print(f"  Silhouette: {best_config['Avg_Silhouette']:.4f} ± {best_config['Std_Silhouette']:.4f}")

print(f"\nWorst Configuration:")
print(f"  {worst_config['Hashtag_Weight']} Hashtag + {worst_config['TFIDF_Weight']} TF-IDF")
print(f"  Silhouette: {worst_config['Avg_Silhouette']:.4f} ± {worst_config['Std_Silhouette']:.4f}")

improvement = ((best_config['Avg_Silhouette'] - worst_config['Avg_Silhouette']) 
               / worst_config['Avg_Silhouette'] * 100)
print(f"\nImprovement: {improvement:.2f}%")

# Find pure hashtag and pure TF-IDF
pure_hashtag = results_df[results_df['Alpha'] == 1.0].iloc[0]
pure_tfidf = results_df[results_df['Alpha'] == 0.0].iloc[0]

print(f"\nPure Hashtag vs Pure TF-IDF:")
print(f"  Hashtag-Only: {pure_hashtag['Avg_Silhouette']:.4f}")
print(f"  TF-IDF-Only:  {pure_tfidf['Avg_Silhouette']:.4f}")
print(f"  Difference:   {(pure_hashtag['Avg_Silhouette'] - pure_tfidf['Avg_Silhouette']) / pure_tfidf['Avg_Silhouette'] * 100:+.2f}%")

# ============================================================================
# Step 6: Key Findings
# ============================================================================
print(f"\n{'='*80}")
print("KEY FINDINGS")
print(f"{'='*80}")

if best_config['Alpha'] == 1.0:
    print(f"\n✓ FINDING: Pure Hashtag-Only features are OPTIMAL")
    print(f"  - Adding TF-IDF features DEGRADES performance")
    print(f"  - Text features introduce noise rather than signal")
elif best_config['Alpha'] == 0.0:
    print(f"\n⚠ FINDING: Pure TF-IDF features are OPTIMAL (unexpected!)")
    print(f"  - This contradicts main results - investigate!")
else:
    print(f"\n⚠ FINDING: Mixed features are optimal ({best_config['Hashtag_Weight']} Hashtag)")
    print(f"  - Balance point: Alpha = {best_config['Alpha']:.2f}")

# Check monotonic relationship
scores_by_alpha = [r['Avg_Silhouette'] for r in sorted(results, key=lambda x: x['Alpha'], reverse=True)]
is_monotonic_increasing = all(scores_by_alpha[i] >= scores_by_alpha[i+1] for i in range(len(scores_by_alpha)-1))
is_monotonic_decreasing = all(scores_by_alpha[i] <= scores_by_alpha[i+1] for i in range(len(scores_by_alpha)-1))

if is_monotonic_increasing:
    print(f"\n✓ Performance increases monotonically with Hashtag weight")
    print(f"  - More hashtag features → Better clustering")
elif is_monotonic_decreasing:
    print(f"\n✓ Performance decreases monotonically with Hashtag weight")
    print(f"  - More TF-IDF features → Better clustering")
else:
    print(f"\n✓ Performance shows non-monotonic relationship")
    print(f"  - Sweet spot at Alpha = {best_config['Alpha']:.2f}")

# ============================================================================
# Step 7: Save Results
# ============================================================================
print(f"\n[Step 7] Saving results...")

# Save summary
results_summary = results_df[['Hashtag_Weight', 'TFIDF_Weight', 'Avg_Silhouette', 'Std_Silhouette']].copy()
results_summary.to_csv('ablation_study_results_election2020.csv', index=False)
print(f"✓ Saved: ablation_study_results_election2020.csv")

# Save detailed results
import pickle
detailed_results = {
    'K': K,
    'results': results,
    'best_config': best_config.to_dict(),
    'worst_config': worst_config.to_dict(),
    'improvement': improvement
}

with open('ablation_study_results_election2020.pkl', 'wb') as f:
    pickle.dump(detailed_results, f)
print(f"✓ Saved: ablation_study_results_election2020.pkl")

# ============================================================================
# Step 8: Visualization (Optional)
# ============================================================================
print(f"\n[Step 8] Creating visualization...")

try:
    import matplotlib.pyplot as plt
    
    alphas = [r['Alpha'] for r in results]
    scores = [r['Avg_Silhouette'] for r in results]
    stds = [r['Std_Silhouette'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, scores, yerr=stds, marker='o', markersize=8, 
                 capsize=5, capthick=2, linewidth=2)
    
    plt.xlabel('Alpha (Hashtag Weight)', fontweight='bold', fontsize=12)
    plt.ylabel('Silhouette Score', fontweight='bold', fontsize=12)
    plt.title('Ablation Study: Feature Weight vs Clustering Quality', 
              fontweight='bold', fontsize=14)
    
    # Add labels
    plt.text(0.0, scores[0] * 0.95, '100% TF-IDF', ha='center', fontsize=9)
    plt.text(1.0, scores[-1] * 1.02, '100% Hashtag', ha='center', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ablation_study_election2020.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: ablation_study_election2020.png")
    plt.close()
except ImportError:
    print(f"  ✗ matplotlib not available, skipping visualization")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*80}")
print("ABLATION STUDY COMPLETE!")
print(f"{'='*80}")

print(f"\nOptimal Configuration:")
print(f"  {best_config['Hashtag_Weight']} Hashtag + {best_config['TFIDF_Weight']} TF-IDF")
print(f"  Silhouette: {best_config['Avg_Silhouette']:.4f}")

print(f"\nConclusion:")
if best_config['Alpha'] == 1.0:
    print(f"  Pure Hashtag features are optimal for Election 2020 clustering.")
    print(f"  This confirms the main finding: text features add noise, not signal.")
else:
    print(f"  Optimal weighting is {best_config['Hashtag_Weight']} Hashtag.")

print(f"\nFiles Generated:")
print(f"  • ablation_study_results_election2020.csv")
print(f"  • ablation_study_results_election2020.pkl")
print(f"  • ablation_study_election2020.png")

print(f"\n{'='*80}")