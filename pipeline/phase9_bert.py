"""
Phase 9: BERT Baseline
======================
- Compute BERT embeddings for test set
- Run clustering with BERT features
- Compare with contextual hashtag method
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_rel, wilcoxon
import pickle
import time

print("="*80)
print("PHASE 9: BERT BASELINE")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[Loading Data]")

test_df = pd.read_pickle('test_with_weights.pkl')

with open('optimal_k.txt', 'r') as f:
    K_OPTIMAL = int(f.read().strip())

with open('clustering_results.pkl', 'rb') as f:
    results = pickle.load(f)
    results_combined = results['combined']

print(f"✓ Loaded test data: {len(test_df):,} tweets")
print(f"✓ Optimal K: {K_OPTIMAL}")

# ============================================================================
# Compute BERT Features
# ============================================================================
print("\n[Computing BERT Features]")

def create_bert_features(df, batch_size=32):
    """
    Create BERT-based features for clustering
    Use [CLS] token embedding as document representation
    """
    
    print(f"  Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"  Using device: {device}")
    
    texts = df['clean_text'].tolist()
    bert_embeddings = []
    
    print(f"  Processing {len(texts):,} texts...")
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Get embeddings
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            bert_embeddings.append(cls_embeddings.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"    Processed {i+len(batch_texts):,}/{len(texts):,}")
    
    bert_embeddings = np.vstack(bert_embeddings)
    
    print(f"✓ BERT embeddings computed")
    print(f"  Shape: {bert_embeddings.shape}")
    
    return bert_embeddings

bert_features = create_bert_features(test_df, batch_size=32)

# Save BERT features
np.save('test_bert_features.npy', bert_features)
print("✓ Saved: test_bert_features.npy")

# ============================================================================
# Run Clustering with BERT
# ============================================================================
print("\n[Running Clustering with BERT]")

def run_clustering_experiment(features, n_clusters, method_name, n_runs=5):
    """Run clustering experiment with multiple runs"""
    
    print(f"\nClustering: {method_name} (K={n_clusters})")
    
    results = {
        'method': method_name,
        'n_clusters': n_clusters,
        'runs': []
    }
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)
        
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
    
    print(f"\nSummary:")
    print(f"  Avg Time: {results['avg_time']:.2f} ± {results['std_time']:.2f} s")
    print(f"  Avg Silhouette: {results['avg_silhouette']:.4f} ± {results['std_silhouette']:.4f}")
    
    return results

results_bert = run_clustering_experiment(
    bert_features,
    K_OPTIMAL,
    "BERT [CLS] Embeddings",
    n_runs=5
)

# ============================================================================
# Compare with Combined Method
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: Combined (Ours) vs BERT Baseline")
print("="*80)

combined_sil = results_combined['avg_silhouette']
bert_sil = results_bert['avg_silhouette']

print(f"\nCombined (Hashtag + TF-IDF): {combined_sil:.4f} ± {results_combined['std_silhouette']:.4f}")
print(f"BERT Baseline:                {bert_sil:.4f} ± {results_bert['std_silhouette']:.4f}")

improvement = ((combined_sil - bert_sil) / bert_sil) * 100
print(f"\nImprovement: {improvement:+.2f}%")

# ============================================================================
# Statistical Test
# ============================================================================
print("\n[Statistical Significance Test]")

combined_scores = [r['silhouette'] for r in results_combined['runs']]
bert_scores = [r['silhouette'] for r in results_bert['runs']]

# Paired t-test
t_stat, p_value_t = ttest_rel(combined_scores, bert_scores)

# Wilcoxon
w_stat, p_value_w = wilcoxon(combined_scores, bert_scores)

# Cohen's d
mean_diff = np.mean(combined_scores) - np.mean(bert_scores)
pooled_std = np.sqrt((np.var(combined_scores) + np.var(bert_scores)) / 2)
cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

print(f"\nPaired t-test:")
print(f"  t-statistic: {t_stat:.4f}")
sig = '***' if p_value_t < 0.001 else '**' if p_value_t < 0.01 else '*' if p_value_t < 0.05 else 'ns'
print(f"  p-value:     {p_value_t:.4f} {sig}")

print(f"\nWilcoxon test:")
print(f"  W-statistic: {w_stat:.4f}")
sig_w = '***' if p_value_w < 0.001 else '**' if p_value_w < 0.01 else '*' if p_value_w < 0.05 else 'ns'
print(f"  p-value:     {p_value_w:.4f} {sig_w}")

print(f"\nCohen's d: {cohens_d:.4f}")
effect = 'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
print(f"Effect size: {effect}")

if p_value_t < 0.05:
    if mean_diff > 0:
        print("\n✓ Combined method is SIGNIFICANTLY BETTER than BERT")
    else:
        print("\n✓ BERT is SIGNIFICANTLY BETTER than Combined method")
else:
    print("\n✗ No significant difference")

# ============================================================================
# Save Results
# ============================================================================
print("\n[Saving Results]")

# Update clustering results
with open('clustering_results.pkl', 'rb') as f:
    all_results = pickle.load(f)

all_results['bert'] = results_bert

with open('clustering_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# Save BERT comparison
bert_comparison = {
    'combined_vs_bert': {
        't_stat': t_stat,
        'p_value_t': p_value_t,
        'w_stat': w_stat,
        'p_value_w': p_value_w,
        'cohens_d': cohens_d,
        'significant': p_value_t < 0.05
    }
}

with open('bert_comparison.pkl', 'wb') as f:
    pickle.dump(bert_comparison, f)

print("✓ Results saved")

print("\n" + "="*80)
print("PHASE 9 COMPLETE!")
print("="*80)
print("\n✓ Ready for Phase 10: Final report generation\n")
