"""
Phase 8: Statistical Significance Testing
==========================================
- Paired t-tests for method comparisons
- Wilcoxon signed-rank tests (non-parametric)
- Cohen's d effect sizes
- Friedman test for all methods
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare
import pickle

print("="*80)
print("PHASE 8: STATISTICAL SIGNIFICANCE TESTING")
print("="*80)

# ============================================================================
# Load Results
# ============================================================================
print("\n[Loading Results]")

with open('clustering_results.pkl', 'rb') as f:
    results = pickle.load(f)

results_hashtag = results['hashtag']
results_tfidf = results['tfidf']
results_combined = results['combined']

print("✓ Loaded clustering results")

# ============================================================================
# Statistical Comparison Function
# ============================================================================

def statistical_comparison(results1, results2, metric='silhouette', 
                           method1_name='Method1', method2_name='Method2'):
    """
    Perform statistical tests comparing two methods
    """
    
    scores1 = [r[metric] for r in results1['runs']]
    scores2 = [r[metric] for r in results2['runs']]
    
    # Paired t-test
    t_stat, p_value_t = ttest_rel(scores1, scores2)
    
    # Wilcoxon signed-rank test (non-parametric)
    w_stat, p_value_w = wilcoxon(scores1, scores2)
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(scores1) - np.mean(scores2)
    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Statistical Test: {method1_name} vs {method2_name}")
    print(f"Metric: {metric}")
    print(f"{'='*80}")
    print(f"Mean {method1_name}: {np.mean(scores1):.4f} ± {np.std(scores1):.4f}")
    print(f"Mean {method2_name}: {np.mean(scores2):.4f} ± {np.std(scores2):.4f}")
    print(f"Mean Difference:  {mean_diff:.4f}")
    
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    sig_t = '***' if p_value_t < 0.001 else '**' if p_value_t < 0.01 else '*' if p_value_t < 0.05 else 'ns'
    print(f"  p-value:     {p_value_t:.4f} {sig_t}")
    
    print(f"\nWilcoxon signed-rank test:")
    print(f"  W-statistic: {w_stat:.4f}")
    sig_w = '***' if p_value_w < 0.001 else '**' if p_value_w < 0.01 else '*' if p_value_w < 0.05 else 'ns'
    print(f"  p-value:     {p_value_w:.4f} {sig_w}")
    
    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    effect_interpretation = ("negligible" if abs(cohens_d) < 0.2 else 
                            "small" if abs(cohens_d) < 0.5 else 
                            "medium" if abs(cohens_d) < 0.8 else "large")
    print(f"  Interpretation: {effect_interpretation}")
    
    # Conclusion
    print(f"\n{'='*80}")
    if p_value_t < 0.05:
        if mean_diff > 0:
            print(f"✓ {method1_name} is SIGNIFICANTLY BETTER than {method2_name}")
        else:
            print(f"✓ {method2_name} is SIGNIFICANTLY BETTER than {method1_name}")
    else:
        print(f"✗ No significant difference between methods")
    print(f"{'='*80}")
    
    return {
        't_stat': t_stat,
        'p_value_t': p_value_t,
        'w_stat': w_stat,
        'p_value_w': p_value_w,
        'cohens_d': cohens_d,
        'significant': p_value_t < 0.05,
        'better': method1_name if mean_diff > 0 else method2_name
    }

# ============================================================================
# Run Pairwise Comparisons
# ============================================================================

print("\n" + "="*80)
print("PAIRWISE COMPARISONS")
print("="*80)

# Comparison 1: Hashtag vs TF-IDF
stats_h_vs_t = statistical_comparison(
    results_hashtag, results_tfidf,
    'silhouette',
    'Hashtag-Only', 'TF-IDF-Only'
)

# Comparison 2: Combined vs Hashtag
stats_c_vs_h = statistical_comparison(
    results_combined, results_hashtag,
    'silhouette',
    'Combined', 'Hashtag-Only'
)

# Comparison 3: Combined vs TF-IDF
stats_c_vs_t = statistical_comparison(
    results_combined, results_tfidf,
    'silhouette',
    'Combined', 'TF-IDF-Only'
)

# ============================================================================
# Friedman Test (All Three Methods)
# ============================================================================

print("\n" + "="*80)
print("FRIEDMAN TEST (All Three Methods)")
print("="*80)

silhouette_hashtag = [r['silhouette'] for r in results_hashtag['runs']]
silhouette_tfidf = [r['silhouette'] for r in results_tfidf['runs']]
silhouette_combined = [r['silhouette'] for r in results_combined['runs']]

friedman_stat, friedman_p = friedmanchisquare(
    silhouette_hashtag, 
    silhouette_tfidf, 
    silhouette_combined
)

print(f"Friedman χ²: {friedman_stat:.4f}")
sig_friedman = '***' if friedman_p < 0.001 else '**' if friedman_p < 0.01 else '*' if friedman_p < 0.05 else 'ns'
print(f"p-value:     {friedman_p:.4f} {sig_friedman}")

if friedman_p < 0.05:
    print("\n✓ Significant difference exists among the three methods")
else:
    print("\n✗ No significant difference among the three methods")

# ============================================================================
# Save Statistical Results
# ============================================================================
print("\n[Saving Statistical Results]")

statistical_results = {
    'h_vs_t': stats_h_vs_t,
    'c_vs_h': stats_c_vs_h,
    'c_vs_t': stats_c_vs_t,
    'friedman': {
        'statistic': friedman_stat,
        'p_value': friedman_p
    }
}

with open('statistical_tests.pkl', 'wb') as f:
    pickle.dump(statistical_results, f)

print("✓ Results saved: statistical_tests.pkl")

# ============================================================================
# Create Summary Table
# ============================================================================
print("\n[Creating Summary Table]")

summary_df = pd.DataFrame({
    'Comparison': [
        'Hashtag vs TF-IDF',
        'Combined vs Hashtag',
        'Combined vs TF-IDF'
    ],
    'p-value (t-test)': [
        stats_h_vs_t['p_value_t'],
        stats_c_vs_h['p_value_t'],
        stats_c_vs_t['p_value_t']
    ],
    'Significant': [
        'Yes' if stats_h_vs_t['significant'] else 'No',
        'Yes' if stats_c_vs_h['significant'] else 'No',
        'Yes' if stats_c_vs_t['significant'] else 'No'
    ],
    'Cohen\'s d': [
        stats_h_vs_t['cohens_d'],
        stats_c_vs_h['cohens_d'],
        stats_c_vs_t['cohens_d']
    ],
    'Effect Size': [
        'negligible' if abs(stats_h_vs_t['cohens_d']) < 0.2 else 
        'small' if abs(stats_h_vs_t['cohens_d']) < 0.5 else 
        'medium' if abs(stats_h_vs_t['cohens_d']) < 0.8 else 'large',
        
        'negligible' if abs(stats_c_vs_h['cohens_d']) < 0.2 else 
        'small' if abs(stats_c_vs_h['cohens_d']) < 0.5 else 
        'medium' if abs(stats_c_vs_h['cohens_d']) < 0.8 else 'large',
        
        'negligible' if abs(stats_c_vs_t['cohens_d']) < 0.2 else 
        'small' if abs(stats_c_vs_t['cohens_d']) < 0.5 else 
        'medium' if abs(stats_c_vs_t['cohens_d']) < 0.8 else 'large'
    ]
})

print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)

summary_df.to_csv('statistical_summary.csv', index=False)
summary_df.to_latex('statistical_summary.tex', index=False, float_format="%.4f")

print("\n✓ Tables saved:")
print("  - statistical_summary.csv")
print("  - statistical_summary.tex")

# ============================================================================
# Key Findings
# ============================================================================
print("\n" + "="*80)
print("KEY STATISTICAL FINDINGS")
print("="*80)

# Find which method is best
methods = ['Hashtag-Only', 'TF-IDF-Only', 'Combined']
mean_scores = [
    np.mean(silhouette_hashtag),
    np.mean(silhouette_tfidf),
    np.mean(silhouette_combined)
]
best_idx = np.argmax(mean_scores)
best_method = methods[best_idx]

print(f"\n1. Best Method: {best_method} (Silhouette: {mean_scores[best_idx]:.4f})")

print(f"\n2. Pairwise Comparisons:")
for comp, stats in [('Hashtag vs TF-IDF', stats_h_vs_t),
                     ('Combined vs Hashtag', stats_c_vs_h),
                     ('Combined vs TF-IDF', stats_c_vs_t)]:
    if stats['significant']:
        print(f"   ✓ {comp}: {stats['better']} is significantly better (p={stats['p_value_t']:.4f})")
    else:
        print(f"   ✗ {comp}: No significant difference (p={stats['p_value_t']:.4f})")

print(f"\n3. Overall Test (Friedman):")
if friedman_p < 0.05:
    print(f"   ✓ Significant difference exists among methods (p={friedman_p:.4f})")
else:
    print(f"   ✗ No significant difference among methods (p={friedman_p:.4f})")

print("="*80)

print("\n" + "="*80)
print("PHASE 8 COMPLETE!")
print("="*80)
print("\n✓ Ready for Phase 9: BERT baseline\n")
