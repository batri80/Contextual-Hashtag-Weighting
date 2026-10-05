"""
Phase 7: Performance Analysis and Visualization
================================================
- Create comparison tables
- Generate performance visualizations
- Analyze computational efficiency
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("="*80)
print("PHASE 7: PERFORMANCE ANALYSIS")
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
K = results['K']

print(f"✓ Loaded clustering results (K={K})")

# ============================================================================
# Create Comparison Table
# ============================================================================
print("\n[Creating Comparison Table]")

comparison_df = pd.DataFrame({
    'Method': [
        'Hashtag-Only (Contextual)',
        'TF-IDF-Only',
        'Combined (Hashtag + TF-IDF)'
    ],
    'Avg Time (s)': [
        results_hashtag['avg_time'],
        results_tfidf['avg_time'],
        results_combined['avg_time']
    ],
    'Std Time (s)': [
        results_hashtag['std_time'],
        results_tfidf['std_time'],
        results_combined['std_time']
    ],
    'Avg Silhouette': [
        results_hashtag['avg_silhouette'],
        results_tfidf['avg_silhouette'],
        results_combined['avg_silhouette']
    ],
    'Std Silhouette': [
        results_hashtag['std_silhouette'],
        results_tfidf['std_silhouette'],
        results_combined['std_silhouette']
    ],
    'Avg Inertia': [
        results_hashtag['avg_inertia'],
        results_tfidf['avg_inertia'],
        results_combined['avg_inertia']
    ]
})

print("\n" + "="*80)
print("CLUSTERING PERFORMANCE COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Save table
comparison_df.to_csv('clustering_comparison.csv', index=False)
comparison_df.to_latex('clustering_comparison.tex', index=False, float_format="%.4f")

print("\n✓ Tables saved:")
print("  - clustering_comparison.csv")
print("  - clustering_comparison.tex")

# ============================================================================
# Visualizations
# ============================================================================
print("\n[Generating Visualizations]")

# Set style
sns.set_style("whitegrid")
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# ============================================================================
# Figure 1: Performance Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Silhouette Score
ax1 = axes[0]
x = range(len(comparison_df))
ax1.bar(x, comparison_df['Avg Silhouette'], 
        yerr=comparison_df['Std Silhouette'],
        capsize=5, alpha=0.7, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xticks(x)
ax1.set_xticklabels(['Hashtag-Only', 'TF-IDF-Only', 'Combined'], 
                     rotation=0, ha='center', fontsize=11)
ax1.set_ylabel('Silhouette Score', fontsize=13, fontweight='bold')
ax1.set_title('Clustering Quality (Higher is Better)', fontsize=15, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(comparison_df['Avg Silhouette'], 
                                     comparison_df['Std Silhouette'])):
    ax1.text(i, mean + std + 0.005, f'{mean:.4f}', 
             ha='center', fontsize=11, fontweight='bold')

# Plot 2: Computation Time
ax2 = axes[1]
ax2.bar(x, comparison_df['Avg Time (s)'],
        yerr=comparison_df['Std Time (s)'],
        capsize=5, alpha=0.7, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xticks(x)
ax2.set_xticklabels(['Hashtag-Only', 'TF-IDF-Only', 'Combined'], 
                     rotation=0, ha='center', fontsize=11)
ax2.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
ax2.set_title('Computational Efficiency (Lower is Better)', fontsize=15, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(comparison_df['Avg Time (s)'], 
                                     comparison_df['Std Time (s)'])):
    ax2.text(i, mean + std + 0.5, f'{mean:.2f}s', 
             ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: clustering_comparison.png")
plt.close()

# ============================================================================
# Figure 2: Detailed Metrics Across Runs
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

methods = ['Hashtag-Only', 'TF-IDF-Only', 'Combined']
results_list = [results_hashtag, results_tfidf, results_combined]

for idx, (method, result) in enumerate(zip(methods, results_list)):
    ax = axes[idx]
    
    # Extract silhouette scores across runs
    silhouettes = [r['silhouette'] for r in result['runs']]
    runs = range(1, len(silhouettes) + 1)
    
    ax.plot(runs, silhouettes, 'o-', markersize=10, linewidth=2, color=colors[idx])
    ax.axhline(y=np.mean(silhouettes), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(silhouettes):.4f}')
    
    ax.set_xlabel('Run Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title(method, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(runs)

plt.tight_layout()
plt.savefig('clustering_runs_consistency.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: clustering_runs_consistency.png")
plt.close()

# ============================================================================
# Figure 3: Performance-Efficiency Trade-off
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot: Time vs Silhouette
for idx, (method, result, color) in enumerate(zip(methods, results_list, colors)):
    times = [r['time'] for r in result['runs']]
    silhouettes = [r['silhouette'] for r in result['runs']]
    
    ax.scatter(times, silhouettes, s=200, alpha=0.6, color=color, 
               edgecolors='black', linewidth=2, label=method)
    
    # Add mean marker
    ax.scatter(result['avg_time'], result['avg_silhouette'], 
               s=400, marker='*', color=color, edgecolors='black', linewidth=2)

ax.set_xlabel('Computation Time (seconds)', fontsize=13, fontweight='bold')
ax.set_ylabel('Silhouette Score', fontsize=13, fontweight='bold')
ax.set_title('Performance-Efficiency Trade-off\n(★ = mean across runs)', 
             fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_efficiency_tradeoff.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: performance_efficiency_tradeoff.png")
plt.close()

# ============================================================================
# Compute Performance Improvements
# ============================================================================
print("\n[Computing Performance Improvements]")

hashtag_sil = comparison_df.loc[0, 'Avg Silhouette']
tfidf_sil = comparison_df.loc[1, 'Avg Silhouette']
combined_sil = comparison_df.loc[2, 'Avg Silhouette']

improvement_over_hashtag = ((combined_sil - hashtag_sil) / hashtag_sil) * 100
improvement_over_tfidf = ((combined_sil - tfidf_sil) / tfidf_sil) * 100

print("\n" + "="*80)
print("PERFORMANCE IMPROVEMENTS")
print("="*80)
print(f"Combined vs Hashtag-Only: {improvement_over_hashtag:+.2f}%")
print(f"Combined vs TF-IDF-Only:  {improvement_over_tfidf:+.2f}%")
print("="*80)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("PHASE 7 COMPLETE!")
print("="*80)
print("Generated files:")
print("  - clustering_comparison.csv")
print("  - clustering_comparison.tex")
print("  - clustering_comparison.png")
print("  - clustering_runs_consistency.png")
print("  - performance_efficiency_tradeoff.png")
print("="*80)
print("\n✓ Ready for Phase 8: Statistical significance testing\n")
