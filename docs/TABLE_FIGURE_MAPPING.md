# Table and number mapping

Each entry gives the script that produces the numbers and the file it writes.
Scripts run inside the dataset folder that holds the pipeline outputs. Files
marked *(local)* are produced by the script but are not yet in `results/`; add
them after running the script.

| Paper item | Content | Script | Output file |
|---|---|---|---|
| Table 1 | Split sizes and periods | `pipeline/phase1_data_prep.py` (printed); `scripts/revision/revision_diagnostics.py` (Election 2020); `scripts/revision/covid19_revision.py --no-ablation` (COVID-19) | `results/election2020/diagnostics_summary.json`; `covid19_revision/split_stats.csv` *(local)* |
| Section 4.2 | Vocabulary sizes (16,841 and 897); tweets without a vocabulary hashtag (4 and 1,361) | `scripts/revision/covid19_revision.py`, `scripts/revision/revision_diagnostics.py` | as Table 1 |
| Section 3.2.3 | Engagement factor: zero for 49.1% of tweets, mean 0.81, maximum 11.04 (Election 2020) | `scripts/revision/revision_diagnostics.py` | `results/election2020/diagnostics_summary.json` |
| Table 2 | K-sensitivity, Silhouette and DB (seed 42) | `pipeline/enhanced_phase6.py` | `enhanced_clustering_results.csv` *(local)* |
| Table 3 | Feature comparison at each method's best K (seed 42) | `pipeline/enhanced_phase6.py` | `enhanced_clustering_results.csv` *(local)* |
| Table 4 | Hashtag-Only vs BERT [CLS], K=20, seeds 42–46: means, times, paired t, Cohen's d | `pipeline/phase9_bert.py` (embeddings), `scripts/experiments/quick_bert_comparison_k20.py` | `bert_comparison_k20.pkl` *(local)* |
| Table 4 | Welch, Mann–Whitney, Shapiro–Wilk | `scripts/revision/table_checks.py` (both datasets); `scripts/revision/sensitivity_alpha_tau.py` (Election 2020) | `results/election2020/normality_tests.csv` |
| Table 5 | Ablation, Election 2020 (three seeds, L2-normalised) | `scripts/experiments/ablation_study.py` | `ablation_study_results_election2020.csv` *(local)* |
| Table 5 | Ablation, COVID-19 | `scripts/revision/covid19_revision.py` | `covid19_revision/ablation_covid19.csv` *(local)* |
| Table 6 | Sensitivity to α (16 settings, five seeds) | `scripts/revision/sensitivity_alpha_tau.py` | `results/election2020/sensitivity_alpha.csv`, `sensitivity_runs.csv` |
| Table 7 | Sensitivity to τ_dormant, τ_decay (20 settings) | `scripts/revision/sensitivity_alpha_tau.py` | `results/election2020/sensitivity_tau.csv`, `sensitivity_runs.csv` |
| Section 5.5 | Holm-corrected Welch tests and ANOVA for Tables 6–7 | `scripts/revision/sensitivity_stats.py` | computed from `results/election2020/sensitivity_runs.csv` |
| Section 5.5 | Deviation from uniform weights (0.014, 95th percentile 0.047); uniform (0.149 ± 0.004) and binary (0.105 ± 0.005) weights | `scripts/revision/revision_diagnostics.py` | `results/election2020/diagnostics_runs.csv`, `diagnostics_summary.json` |
| Section 6.6 | Repeated hashtag vectors (23%); Silhouette after removing them (0.091; uniform 0.089) | `scripts/revision/revision_diagnostics.py` | `results/election2020/diagnostics_runs.csv`, `diagnostics_summary.json` |
| Section 6.3 | Top hashtags per cluster | `scripts/experiments/qualitative_analysis.py` | `cluster_stats_k20_election2020.csv` *(local)* |
| Table 8 | Cross-domain summary | Derived from Tables 2–7 | — |

## Notes

- **Two Silhouette values for the same experiment.** The default setting in
  Table 6 (0.143 ± 0.007) reproduces the feature matrix of Table 4
  (0.1424 ± 0.006). They differ because `quick_bert_comparison_k20.py` draws the
  10,000-point Silhouette sample without a seed, whereas the revision scripts
  seed it with the K-means seed.
- **Standard deviations.** Table 4 and Cohen's d use the population standard
  deviation (`np.std`). The revision scripts report the sample standard
  deviation (`ddof=1`). The Election 2020 column of Table 5 uses the population
  value and the COVID-19 column the sample value. Each table caption states its
  convention.
- **`pipeline/phase8_statistics.py`** is the original testing script. It reports
  Wilcoxon signed-rank tests, which cannot reach p < 0.05 with five paired runs
  (minimum p = 0.0625). The revised paper does not use them; its tests come from
  the revision scripts listed above.
