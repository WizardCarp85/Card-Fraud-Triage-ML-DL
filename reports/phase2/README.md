# Phase 2 technical report

Paste all of main.tex into an Overleaf project's main.tex and select pdfLaTeX. Compile twice to resolve references. All diagrams, PR curve coordinates, tables and references are embedded, so no additional files are needed. The manuscript uses IEEEtran conference mode with two text columns. Its title and the wide policy table span both columns.

The companion main.pdf is the locally compiled, visually checked version. Local build command from the repository root is tectonic reports/phase2/main.tex. Standard LaTeX with pdflatex also supports the source.

The report is based on repository snapshot 252197f, after the final paired comparison was merged. Overleaf's shared project could not be read through the web tool. Team and supervisor details follow the repository's Phase 1 report. The two-column formatting follows the supplied screenshot.

## Evidence map

| Report content | Authoritative source |
|---|---|
| Source counts, cleaned counts, partition sizes | data/split_manifest.json |
| Feature order and scaler identity | data/artifacts/feature_metadata.json |
| GMM parameters, baseline validation AP | data/scores/gmm_baseline_provenance.json |
| AE dimensions, selected seed and checkpoint | results/ae_frozen_model_manifest.json and src/models/autoencoder.py |
| Stage 1 AP, precision and best epochs | results/phase2/ae_run_log.csv |
| Stage 2 means and standard deviations | results/phase2/ae_run_log.csv and results/ae_seed_stability.md |
| Model-specific validation policies | results/phase2/gmm/threshold_manifest.json and results/phase2/autoencoder/threshold_manifest.json |
| Final AP and ROC-AUC | results/phase2/paired/ranking.csv |
| Frozen test operating points | results/phase2/paired/thresholds.csv |
| Review capacities, counts and overlap | results/phase2/paired/triage.csv and results/phase2/paired/overlap.csv |
| Embedded PR curve | data/scores/gmm_test.csv, data/scores/autoencoder_test.csv and data/scores/test_labels.csv |
| Inference identity, environment and numerical equivalence | results/phase2/paired/inference_provenance.json |
| Current full-suite evidence | reports/phase2/full_suite_test_log.txt, command python -m pytest -q, Python/package versions in log |
| Individual contributions | Repository issues #25–#43 and #62, with the merged implementation |

The PR curves retain endpoints of constant-recall segments from sklearn.metrics.precision_recall_curve. This embeds the empirical curve without a 42,560-row LaTeX coordinate list. AP is taken from the committed ranking CSV, without integrating the plotted line.

Earlier planning notes and the methods note include conflicting architecture descriptions. The manuscript uses the implemented 29-input topology, BatchNorm/LeakyReLU backbone, noise standard deviation 0.5, and the frozen seed-7 checkpoint. It distinguishes historical test exposure and deferred hybrid work explicitly.

The faculty-suggested autoencoder paper and the denoising paper are cited as background. The study's quantitative results are cited to repository artifacts, with no transfer of performance claims from external studies.

## Current testing evidence

The report revision includes a fresh full-suite run against main snapshot 4f83b78 with the report branch changes. The command is python -m pytest -q from the repository root, with the ULB CSV available at data/creditcard.csv and requirements-data.txt installed. The complete captured output and package versions are in full_suite_test_log.txt.

The result is 105 passed, three skipped and one failure. The three historical AE baseline checks skip because their run directory is absent. The failure checks the inference-code file hash. Workflow integration changed src/models/autoencoder.py after the paired score receipt was frozen, so the current file identity differs from the receipt. This report revision records that unresolved verification limitation and supersedes its former 98-pass summary. The experiment tables continue to use the original paired artifacts at 252197f.

The source and PDF are tracked on PR #71 at reports/phase2/main.tex and reports/phase2/main.pdf. They become available on the default branch when the report PR is merged. This revision adds exact time boundaries and frozen boundary-row assignments, the complete workflow diagram, the AP rationale, explicit external-validation/grouping/cost/deployment limitations and a capacity-dependent practical comparison.
