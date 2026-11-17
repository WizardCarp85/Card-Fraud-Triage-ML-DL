# Final paired evaluation

The GMM freeze from #37 and AE freeze from #62 are merged. This evaluation uses the selected RUN-S2-W16-DNZ-S7 checkpoint and the two committed threshold manifests. Issue #38's older #46 dependency reference resolves to a split PR.

## Reproduction

Run from the repository root with requirements-data.txt and matplotlib installed. Supply the ULB CSV whose SHA-256 matches data/split_manifest.json.

```sh
python -m scripts.score_frozen_test --data /path/to/creditcard.csv
python -m scripts.evaluate_frozen_pair \
  --ae-scores data/scores/autoencoder_test.csv \
  --ae-thresholds results/phase2/autoencoder/threshold_manifest.json
```

Frozen scoring runs on CPU with PyTorch 2.8.0, scikit-learn 1.6.1 and the remaining pinned data dependencies under Python 3.12.14. Exact package versions and code/input hashes are in results/phase2/paired/inference_provenance.json. Existing AE exports are checked against a repeated inference within rtol=1e-6 and atol=1e-8 before writing. This verification repeats inference only.

Both models score identical raw test row IDs using the shared feature builder and their respective frozen scalers. The resulting feature matrices match exactly. Regenerated GMM scores agree with the committed #30 export within a maximum absolute difference of 2.85e-14, so that verified export is retained for evaluation. The AE checkpoint is reloaded into the frozen architecture and scored through AutoencoderAnomalyModel.score in eval mode with gradients disabled. Labels remain outside model inputs.

The runner verifies model/scaler and validation artifact hashes, the selected AE checkpoint, feature order, shared split identity, test membership and score-to-label alignment. The inference receipt connects the actual score exports to both policy manifests. It applies saved thresholds without calibration or fitting. Tied scores move together for fixed thresholds. Top-K ties follow lexical stable row ID.

## Results and artifacts

The test partition contains 42,560 transactions and 52 frauds, a prevalence of 0.1222%. It was previously examined in Phase 1. Interpret this paired comparison with that prior exposure disclosed.

| Measure | GMM | Autoencoder |
|---|---:|---:|
| Average precision | 0.677302 | 0.486678 |
| ROC-AUC | 0.906694 | 0.919871 |
| Frauds caught in 100 reviews | 38 | 40 |
| False alarms in 100 reviews | 62 | 60 |
| Frauds caught in 500 reviews | 39 | 40 |
| Frauds caught in 1,000 reviews | 41 | 41 |

GMM has higher AP across the ranking. The AE's 100-case queue includes all 38 GMM catches and two additional frauds, raw IDs 272521 and 274475. This capacity-specific result does not establish improvement at every operating point. At the frozen 0.1% validation FPR budget, GMM catches 33 frauds with four false alarms and AE catches 24 with 14. At the 1% budget, GMM catches 38 with 69 false alarms and AE catches 40 with 201. At the 5% budget both catch 41, with 914 GMM false alarms and 767 AE false alarms. The policy workloads differ because realized test FPR differs from validation FPR.

Every value traces to committed artifacts.

- data/scores/autoencoder_test.csv contains the selected AE's shared-contract test export.
- data/scores/gmm_test.csv is the verified GMM export.
- results/phase2/paired/ranking.csv and precision_recall.png show AP, secondary ROC-AUC and the prevalence baseline.
- thresholds.csv reports frozen thresholds, validation budgets, realized test FPR/recall/precision, caught/missed fraud and false alarms.
- triage.csv and overlap.csv show capacity results, shared catches and unique fraud IDs.
- inference_provenance.json and provenance.json record input/model/code hashes and the selected inference environment.
- interpretation.md provides a concise results narrative.

The threshold and model artifacts retain their frozen identities. A single fraud moves recall by about 1.92 percentage points on this 52-fraud test partition. Both models miss 11 frauds at the 1,000-case capacity.
