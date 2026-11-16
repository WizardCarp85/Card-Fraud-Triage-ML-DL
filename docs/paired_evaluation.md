# Final paired evaluation

## Current execution status

The comparison implementation is ready for frozen test exports. Actual paired results are pending the AE validation threshold manifest owned by Yash in issue #62 and the selected AE checkpoint's test score export. Issue #38 still refers to #46, which resolves to an older split PR. The external creditcard.csv is also absent from the implementation environment, so AE test inference has not been run here.

No experimental paired metrics or plots are committed by this change. Synthetic fixtures exercise the evaluation code. Complete the final execution and commit its artifacts before closing issue #38.

## Execution after the prerequisites arrive

From the repository root, use the selected checkpoint's shared-contract test score CSV and the AE threshold manifest conforming to docs/threshold_manifest_schema.md.

```sh
python -m scripts.evaluate_frozen_pair \
  --ae-scores data/scores/autoencoder_test.csv \
  --ae-thresholds results/phase2/autoencoder/threshold_manifest.json
```

The AE path arguments are explicit inputs. These example paths will exist after their producers export the files. The threshold manifest model must be autoencoder and its provenance must identify the frozen checkpoint from results/ae_frozen_model_manifest.json. Inference must reload that checkpoint, use the frozen preprocessing, run in eval mode with gradients disabled, and preserve raw test IDs. Use the existing inference implementation; training and candidate selection have already concluded.

The runner verifies both policy manifests and their artifact hashes before loading test scores. It checks selected AE checkpoint identity, feature order, shared split identity, frozen test membership, score direction and one-to-one label alignment. It verifies the existing GMM test export against the #30 provenance. It applies only saved thresholds and never calls threshold selection or model fitting. Fixed thresholds flag ties together. Top-K ties follow lexical stable row ID, as in the shared contract.

The output directory defaults to results/phase2/paired and contains ranking.csv, thresholds.csv, triage.csv, overlap.csv, precision_recall.png, provenance.json and interpretation.md. AP is primary and ROC-AUC secondary. Threshold tables show validation budget and realized test FPR/recall/precision, frauds caught/missed and false alarms. The capacity tables report 100, 500 and 1,000 reviews. The overlap table identifies common and unique fraud catches, including unique IDs. Provenance links scores, policies, labels and test membership by hash.

A no-alert frozen policy produces zero alerts and retains its explicit status. Undefined policies stop execution. The historical holdout was examined during Phase 1. Describe the final comparison with that prior exposure disclosed.

## Completion procedure

1. Merge the validation-only AE threshold freeze from #62.
2. Generate test scores from the already selected checkpoint using the frozen test IDs, record checkpoint/scaler provenance, and commit the score export.
3. Run the command above once all choices are frozen.
4. Review the saved outputs, commit them with the interpretation, and mark the paired-evaluation PR ready for review. Model or threshold changes must follow a separately documented protocol, without selection against these test results.
