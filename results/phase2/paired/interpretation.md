# Paired test evaluation

The test partition was previously examined in Phase 1. Model selection and threshold calibration used validation data. This run applies the two saved policies.

GMM AP is 0.6773; AE AP is 0.4867. At 100 reviews, 38 frauds are caught by both, 0 only by GMM and 2 only by AE. See triage.csv for false alarms and recall.

GMM has the higher AP across the full ranking, while the AE catches two additional frauds in the 100-case queue. At 1,000 reviews both catch 41 of 52 frauds and generate 959 false alarms. A single fraud changes recall by about 1.92 percentage points.

At the validation-selected 0.1% FPR budget, GMM catches 33 frauds with four false alarms, while AE catches 24 with 14 false alarms. At the 1% budget, GMM catches 38 with 69 false alarms and AE catches 40 with 201. These frozen policies have different realized workloads.

thresholds.csv reports realized test FPR against each validation budget. These rates can differ under temporal shift. All reported values trace to provenance.json.
