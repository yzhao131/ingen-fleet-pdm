# Week 03 Log — Windowed Feature Engineering, Leakage Control & Signal Analysis

**Week:** October 12–17, 2026  
**Project:** Synthetic Aido Rover Fleet Predictive Maintenance

## Work completed

This week, I connected the revised causal fleet simulator to the existing Week 2 cleaning and robot-day modeling pipeline. To preserve reproducibility, I kept the new cleaned data and processed modeling frame under `data/causal_preview/`, separate from the earlier outputs. The refreshed dataset includes 109,124 cleaned telemetry observations, 46,327 mission records, 273 faults, and 133 maintenance events. The modeling frame contains 26,145 robot-days across chronological train, validation, and test splits.

I then ran the Week 3 feature pipeline and produced 67 engineered features, covering recent usage, telemetry condition, historical events, and robot context. The feature dictionary was exported for review. I also ran a historical recomputation audit: 45 sampled robot-days generated 3,015 feature checks, all of which passed. This provides evidence that sampled feature calculations agree with a historical-only recomputation, although it does not rule out every possible upstream timestamp or label leakage issue.

## Signal analysis and interpretation

Using the training split, I analyzed 518 matched positive/control pairs across 68 numeric fields. Thirty-eight fields met the exploratory Benjamini–Hochberg FDR threshold of 0.05. Charge-cycle features, unit age, battery capacity, wheel wear, and motor current showed some of the strongest differences. The direction of these results is consistent with the intended synthetic degradation process. I also generated train-to-validation coverage and standardized mean-shift summaries to identify features whose distributions change over time.

One result that challenged my initial expectations was that cumulative charge cycles and robot age ranked above motor-current signals. I expected motor current to be among the strongest early indicators because bearing degradation directly affects it in the simulator. This suggests that cumulative exposure and fleet aging can dominate a matched comparison even when a more immediate condition signal is present. It also highlights why statistical significance alone is not sufficient for selecting useful predictive features.

## Risks and next steps

The seven-day outcome windows overlap, so neighboring observations are correlated and the reported p-values should remain exploratory. The analysis does not yet establish independent event-level lead time, cross-site generalization, or the absence of all upstream leakage. Before model training, I plan to review exact feature definitions against the source code, investigate age-related confounding, validate timestamp boundaries, and examine event-aligned signal trajectories. The sealed evaluation data remains untouched.
