# Week 02 — Data Cleaning and Modeling-Frame Decisions

**Period:** October 5–10, 2026  
**Track:** Predictive Maintenance & Fleet Forecasting — Aido Rover  
**Author:** Yang Zhao

## Decision and rationale

The most consequential Week 2 decision was how to exclude robot-days affected by maintenance. My initial implementation removed a robot-day whenever any part of the calendar day overlapped a maintenance interval. Although conservative, this definition could use information that was not available when a prediction was issued at 00:00 UTC. For example, a robot scheduled for maintenance at 15:00 on day T would be removed from the dataset even though the prediction at midnight could not necessarily know that later intervention would occur. This introduces a risk of future-information selection bias and makes the offline population less representative of daily scoring.

I therefore changed the eligibility rule to exclude a robot only when maintenance is **active at the prediction timestamp**: `start_time <= T < end_time`. An independent comparison of the two rules found that the old rule excluded 329 robot-days, whereas the corrected rule excluded 168; the extra 161 exclusions were attributable only to the old calendar-day definition. The final dataset contains 26,135 eligible robot-days after maintenance exclusions, incomplete outcome windows, and chronological split gaps. Of these, 1,231 are positive for a severity-3-or-higher fault in the next seven days (4.71%). The train, validation, and test periods contain 700, 314, and 217 positive robot-days respectively. Label reconstruction reported zero mismatches, and the final QA suite reported zero duplicate keys, maintenance-at-prediction overlaps, invalid labels, or split-boundary violations.

Other cleaning decisions prioritized traceability over aggressive filtering. I removed 200 exact duplicate telemetry records, corrected fixed clock offsets for eight robots, and retained an audit of original versus corrected timestamps. The independent clock-drift check confirmed eight affected robots before correction and none afterward. I flagged 2,906 telemetry records as potential IQR outliers instead of deleting them, since elevated motor current may itself be a meaningful pre-failure signal. I also identified 82 missing scheduled telemetry readings; 38 coincided with mission activity and were classified as possible connectivity gaps, while 44 remained unknown. I did not equate missing readings with confirmed robot downtime.

## Implications and next steps

The cleaned modeling frame preserves a seven-day forward label window and fixed chronological splits separated by seven-day gaps. Descriptive pre-fault analysis showed average motor current of 4.266 A for positive robot-days versus 4.029 A for negatives, but this association is not yet validated predictive performance. In Week 3, I will engineer lagged and trend features with strict timestamp cutoffs, test leakage explicitly, and use validation data rather than the test period to make modeling decisions. Before final repository submission, I will confirm the required Parquet export and statistical-power workbook are present.
