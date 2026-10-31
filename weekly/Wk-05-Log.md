# Week 05 Log — Service Demand Forecasting & Battery Degradation

**Period:** October 26–31, 2026  
**Project:** inGen Dynamics — Predictive Maintenance for a Synthetic Robot Fleet  
**Data cutoff:** June 30, 2026 (simulation)

## Objectives and Work Completed

This week I extended the predictive-maintenance pipeline from robot-level failure classification to operational planning. I aggregated synthetic maintenance records into weekly service-event counts, excluded incomplete boundary weeks, and compared five forecasting approaches: Last Value, four-week Moving Average, four-week Seasonal Naive, Simple Exponential Smoothing, and Prophet. I evaluated the methods using seven rolling forecast origins and 12-week horizons, recording forecast errors by horizon. Last Value performed best, with a mean absolute error (MAE) of 1.02 service events per week and a weighted absolute percentage error (WAPE) of 57.33%. The relatively high WAPE highlighted the difficulty of forecasting sparse service events.

I also assessed uncertainty rather than relying solely on point estimates. Exploratory horizon-specific intervals targeting 80% coverage achieved 70.8% empirical coverage on 24 evaluated forecasts with non-overlapping target weeks. This shortfall means the intervals should be treated as planning scenarios, not calibrated guarantees.

For battery health, I constructed per-robot lifecycle segments and compared per-unit linear, per-unit exponential, and pooled linear degradation models using chronological evaluation. The per-unit linear approach achieved the lowest MAE, 0.4107 battery-capacity percentage points. I projected first crossing dates for 80% and 90% capacity thresholds, with pooled-slope fallbacks for robots with limited recent history. These are alternative policy scenarios, not additive replacement demand.

## Operational Outputs and Cross-Check

I translated the selected service forecast into a 12-week staffing plan. Under illustrative assumptions of eight labor hours per service event and 30 productive hours per technician per week, the point forecast totals 12 events and 96 technician-hours. Five weeks in August require 32 hours under the upper planning scenario, suggesting a small flexible capacity reserve rather than an automatic full-time hire. For battery procurement, I applied an assumed 30-day lead time plus a 14-day buffer and generated unit-level and monthly order schedules.

Finally, I compared historical Week 4 robot risk scores with later battery capacity for 50 robots. The Pearson correlation was -0.1783, indicating only a weak negative association. Because the validation and planning periods differ, this was a descriptive cross-check rather than a validation of a future service queue.

## Challenges and Next Steps

Key limitations include synthetic data, sparse service counts, uncertain battery crossing dates, and unverified supplier and staffing assumptions. Next, I would calibrate intervals on more independent periods, validate predictions against actual maintenance tickets, and connect current robot-risk scores to live work orders and inventory. I packaged the analysis in `W05_Forecast_Report_Final.docx` and `W05_Planning_Outputs.xlsx`.
