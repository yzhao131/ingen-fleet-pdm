# Week 1 Log — Prediction Horizon Design

During Week 1, I focused on defining the predictive maintenance problem before
beginning model development. One of the most important design decisions was the
prediction horizon. I selected a 7-day horizon, meaning that for each robot-day,
the target is whether the robot will experience a severity 3 or higher fault
during the following seven days.

The main reason for choosing seven days is operational usefulness. A predictive
maintenance model should provide enough advance warning for technicians to
inspect a robot, prioritize maintenance, and plan resources before a serious
failure occurs. A very short horizon, such as one day, could potentially make
prediction easier because the robot may already show strong signs of failure.
However, a one-day warning may not provide enough time for the maintenance team
to respond. In addition, relying heavily on signals immediately before failure
could make the model less useful as an early-warning system.

I also considered a longer horizon, such as 14 or 30 days. A longer horizon
would give maintenance teams more planning time, but the relationship between
current telemetry and a future failure may become weaker as the horizon
increases. Longer horizons may also create more positive robot-days around each
failure and make it harder to determine whether the model is detecting a useful
precursor or simply identifying general long-term degradation.

The 7-day horizon therefore represents a compromise between predictive signal
and operational lead time. The prediction unit is one robot-day, and features
will only use information available through the day before the prediction date.
The outcome window covers the prediction date and the following six days. This
creates a one-day blackout between the feature window and the outcome window
and reduces the risk of using information that would not have been available
when the prediction was made.

This decision also affects evaluation. Chronological train, validation, and test
splits will be used instead of random cross-validation, with gaps of at least
seven days between splits to reduce leakage across outcome windows.

Overall, the 7-day horizon provides a practical starting point for the project.
Later analysis can compare alternative horizons, such as 3, 14, or 30 days, to
understand the trade-off between early warning, predictive performance, and
maintenance usefulness.