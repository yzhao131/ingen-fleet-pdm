# Aido Rover Synthetic Fleet Data Contract

## Dataset Overview

This synthetic dataset represents 50 Aido Rover units operating across
7 sites from 2025-01-01 through 2026-06-30.

All timestamps are stored in UTC.

The dataset contains five operational tables:

1. `robots.csv` - fleet and site information
2. `telemetry.csv` - time-indexed robot sensor measurements
3. `missions.csv` - robot operating activity
4. `faults.csv` - fault occurrence and logging records
5. `maintenance.csv` - maintenance periods