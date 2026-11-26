# NLR Physics Layer — Validation Notes

## Dataset
- Source: characterization_200.zip
- Records: 18,000
- Recording duration: approximately 5 hours
- Sampling interval: 1 second
- Primary measured hydrogen output: IVAL_f_FM011_Flow (kg/hr)

## 1. Electrical consistency

Electrical consistency was investigated using:

P_calculated = Voltage × Actual_Current

P_calculated_kW =
    Power Supply Average Voltage (Vdc)
    × H2E_f_PSU_A_Current / 1000

The calculated value was compared with:
H2E_n_PSU_A_Power.

### Startup period
First 121 records, elapsed time 0–120 seconds.

- Mean reported power: 1181.776860 kW
- Mean V × I power: 1155.254584 kW
- Mean power error: +26.522275 kW
- Mean absolute power error: 27.461602 kW
- Mean relative power error: 2.336145%

### Steady period
Remaining 17,879 records.

- Mean reported power: 635.372784 kW
- Mean V × I power: 623.898724 kW
- Mean power error: +11.474059 kW
- Mean absolute power error: 11.474220 kW
- Mean relative power error: 1.660024%
- Maximum absolute power error: 48.703 kW

### Interpretation
The discrepancy persists beyond startup.
The measurement boundaries, averaging, calibration, and PSU A/B
tag mapping require further verification.

Both power channels must be retained.
No correction factor or anomaly threshold has been established.

## 2. Hydrogen production-rate comparison

Measured flow:
IVAL_f_FM011_Flow

Calculated production-rate channel:
H2E_f_Elec_CalcProdRate

The two channels have a high overall correlation (0.997581),
but large discrepancies occur during the initial transient.

The calculated production-rate channel has not been independently
verified from source documentation.

Decision:
- Retain for investigation.
- Do not use as an independent measured output.
- Exclude from initial physics predictors and Faraday-law validation.

## 3. Startup behaviour

The first 120 seconds show substantial variation in current,
measured flow, and calculated production rate.

The measured flow approaches approximately 22 kg/hr during
the sampled startup period.

Possible explanations such as sensor response delay,
initialization behaviour, and calculation transients remain
hypotheses and have not been established.

Do not automatically remove startup observations.

## 4. Physics modelling decisions

- Use actual stack current, not current-command signals, for
  Faraday-law calculations.
- Retain voltage and both power channels.
- Use measured hydrogen flow as the observed output.
- Do not assume a stack cell count or Faradaic efficiency.
- Do not assume NORCE stack parameters apply to NLR.
- Do not establish physical thresholds without documentation
  or validated operating evidence.

## 5. Completed validation scripts

Source code:
- src/physics/data_validation/nlr_characterization_validation.py
- src/physics/data_validation/nlr_startup_diagnostic.py
- src/physics/data_validation/nlr_power_error_analysis.py

Generated reports:
- nlr_timestamp_validation.csv
- nlr_power_consistency.csv
- nlr_production_rate_validation.csv
- nlr_physics_validation_summary.csv
- nlr_startup_120s_diagnostic.csv
- nlr_startup_diagnostic_summary.csv
- nlr_power_error_period_analysis.csv

## 6. Current-command versus actual-current tracking validation

Source signals:
- H2E_f_NLR_CurrentCmd (A): commanded current setpoint (A).
- H2E_f_NLR_CurrentCmd: second numerical representation of the
  command signal; numerically related to the ampere command by
  a factor of 30. Its exact metadata meaning remains unverified.
- H2E_f_PSU_A_Current: measured actual stack current (A).

Tracking error:
actual current - commanded current.

Dataset:
- 18,000 observations.
- Approximately one-second sampling.
- 10 distinct command levels from 3000 A to 300 A.
- 9 downward command transitions, with no upward transitions.

Overall tracking performance:
- Mean tracking error: -0.892778 A.
- Mean absolute tracking error: 1.906222 A.
- Maximum absolute tracking error: 2140 A.

Period-level tracking performance:

| Period | Samples | Mean error (A) | MAE (A) | Maximum absolute error (A) |
|---|---:|---:|---:|---:|
| 0-120s | 121 | -151.975207 | 152.636364 | 2140 |
| 121-600s | 480 | 0.002083 | 0.581250 | 3 |
| 601-1800s | 1200 | -0.016667 | 0.671667 | 3 |
| 1801-3600s | 1800 | 0.076667 | 0.700000 | 164 |
| >3600s | 14399 | 0.152788 | 0.937426 | 301 |

Startup behaviour:
- Actual current falls substantially below the 3000 A command.
- The largest absolute tracking error is 2140 A.
- Startup tracking behaviour is distinct from subsequent operation.

Downward command transitions:
- All 9 transitions reduce the command by 300 A.
- At each transition sample, actual current remains close to
  the preceding setpoint, producing approximately +300 A error.
- At the following one-second sample, actual current is close
  to the new setpoint.
- This is an observed one-sample transition discrepancy at the
  available sampling resolution, not proof of the controller's
  exact physical response time.
- No upward transitions occur in this experiment; upward
  response cannot be assessed.

Modelling decisions:
- Use measured actual current for Faraday-law calculations.
- Retain commanded current as a separate controller input.
- Retain both command representations pending metadata clarification.
- Tracking error can be retained as a diagnostic feature.
- Preserve startup and transition observations; do not
  automatically discard them.
- Do not interpret the observed transition pattern as a
  universal controller characteristic.

## 7. Additional current-tracking validation scripts

Source code:
- src/physics/data_validation/nlr_current_tracking_analysis.py

Generated reports:
- nlr_current_tracking_period_summary.csv
- nlr_current_tracking_transitions.csv
- nlr_current_tracking_validation_summary.csv
