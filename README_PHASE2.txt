HYDROGEN DIGITAL TWIN — PHASE 2 FORECASTING COMPLETION BUNDLE

CONTENTS
- src/forecasting/: complete compatible forecasting module replacements
- tests/forecasting/: automated unit and synthetic end-to-end tests
- scripts/run_forecasting_experiments.py: multi-horizon training and evaluation
- This README: installation and execution order

IMPORTANT
1. Back up your existing src/forecasting files before replacing them.
2. The source files in this bundle are complete replacements, not patches.
3. The script assumes the cleaned NLR file exists at:
   data/processed/nlr/nlr_observation_clean.csv
4. It excludes characterization_200.csv from the wind experiment track.
5. It excludes H2E_f_TT641_Temp, data bookkeeping fields, diagnostic PSU
   power fields, Efficiency and calculated production-rate leakage candidates.
6. Forecast horizons are row-based steps. Do not call them seconds or minutes
   until the wind dataset sampling interval is verified.
7. joblib artifacts must only be loaded from trusted local sources.

EXECUTION ORDER (Git Bash, from project root)
1. Activate environment:
   source .venv/Scripts/activate

2. Back up existing source:
   cp -r src/forecasting src/forecasting_backup_before_phase2_completion

3. Copy/merge this bundle's src/forecasting files into project src/forecasting.
   Copy tests/forecasting into project tests/forecasting.
   Copy scripts/run_forecasting_experiments.py into project scripts/.

4. Install/check dependencies:
   python -m pip install -U pytest scikit-learn pandas numpy joblib
   Optional: python -m pip install xgboost lightgbm catboost

5. Compile:
   python -m compileall src/forecasting tests/forecasting scripts/

6. Run tests:
   python -m pytest tests/forecasting -q

7. Run sklearn + optional models, chronological and LOEO evaluation:
   python scripts/run_forecasting_experiments.py --mode both

   This can be computationally expensive (multiple models, 12 LOEO folds,
   and four horizons). For an initial faster run, use:
   python scripts/run_forecasting_experiments.py --mode chronological

   To omit optional gradient boosting libraries:
   python scripts/run_forecasting_experiments.py --mode both --skip-optional

OUTPUTS
- reports/forecasting/chronological_<horizon>_step.csv
- reports/forecasting/loeo_<horizon>_step_by_experiment.csv
- reports/forecasting/loeo_<horizon>_step_summary.csv
- models/forecasting/nlr_<horizon>_step_<model>.joblib (best chronological
  model for each horizon, only when the best candidate is an ML model)

SCIENTIFIC NOTES
- Chronological evaluation is an operational within-experiment test.
- LOEO evaluates generalization to an unseen experiment.
- The LOEO score is the average of per-experiment fold metrics in the summary.
- Training-only imputers are inside each model pipeline.
- Raw sensor values at row t predict the target at row t+h. This assumes those
  sensor values are available when the forecast is issued.
- The current measured target at row t is included as a valid persistence reference if available at issue time; lag and rolling features use values no later than t-1.
- No sampling-time interpretation is made.
- Performance must be reported from actual terminal outputs, not predicted.
