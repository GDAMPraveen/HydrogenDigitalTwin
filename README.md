# Uncertainty-Aware Explainable Digital Twin for Resilient Green Hydrogen Production (PEM Electrolysis)

A modular, physics-informed AI digital twin for Proton Exchange Membrane (PEM) electrolyzers. Every incoming sensor packet is wrapped in a uniquely tagged **`TwinFrame`** and processed sequentially through five synchronized layers—**Physics → Hybrid Forecasting → Resilience & OOD → Uncertainty Quantification → Local XAI**—ensuring alerts, intervals, and explanations always correspond to the exact same operating state.

---

## Architecture

```text
Sensor Packet (frame_id, step_index, raw_sensors)
   └─► 1. Physics Engine        (Faraday's law, DC power, mass/thermal/water/gas balance)
   └─► 2. Hybrid Forecaster     (Physics-Residual HistGradientBoosting / Ridge, bounded >= 0)
   └─► 3. Resilience Layer      (Live sensor-fault detection + Isolation Forest OOD)
   └─► 4. Uncertainty Layer     (Adaptive Aleatoric CQR intervals + Epistemic OOD score)
   └─► 5. Local XAI Layer       (Per-frame root-cause attribution + LOEO permutation importance)
   └─► Synchronized TwinFrame   (Streamlit Dashboard & audit logs)
```

---

## Key Validated Results (12-Fold Leave-One-Experiment-Out on NLR Wind Datasets)

* **Feature Ablation (`no_history` vs. `full`):** Removing lag/rolling target history improved Ridge LOEO Macro MAE by **14.0%** (`0.1921` vs. `0.2234 kg/hr`) and stabilized the high-wind outlier fold (`windIEA3.4-9ms-A_1-400.csv`).
* **Physics–AI Hybrid vs. Pure AI:** Training-only Faraday calibration converged to $k_{\text{eff}} \approx 196.2 - 196.5$ across all 12 folds. Anchoring `HistGradientBoosting` to Faraday's law (`physics_residual`) outperformed Pure AI on every multi-step horizon with **0 physically inadmissible (<0) predictions**:

| Horizon | Hybrid MAE (kg/hr) | Pure AI MAE (kg/hr) | Persistence MAE (kg/hr) | Hybrid Gain vs. Pure AI | Neg. Preds (Hybrid) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 step** | 0.1920 *(Ridge)* / 0.1982 *(HGB)* | 0.2017 *(HGB)* | 0.2320 | -1.7% *(HGB)* | **0** |
| **3 steps** | **0.3300** | 0.3387 | 0.5669 | **-2.6%** | **0** |
| **5 steps** | **0.4263** | 0.4451 | 0.8853 | **-4.2%** | **0** |
| **10 steps** | **0.7711** | 0.8243 | 1.5973 | **-6.5%** | **0** |

* **Uncertainty & OOD Coupling:** Split Conformal Prediction achieved **88.0%–90.1% mean LOEO coverage** (90% target) across horizons 1–10. Isolation Forest independently flagged `windIEA3.4-9ms-A_1-400.csv` #1 at **57.4% OOD**, catching the exact regime where conformal coverage dropped to 45%–52%.
* **Sensor-Fault Resilience:** Voltage faults had minimal impact (`0.198 → 0.242` MAE), whereas missing/stuck stack current caused a 10×–15× error spike (`2.07–3.11` MAE), motivating live current-sensor health checks.
* **Explainability (XAI):** LOEO Permutation Importance confirmed the hybrid model relies primarily on the live **Faraday physics residual** (`+1.36 kg/hr`), followed by **Wind Turbine Power** (`+0.094`) and **PT307 Pressure** (`+0.072`).

---

## Repository Structure

```text
HydrogenDigitalTwin/
├── app.py                              # Integrated Streamlit Digital Twin Dashboard
├── src/
│   ├── physics/                        # 7 validated PEM physics modules (Faraday, thermal, gas, water, etc.)
│   ├── integration/                    # PEMPhysicsEngine + HybridPEMForecaster
│   ├── forecasting/                    # Data prep, feature engineering, multi-horizon pipeline
│   ├── uncertainty/                    # Split Conformal Prediction & Adaptive CQR
│   ├── resilience/                     # Isolation Forest OOD & sensor-fault injection
│   └── digital_twin/                   # Tagged TwinFrame contract & sequential DigitalTwinOrchestrator
├── scripts/                            # Reproducible LOEO ablation, hybrid, uncertainty, resilience & XAI benchmarks
├── tests/                              # 105 unit & integration tests (pytest)
└── reports/                            # Saved LOEO benchmark CSVs & validation notes
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full test suite (105 tests)
python -m pytest

# 3. Launch the Digital Twin Dashboard
streamlit run app.py
```

### Reproducing Benchmarks

```bash
python -m scripts.ridge_feature_ablation
python -m scripts.run_hybrid_loeo --horizons 1 3 5 10
python -m scripts.run_uncertainty_loeo --horizons 1 3 5 10 --alpha 0.10
python -m scripts.run_resilience_benchmark --horizon 1
python -m scripts.run_xai_analysis --horizon 1 --repeats 5
```