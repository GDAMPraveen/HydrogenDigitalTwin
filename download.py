from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import shutil

archive = Path("C:/Users/PRAVEEN/Downloads/HydrogenDigitalTwin_CrossDataset_Final.zip")
root = Path.cwd()

if not archive.is_file():
    raise SystemExit(f"Archive not found: {archive}")

# Map archive contents to the project structure.
mapping = {
    "source_packages/nlr/processed/nlr_observation_clean.csv":
        "data/processed/nlr/nlr_observation_clean.csv",
    "source_packages/nlr/processed/nlr_minute_candidate.csv":
        "data/processed/nlr/nlr_minute_candidate.csv",
    "source_packages/nlr/metadata/nlr_feature_roles.csv":
        "data/metadata/nlr/nlr_feature_roles.csv",

    "source_packages/norce/NORCE_processed_and_metadata/processed/norce_test2_processed.csv":
        "data/processed/norce/norce_test2_processed.csv",
    "source_packages/norce/NORCE_processed_and_metadata/processed/norce_test3_processed.csv":
        "data/processed/norce/norce_test3_processed.csv",
    "source_packages/norce/NORCE_processed_and_metadata/processed/norce_test4_processed.csv":
        "data/processed/norce/norce_test4_processed.csv",
    "source_packages/norce/NORCE_processed_and_metadata/metadata/norce_data_quality_report.csv":
        "data/metadata/norce/norce_data_quality_report.csv",
    "source_packages/norce/NORCE_processed_and_metadata/metadata/norce_eda_decisions.txt":
        "data/metadata/norce/norce_eda_decisions.txt",
    "source_packages/norce/NORCE_processed_and_metadata/metadata/norce_feature_roles.csv":
        "data/metadata/norce/norce_feature_roles.csv",

    "source_packages/meteosat/processed/meteosat_observation_clean.csv":
        "data/processed/meteosat/meteosat_observation_clean.csv",
    "source_packages/meteosat/metadata/meteosat_visual_eda_findings.txt":
        "data/metadata/meteosat/meteosat_visual_eda_findings.txt",
    "source_packages/meteosat/metadata/meteosat_data_quality_report.csv":
        "data/metadata/meteosat/meteosat_data_quality_report.csv",
    "source_packages/meteosat/metadata/meteosat_feature_roles.csv":
        "data/metadata/meteosat/meteosat_feature_roles.csv",
    "source_packages/meteosat/metadata/meteosat_eda_decisions.txt":
        "data/metadata/meteosat/meteosat_eda_decisions.txt",

    "source_packages/pvdaq/processed/pvdaq_system_catalogue_processed.csv":
        "data/processed/pvdaq/pvdaq_system_catalogue_processed.csv",
    "source_packages/pvdaq/metadata/pvdaq_data_quality_report.csv":
        "data/metadata/pvdaq/pvdaq_data_quality_report.csv",
    "source_packages/pvdaq/metadata/pvdaq_selection_readiness.csv":
        "data/metadata/pvdaq/pvdaq_selection_readiness.csv",
    "source_packages/pvdaq/metadata/pvdaq_eda_decisions.txt":
        "data/metadata/pvdaq/pvdaq_eda_decisions.txt",
    "source_packages/pvdaq/metadata/pvdaq_feature_roles.csv":
        "data/metadata/pvdaq/pvdaq_feature_roles.csv",
}

assessment_files = [
    "cross_dataset_overview.csv",
    "nlr_experiment_wise_correlations.csv",
    "meteosat_daytime_irradiance_summary.csv",
    "cross_dataset_compatibility_matrix.csv",
    "norce_correlation_matrices_long.csv",
    "meteosat_daytime_irradiance_correlations.csv",
    "export_manifest.csv",
    "HydrogenDigitalTwin_CrossDataset_EDA_Report.txt",
    "cross_dataset_variable_inventory.csv",
    "norce_cv_mean_distribution_summary.csv",
    "pvdaq_catalogue_assessment.csv",
]

for name in assessment_files:
    mapping[f"cross_dataset_assessment/{name}"] = (
        f"reports/cross_dataset_assessment/{name}"
    )

# Verify archive members and destination collisions before changing anything.
with ZipFile(archive) as zf:
    members = set(zf.namelist())
    missing = [name for name in mapping if name not in members]
    if missing:
        raise SystemExit("Missing archive entries:\n" + "\n".join(missing))

destinations = [root / dest for dest in mapping.values()]
collisions = [str(p) for p in destinations if p.exists()]
if collisions:
    raise SystemExit(
        "STOPPED: These destination files already exist; nothing was extracted:\n"
        + "\n".join(collisions)
    )

# Extract to a temporary location, then copy only the selected files.
with TemporaryDirectory(prefix="hdt_eda_") as temp:
    temp = Path(temp)
    with ZipFile(archive) as zf:
        for source in mapping:
            zf.extract(source, temp)

    for source, destination in mapping.items():
        target = root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temp / source, target)
        print(f"ADDED: {destination}")

print(f"\nSUCCESS: Added {len(mapping)} files.")
print("Existing raw datasets, models, scripts, and ZIP backups were not modified.")
