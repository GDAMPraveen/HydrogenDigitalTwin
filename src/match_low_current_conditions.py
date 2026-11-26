import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

FEATURES = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "T-ELY-CH1",
]
TARGET = "CV-mean"
CURRENT_LIMIT = 0.5

train = pd.read_csv("data/test4_subset.csv")
test = pd.read_csv("data/test2_subset.csv")

# Match only low-current observations.
# Voltage is deliberately excluded from the matching features.
train_low = train.loc[train["PS-I-MON"] < CURRENT_LIMIT].copy()
test_low = test.loc[test["PS-I-MON"] < CURRENT_LIMIT].copy()

# Remove rows with missing matching features or target values.
train_low = train_low.dropna(subset=FEATURES + [TARGET]).reset_index(names="test4_row")
test_low = test_low.dropna(subset=FEATURES + [TARGET]).reset_index(names="test2_row")

# Robust scaling reduces the influence of extreme feature values.
scaler = RobustScaler()
X_train = scaler.fit_transform(train_low[FEATURES])
X_test = scaler.transform(test_low[FEATURES])

# Each Test2 observation is matched to its nearest Test4 observation.
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_train)
distances, indices = nn.kneighbors(X_test)

matched = test_low[["test2_row", *FEATURES, TARGET]].copy()
matched["nearest_test4_row"] = train_low.iloc[indices.ravel()]["test4_row"].to_numpy()
matched["nearest_test4_voltage_V"] = (
    train_low.iloc[indices.ravel()][TARGET].to_numpy()
)
matched["scaled_distance"] = distances.ravel()
matched["voltage_difference_V"] = (
    matched[TARGET] - matched["nearest_test4_voltage_V"]
).abs()

print("\n=== DATA COUNTS ===")
print(f"Test4 low-current rows: {len(train_low)}")
print(f"Test2 low-current rows: {len(test_low)}")

print("\n=== NEAREST-NEIGHBOUR DISTANCE DISTRIBUTION ===")
print("Distances are dimensionless RobustScaler-space distances.")
print("No cutoff is assumed to define a good match.")
print(
    matched["scaled_distance"]
    .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    .round(4)
    .to_string()
)

print("\n=== VOLTAGE COMPARISON AFTER MATCHING ON OPERATING CONDITIONS ===")
print(
    matched[["CV-mean", "nearest_test4_voltage_V",
             "voltage_difference_V"]]
    .describe(percentiles=[0.10, 0.50, 0.90])
    .round(4)
    .to_string()
)

# These voltage bands are for post-hoc diagnosis only.
# They are not used to find the nearest neighbours.
bins = [-np.inf, 0.1, 0.4, 0.7, 1.0, 1.3, np.inf]
labels = ["<0.1", "0.1–0.4", "0.4–0.7", "0.7–1.0", "1.0–1.3", ">=1.3"]
matched["test2_voltage_band"] = pd.cut(
    matched[TARGET], bins=bins, labels=labels, right=False
)

summary = matched.groupby("test2_voltage_band", observed=False).agg(
    rows=("test2_row", "size"),
    median_distance=("scaled_distance", "median"),
    p90_distance=("scaled_distance", lambda x: x.quantile(0.90)),
    mean_test2_voltage=(TARGET, "mean"),
    mean_nearest_test4_voltage=("nearest_test4_voltage_V", "mean"),
    median_voltage_difference=("voltage_difference_V", "median"),
)

print("\n=== RESULTS BY TEST2 MEASURED-VOLTAGE BAND ===")
print(summary.round(4).to_string())

matched.to_csv("data/low_current_nearest_matches.csv", index=False)
print("\nSaved row-level matches to data/low_current_nearest_matches.csv")
