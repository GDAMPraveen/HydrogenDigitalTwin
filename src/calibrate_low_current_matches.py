import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
CURRENT_LIMIT = 0.5

test4 = pd.read_csv("data/test4_subset.csv")
test2 = pd.read_csv("data/test2_subset.csv")

train = (
    test4.loc[test4["PS-I-MON"] < CURRENT_LIMIT]
    .dropna(subset=FEATURES)
    .reset_index()
)
query = (
    test2.loc[test2["PS-I-MON"] < CURRENT_LIMIT]
    .dropna(subset=FEATURES)
    .reset_index()
)

scaler = RobustScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_test = scaler.transform(query[FEATURES])

# For Test4, request two neighbours so the first can be excluded
# as the row itself. The second distance is the nearest other row.
within_model = NearestNeighbors(n_neighbors=2)
within_model.fit(X_train)
within_distances, _ = within_model.kneighbors(X_train)
within_distances = within_distances[:, 1]

# For Test2, find the nearest Test4 row.
cross_model = NearestNeighbors(n_neighbors=1)
cross_model.fit(X_train)
cross_distances, _ = cross_model.kneighbors(X_test)
cross_distances = cross_distances.ravel()

def report(name, distances):
    s = pd.Series(distances)
    print(f"\n{name} (n={len(s)})")
    print(
        s.describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99])
        .round(4)
        .to_string()
    )

report("TEST4: distance to nearest other Test4 row", within_distances)
report("TEST2: distance to nearest Test4 row", cross_distances)

print("\n=== TEST2 DISTANCES RELATIVE TO TEST4 REFERENCE ===")
for percentile in [50, 75, 90, 95, 99]:
    cutoff = pd.Series(within_distances).quantile(percentile / 100)
    fraction = (cross_distances > cutoff).mean() * 100
    print(
        f"Test4 {percentile}th-percentile distance = {cutoff:.4f}; "
        f"Test2 rows above it = {fraction:.2f}%"
    )

print(
    "\nNote: these are density/support diagnostics, not proof of "
    "physical equivalence or model accuracy."
)
