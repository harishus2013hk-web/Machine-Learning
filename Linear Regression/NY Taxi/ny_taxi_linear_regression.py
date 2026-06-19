import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ── 1. Load data ──────────────────────────────────────────────────────────────
df=pd.read_csv(r"D:\Python\ML\Projects\Linear\NY Taxi\nyc_taxi_trip_duration.csv")

print("Shape:", df.shape)
print(df.head())

# ── 2. Parse datetimes ────────────────────────────────────────────────────────
df["pickup_datetime"]  = pd.to_datetime(df["pickup_datetime"],  dayfirst=True)
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], dayfirst=True)

# ── 3. Feature engineering ────────────────────────────────────────────────────
# Haversine distance between pickup and dropoff (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(
    df["pickup_latitude"],  df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)

# Datetime-based features
df["hour"]        = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek   # 0=Mon, 6=Sun
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["is_rush"]     = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
df["is_night"]    = (df["hour"] < 6).astype(int)

# Vendor flag
df["vendor_id"] = df["vendor_id"].astype(int)

# ── 4. Target & cleaning ──────────────────────────────────────────────────────
target = "trip_duration"

df = df.dropna(subset=[target, "distance_km"])
df = df[df[target]        > 0]       # remove zero/negative durations
df = df[df[target]        < 10_000]  # remove extreme outliers (seconds)
df = df[df["distance_km"] > 0]       # remove zero-distance trips
df = df[df["distance_km"] < 100]     # remove unrealistic distances
df = df[df["passenger_count"].between(1, 6)]

print(f"\nShape after cleaning: {df.shape}")
print(df[["distance_km", target]].describe().round(2))

# ── 5. Define features ────────────────────────────────────────────────────────
features = [
    "distance_km",
    "passenger_count",
    "vendor_id",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush",
    "is_night",
]

X = df[features]
y = df[target]

# ── 6. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 7. Scale ──────────────────────────────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 8. Train ──────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)

print("\n── Model results ─────────────────────────")
print(f"  R²   : {r2:.4f}")
print(f"  RMSE : {rmse:.2f} seconds  ({rmse/60:.2f} min)")
print(f"  Intercept : {model.intercept_:.4f}")
print("\n  Feature coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"    {feat:<20}: {coef:.4f}")

# ── 10. Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("NY Taxi – Trip Duration Linear Regression", fontsize=14, fontweight="bold")

# (a) Actual vs Predicted
axes[0].scatter(y_test / 60, y_pred / 60, alpha=0.2, s=10, color="steelblue")
lims = [0, max(y_test.max(), y_pred.max()) / 60]
axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual duration (min)")
axes[0].set_ylabel("Predicted duration (min)")
axes[0].set_title(f"Actual vs Predicted  (R² = {r2:.3f})")
axes[0].legend()

# (b) Residuals histogram
residuals = y_test - y_pred
axes[1].hist(residuals / 60, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Residual (min)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Residual distribution  (RMSE = {rmse/60:.2f} min)")

plt.tight_layout()
plt.savefig("ny_taxi_lr_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → ny_taxi_lr_plots.png")
