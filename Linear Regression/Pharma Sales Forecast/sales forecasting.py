# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load Dataset
df = pd.read_csv("pharma_sales_dataset.csv")

# Convert date column
df["month"] = pd.to_datetime(df["month"])

# Sort data
df = df.sort_values(["rep_id", "month"])

print(df.head())

# 3. Feature Engineering

# Create month number
df["month_num"] = df["month"].dt.month

# Create lag feature
df["lag_1"] = df.groupby("rep_id")["sales_units"].shift(1)

# Remove null rows
df = df.dropna()

# Encode categorical columns
le_region = LabelEncoder()
df["region_enc"] = le_region.fit_transform(df["region"])

# 4. Select Features
features = [
    "lag_1",
    "doctor_coverage",
    "calls_made",
    "target_ach_pct",
    "tenure_months",
    "region_enc",
    "month_num"
]

target = "sales_units"

X = df[features]
y = df[target]

# 5. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 6. Train Model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 7. Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)
print("RMSE :", round(rmse, 2))
print("MAE  :", round(mae, 2))
print("R²   :", round(r2, 4))

# 8. Plot Actual vs Predicted
plt.figure(figsize=(10,5))

plt.plot(y_test.values, label="Actual Sales")
plt.plot(y_pred, label="Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Records")
plt.ylabel("Sales Units")

plt.legend()
plt.show()