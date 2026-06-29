# ==========================================================
# Hospital Patient Status Prediction using Machine Learning
# ==========================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Library for connecting Python with MySQL
from sqlalchemy import create_engine

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Library to save trained model
import joblib

# ==========================================================
# MySQL Database Connection
# ==========================================================

username = 'root'
password = '1234'
host = 'localhost'
port = 3306
database = 'healthcare'

# Create database connection
engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
)

# ==========================================================
# Load Data from MySQL Database
# ==========================================================

query = """
SELECT f.*, Department_Name
FROM fact_table f
JOIN Department d
ON f.Dpt_ID = d.Dpt_ID
"""

# Read SQL data into DataFrame
df = pd.read_sql(query, engine)

# Display dataset information
print("Shape:", df.shape)

print("\nFirst 5 Rows")
print(df.head())

# ==========================================================
# Check Missing Values
# ==========================================================

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================================
# Data Cleaning
# ==========================================================

# Remove dollar sign and commas from treatment cost
df["treatemencost"] = (
    df["treatemencost"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

# Rename incorrect column name
df.rename(columns={"treatemencost": "treatmentcost"}, inplace=True)

# Convert columns into correct data types
df["Age"] = df["Age"].astype(int)
df["LOS"] = df["LOS"].astype(int)
df["ER_Time"] = df["ER_Time"].astype(int)
df["treatmentcost"] = df["treatmentcost"].astype(float)

# Display data types
print(df.dtypes)

# ==========================================================
# Exploratory Data Analysis (EDA)
# ==========================================================

# Gender Distribution
gender_counts = df["Gender"].value_counts()

# Patient Status Distribution
status_count = df["Status"].value_counts()

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Pie Chart for Gender
ax[0].pie(
    gender_counts,
    labels=gender_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
ax[0].set_title("Gender Distribution")

# Pie Chart for Patient Status
ax[1].pie(
    status_count,
    labels=status_count.index,
    autopct="%1.1f%%",
    startangle=90
)
ax[1].set_title("Status Distribution")

plt.show()

# ==========================================================
# Age Distribution
# ==========================================================

plt.figure(figsize=(6,3))

plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()

# ==========================================================
# Top 10 Diseases
# ==========================================================

top_disease = df["disease_name"].value_counts().head(10)

plt.figure(figsize=(6,3))

top_disease.plot(kind="bar")

plt.title("Top 10 Diseases")

plt.show()

# ==========================================================
# Disease Distribution by Gender
# ==========================================================

plt.figure(figsize=(10,4))

sns.countplot(
    data=df,
    x="disease_name",
    hue="Gender"
)

plt.xticks(rotation=90)

plt.show()

# ==========================================================
# Department-wise Treatment Cost
# ==========================================================

plt.figure(figsize=(10,5))

sns.barplot(
    data=df,
    x="Department_Name",
    y="treatmentcost",
    color="green"
)

plt.xticks(rotation=90)

plt.title("Department Wise Treatment Cost ($)")

plt.show()

# ==========================================================
# Label Encoding (Correct)
# ==========================================================

from sklearn.preprocessing import LabelEncoder

encoders = {}

cat_cols = [
    "Gender",
    "Patient type",
    "Department_Name",
    "disease_name",
    "Status"
]

# Store original text values and encode only once
for col in cat_cols:

    le = LabelEncoder()

    df[col] = df[col].astype(str)

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

# Save encoders
joblib.dump(encoders, "label_encoders.pkl")

print("Label Encoders Saved Successfully")

# ==========================================================
# Feature Selection
# ==========================================================

X = df[
    [
        "Age",
        "LOS",
        "ER_Time",
        "treatmentcost",
        "Gender",
        "Patient type",
        "Department_Name",
        "disease_name"
    ]
]

# Target Variable
y = df["Status"]

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])

# ==========================================================
# Machine Learning Models
# ==========================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            random_state=42
        ))
    ]),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

# ==========================================================
# Model Training & Evaluation
# ==========================================================

results = {}

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Cross Validation
    cv_score = cross_val_score(
        model,
        X,
        y,
        cv=5
    ).mean()

    # Store accuracy
    results[name] = acc

    print("\n", "="*50)
    print(name)
    print("="*50)

    print("Accuracy:", round(acc,4))
    print("Cross Validation:", round(cv_score,4))

    print("\nClassification Report")

    print(classification_report(y_test, y_pred))

# ==========================================================
# Best Model Selection
# ==========================================================

best_model_name = max(results, key=results.get)

print("\nBest Model:", best_model_name)

print("Accuracy:", results[best_model_name])

best_model = models[best_model_name]

# ==========================================================
# Confusion Matrix
# ==========================================================

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ==========================================================
# Feature Importance using Random Forest
# ==========================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")

plt.show()

# ==========================================================
# Save Trained Random Forest Model
# ==========================================================

joblib.dump(
    rf,
    "hospital_status_model.pkl"
)

print("Model and Label Encoders saved successfully.")