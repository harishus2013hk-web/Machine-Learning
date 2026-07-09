# ==========================================================
# Loan Default Prediction using Machine Learning
# ==========================================================

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================================================
# Step 1: Load the Dataset
# ==========================================================

# Read the loan dataset
df = pd.read_csv("data\\loan_dataset.csv")

# ==========================================================
# Step 2: Explore the Dataset
# ==========================================================

# Display first 5 rows
print("First 5 Rows")
print(df.head())

# Display number of rows and columns
print("\nDataset Shape")
print(df.shape)

# Display column names
print("\nColumns")
print(df.columns)

# Display data types of all columns
print("\nData Types")
print(df.dtypes)

# Display statistical summary
print("\nStatistical Summary")
print(df.describe())

# Display missing values in each column
print("\nMissing Values")
print(df.isnull().sum())

# ==========================================================
# Step 3: Handle Missing Values
# ==========================================================

# Fill missing numerical values using median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values using mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Verify that no missing values remain
print(df.isnull().sum())

# ==========================================================
# Step 4: Convert Categorical Data into Numerical Data
# ==========================================================

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# Encode all object-type columns
for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# ==========================================================
# Step 5: Exploratory Data Analysis (EDA)
# ==========================================================

# Plot loan status distribution
sns.countplot(x="Current_loan_status", data=df)
plt.title("Loan Status Distribution")
plt.show()

# Plot customer age distribution
plt.figure(figsize=(6,4))
sns.histplot(df["customer_age"], bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.show()

# Plot customer income distribution
plt.figure(figsize=(6,4))
sns.histplot(df["customer_income"], bins=20, color="green")
plt.title("Income Distribution")
plt.show()

# Plot loan amount distribution
plt.figure(figsize=(6,4))
sns.histplot(df["loan_amnt"], bins=20, color="orange")
plt.title("Loan Amount Distribution")
plt.show()

# Plot boxplot to detect outliers
sns.boxplot(x=df["loan_amnt"])
plt.title("Loan Amount Boxplot")
plt.show()

# ==========================================================
# Step 6: Separate Features and Target Variable
# ==========================================================

# Independent variables
X = df.drop("Current_loan_status", axis=1)

# Target variable
y = df["Current_loan_status"]

# ==========================================================
# Step 7: Feature Scaling
# ==========================================================

# Standardize the feature values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================================
# Step 8: Split Dataset into Training and Testing Sets
# ==========================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# Step 9: Create Machine Learning Models
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

# ==========================================================
# Step 10: Train and Evaluate Models
# ==========================================================

from sklearn.metrics import accuracy_score

results = []

# Train each model and calculate accuracy
for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results.append([name, accuracy])

# Store model performance
results = pd.DataFrame(results, columns=["Model", "Accuracy"])

print(results)

# ==========================================================
# Step 11: Compare Model Accuracy
# ==========================================================

plt.figure(figsize=(7,5))

plt.bar(results["Model"], results["Accuracy"])

plt.title("Model Comparison")

plt.xlabel("Machine Learning Models")

plt.ylabel("Accuracy")

plt.show()

# ==========================================================
# Step 12: Train the Best Model
# ==========================================================

best_model = RandomForestClassifier(random_state=42)

best_model.fit(X_train, y_train)

prediction = best_model.predict(X_test)

# ==========================================================
# Step 13: Generate Confusion Matrix
# ==========================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, prediction)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix")
plt.show()

# ==========================================================
# Step 14: Display Classification Report
# ==========================================================

from sklearn.metrics import classification_report

print(classification_report(y_test, prediction))

# ==========================================================
# Step 15: Save the Trained Model
# ==========================================================

joblib.dump(scaler, "scaler.pkl")
joblib.dump(best_model, "loan_prediction_model.pkl")

print("Model Saved Successfully")