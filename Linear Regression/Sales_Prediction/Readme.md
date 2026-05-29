# Advertising Sales Prediction using Linear Regression

## Overview

This project uses **Linear Regression** to predict product sales based on advertising expenditures across different media channels:

* TV
* Radio
* Newspaper

The model is trained using the Advertising dataset and includes data preprocessing with feature scaling.

---

## Project Structure

```text
├── Advertising.csv
├── train_model.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset contains the following columns:

| Column    | Description                     |
| --------- | ------------------------------- |
| TV        | TV advertising budget           |
| radio     | Radio advertising budget        |
| newspaper | Newspaper advertising budget    |
| sales     | Product sales (target variable) |

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Pickle

---

## Model Training

Run the training script:

```bash
python train_model.py
```

The script performs:

1. Data loading
2. Train-test split (80:20)
3. Feature scaling using StandardScaler
4. Linear Regression model training
5. Model evaluation
6. Saving trained model and scaler

---

## Evaluation Metrics

The model is evaluated using:

* R² Score
* Mean Squared Error (MSE)

Example Output:

```text
R2 Score: 0.899
MSE: 2.91
Model saved as model.pkl
```

---

## Saved Files

### model.pkl

Serialized Linear Regression model.

### scaler.pkl

Serialized StandardScaler used during training.

These files can be loaded later for inference without retraining.

---

## Making Predictions

```python
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

new_data = np.array([[230.1, 37.8, 69.2]])

new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

print("Predicted Sales:", prediction[0])
```

---

## Model Workflow

```text
Advertising Data
       │
       ▼
Train-Test Split
       │
       ▼
StandardScaler
       │
       ▼
Linear Regression
       │
       ▼
Model Evaluation
       │
       ▼
Save Model (.pkl)
```

---
