# ==== הגדרות - שנו כאן כדי לעבוד עם דאטה אחר ====

KAGGLE_DATASET = "harlfoxem/housesalesprediction"   # שם הדאטה-סט ב-Kaggle
FEATURE_COLUMN = "sqft_living"                       # שם עמודת המאפיין (X) - מה שממנו נחזה
TARGET_COLUMN = "price"                              # שם עמודת המטרה (y) - מה שרוצים לחזות
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/titanic-dataset")

print("Path to dataset files:", path)
import pandas as pd
import os

csv_path = os.path.join(path, "Titanic-Dataset.csv")
data = pd.read_csv(csv_path)
data.dropna()
data.head()
print(data[['Age', 'Survived']].isnull().sum())
import numpy as np

# Drop rows where 'Age' or 'Survived' are NaN before converting to numpy
data_clean = data.dropna(subset=['Age', 'Survived'])

X = data_clean['Age'].to_numpy()
y = data_clean['Survived'].to_numpy()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X sample:", X[:5])
print("y sample:", y[:5])
baseline_prediction = np.mean(y)
print("baseline prediction (always the same value):", baseline_prediction)

baseline_loss = np.mean(np.abs(y - baseline_prediction))
print("baseline loss:", baseline_loss)
from sklearn.linear_model import LinearRegression

# Prepare the reshaped X from the cleaned X
X_reshaped = X.reshape(-1, 1)
print("X shape before reshape:", X.shape)
print("X shape after reshape:", X_reshaped.shape)

model = LinearRegression()
model.fit(X_reshaped, y)

print("training done")
w = model.coef_[0]
b = model.intercept_

print("w:", w)
print("b:", b)
sample_x = X[0]
print("sample_x:", sample_x)

manual_prediction = w * sample_x + b
print("manual calculation (w * x + b):", manual_prediction)

sklearn_prediction = model.predict(np.array([[sample_x]]))
print("sklearn predict:", sklearn_prediction[0])
y_hat = model.predict(X_reshaped)

model_loss = np.mean(np.abs(y - y_hat))

print("baseline loss:", baseline_loss)
print("model loss:", model_loss)
user_input = input(" הזינו גיל של אדם: ")
user_age = float(user_input)

predicted_death = model.predict(np.array([[user_age]]))[0]

print("סיכויי הישרדות", predicted_death * 100 , "%")
