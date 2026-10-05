# ==== הגדרות - שנו כאן כדי לעבוד עם דאטה אחר ====

KAGGLE_DATASET = "harlfoxem/housesalesprediction"
FEATURE_COLUMN = "sqft_living"
TARGET_COLUMN = "price"

import kagglehub
import pandas as pd
import os
import numpy as np
import streamlit as st

# =========================
# הגדרות האתר
# =========================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Predictor")
st.write("הזינו גיל וקבלו הערכה של סיכויי ההישרדות.")

# =========================
# הורדת הדאטה
# =========================

@st.cache_data
def load_data():

    # Download latest version
    path = kagglehub.dataset_download("yasserh/titanic-dataset")

    csv_path = os.path.join(path, "Titanic-Dataset.csv")
    data = pd.read_csv(csv_path)

    return data


data = load_data()

# =========================
# ניקוי הנתונים
# =========================

data_clean = data.dropna(subset=['Age', 'Survived'])

X = data_clean['Age'].to_numpy()
y = data_clean['Survived'].to_numpy()

# =========================
# Baseline
# =========================

baseline_prediction = np.mean(y)
baseline_loss = np.mean(np.abs(y - baseline_prediction))

# =========================
# Linear Regression
# =========================

from sklearn.linear_model import LinearRegression

X_reshaped = X.reshape(-1, 1)

model = LinearRegression()
model.fit(X_reshaped, y)

w = model.coef_[0]
b = model.intercept_

# =========================
# Model Loss
# =========================

y_hat = model.predict(X_reshaped)

model_loss = np.mean(np.abs(y - y_hat))

# =========================
# אתר - הכנסת גיל
# =========================

st.subheader("🔢 הזנת גיל")

user_age = st.number_input(
    "הזינו גיל של אדם:",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=1.0
)

# =========================
# חיזוי
# =========================

if st.button("🔮 חשב סיכויי הישרדות"):

    predicted_survival = model.predict(
        np.array([[user_age]])
    )[0]

    # הפיכה לאחוזים
    survival_percentage = predicted_survival * 100

    # כדי שהתוצאה לא תחרוג מ-0% עד 100%
    survival_percentage = np.clip(
        survival_percentage,
        0,
        100
    )

    st.subheader("📊 תוצאה")

    st.metric(
        label="סיכויי הישרדות משוערים",
        value=f"{survival_percentage:.2f}%"
    )

    # Progress bar
    st.progress(
        int(survival_percentage)
    )

    if survival_percentage >= 50:
        st.success("🟢 לפי המודל, ההסתברות המשוערת היא מעל 50%.")
    else:
        st.error("🔴 לפי המודל, ההסתברות המשוערת היא מתחת ל-50%.")


# =========================
# מידע על המודל
# =========================

with st.expander("📈 מידע על המודל"):

    st.write("**משתנה קלט:** Age")
    st.write("**משתנה מטרה:** Survived")

    st.write(f"**w:** {w:.6f}")
    st.write(f"**b:** {b:.6f}")

    st.write(f"**Baseline loss:** {baseline_loss:.6f}")
    st.write(f"**Model loss:** {model_loss:.6f}")

    st.write(f"**מספר דוגמאות:** {len(X)}")
