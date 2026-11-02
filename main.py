import os
import random

import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KAGGLE_DATASET = "yasserh/titanic-dataset"
FEATURE_COLUMNS = ["Age", "Fare"]
TARGET_COLUMN = "Survived"
TOTAL_ROUNDS = 5

# Simple fallback passenger data lets the app still run if Kaggle download fails.
FALLBACK_PASSENGERS = [
    {"name": "Ada", "age": 8, "fare": 30, "survived": 1},
    {"name": "Arthur", "age": 42, "fare": 8, "survived": 0},
    {"name": "Beatrice", "age": 25, "fare": 80, "survived": 1},
    {"name": "Charles", "age": 55, "fare": 12, "survived": 0},
    {"name": "Diana", "age": 19, "fare": 25, "survived": 1},
    {"name": "Edward", "age": 36, "fare": 7, "survived": 0},
    {"name": "Florence", "age": 31, "fare": 100, "survived": 1},
    {"name": "George", "age": 67, "fare": 10, "survived": 0},
    {"name": "Hazel", "age": 14, "fare": 50, "survived": 1},
    {"name": "Isaac", "age": 29, "fare": 15, "survived": 0},
    {"name": "Julia", "age": 40, "fare": 65, "survived": 1},
    {"name": "Leonard", "age": 23, "fare": 9, "survived": 0},
]


@st.cache_data
def load_titanic_data():
    """Download Titanic data; use fallback examples if Kaggle is unavailable."""
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        csv_path = os.path.join(path, "Titanic-Dataset.csv")
        data = pd.read_csv(csv_path)
        data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

        if data.empty:
            raise ValueError("The downloaded dataset contains no usable rows.")

        return data, "Kaggle Titanic dataset"
    except Exception:
        data = pd.DataFrame(FALLBACK_PASSENGERS)
        return data, "built-in sample data (Kaggle download unavailable)"


@st.cache_resource
def train_model(data):
    """Train a survival-probability model from age and fare."""
    X = data[FEATURE_COLUMNS].to_numpy()
    y = data[TARGET_COLUMN].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X, y)
    return model


def new_game(passengers):
    st.session_state.passengers = passengers.sample(
        n=min(TOTAL_ROUNDS, len(passengers))
    ).reset_index(drop=True)
    st.session_state.round = 0
    st.session_state.score = 0
    st.session_state.finished = False
    st.session_state.feedback = None


st.set_page_config(
    page_title="Titanic Lifeboat Challenge",
    page_icon="🚢",
    layout="centered",
)

st.title("🚢 Titanic: Lifeboat Challenge")
st.write("Check the model's survival chance, then make your own guess.")

data, data_source = load_titanic_data()
model = train_model(data)

st.caption(f"Probability model trained with: {data_source}")

if "passengers" not in st.session_state:
    new_game(data)

if st.session_state.finished:
    st.success(f"Voyage complete! You scored {st.session_state.score}/{TOTAL_ROUNDS}.")

    if st.session_state.score == TOTAL_ROUNDS:
        st.balloons()
        st.write("🏆 Perfect score, Captain!")
    elif st.session_state.score >= 3:
        st.write("🌟 Well done, Captain!")
    else:
        st.write("🧭 Tough voyage. Try again!")

    if st.button("Play again", type="primary"):
        new_game(data)
        st.rerun()

else:
    round_number = st.session_state.round
    passenger = st.session_state.passengers.iloc[round_number]

    age = float(passenger["Age"])
    fare = float(passenger["Fare"])
    features = np.array([[age, fare]])

    survival_chance = model.predict_proba(features)[0][1]
    predicted_survival = survival_chance >= 0.5

    st.progress(round_number / TOTAL_ROUNDS)
    st.subheader(f"Passenger {round_number + 1} of {TOTAL_ROUNDS}")

    col1, col2 = st.columns(2)
    col1.metric("Age", f"{age:.0f} years")
    col2.metric("Ticket fare", f"£{fare:.2f}")

    st.metric("Model's estimated survival chance", f"{survival_chance:.1%}")
    st.progress(float(survival_chance))
    st.caption(
        f"Model prediction: {'survived' if predicted_survival else 'did not survive'}"
    )

    st.write("What do you think happened?")
    choice = st.radio(
        "Your guess:",
        ["Survived", "Did not survive"],
        key=f"answer_{round_number}",
        horizontal=True,
    )

    if st.button("Submit guess", type="primary"):
        guess_survived = choice == "Survived"
        actual_survived = int(passenger[TARGET_COLUMN])
        correct = int(guess_survived) == actual_survived

        if correct:
            st.session_state.score += 1
            st.session_state.feedback = ("correct", actual_survived)
        else:
            st.session_state.feedback = ("incorrect", actual_survived)

    if st.session_state.feedback is not None:
        result, actual_survived = st.session_state.feedback

        if result == "correct":
            st.success("✅ Correct guess!")
        else:
            st.error("❌ Not quite!")

        outcome = "survived" if actual_survived else "did not survive"
        st.write(f"The real passenger **{outcome**}.")
        st.write(f"Current score: {st.session_state.score}/{TOTAL_ROUNDS}")

        if st.button("Next passenger", type="primary"):
            st.session_state.round += 1
            st.session_state.feedback = None

            if st.session_state.round >= len(st.session_state.passengers):
                st.session_state.finished = True

            st.rerun()
