import os

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

FALLBACK_PASSENGERS = [
    {"Name": "Ada", "Age": 8, "Fare": 30, "Survived": 1},
    {"Name": "Arthur", "Age": 42, "Fare": 8, "Survived": 0},
    {"Name": "Beatrice", "Age": 25, "Fare": 80, "Survived": 1},
    {"Name": "Charles", "Age": 55, "Fare": 12, "Survived": 0},
    {"Name": "Diana", "Age": 19, "Fare": 25, "Survived": 1},
    {"Name": "Edward", "Age": 36, "Fare": 7, "Survived": 0},
    {"Name": "Florence", "Age": 31, "Fare": 100, "Survived": 1},
    {"Name": "George", "Age": 67, "Fare": 10, "Survived": 0},
    {"Name": "Hazel", "Age": 14, "Fare": 50, "Survived": 1},
    {"Name": "Isaac", "Age": 29, "Fare": 15, "Survived": 0},
    {"Name": "Julia", "Age": 40, "Fare": 65, "Survived": 1},
    {"Name": "Leonard", "Age": 23, "Fare": 9, "Survived": 0},
]


@st.cache_data
def load_titanic_data():
    """Download Titanic data, or use sample data if Kaggle is unavailable."""
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        csv_path = os.path.join(path, "Titanic-Dataset.csv")
        data = pd.read_csv(csv_path)
        data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

        if data.empty:
            raise ValueError("The downloaded dataset contains no usable rows.")

        return data, "Kaggle Titanic dataset"
    except Exception:
        return pd.DataFrame(FALLBACK_PASSENGERS), "built-in sample data"


@st.cache_resource
def train_model(data):
    """Estimate survival probability using passenger age and fare."""
    X = data[FEATURE_COLUMNS].to_numpy()
    y = data[TARGET_COLUMN].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X, y)
    return model


def start_new_game(passengers):
    st.session_state.passengers = passengers.sample(
        n=min(TOTAL_ROUNDS, len(passengers))
    ).reset_index(drop=True)
    st.session_state.round_number = 0
    st.session_state.score = 0
    st.session_state.finished = False
    st.session_state.feedback = None


st.set_page_config(
    page_title="Titanic Lifeboat Challenge",
    page_icon="🚢",
    layout="centered",
)

st.title("🚢 Titanic: Lifeboat Challenge")
st.write("Guess whether each passenger survived. The model's chance appears after your guess.")

data, data_source = load_titanic_data()
model = train_model(data)
st.caption(f"Model data: {data_source}")

if "passengers" not in st.session_state:
    start_new_game(data)

if st.session_state.finished:
    total_passengers = len(st.session_state.passengers)
    st.success(
        f"Voyage complete! You scored "
        f"{st.session_state.score}/{total_passengers}."
    )

    if st.session_state.score == total_passengers:
        st.balloons()
        st.write("🏆 Perfect score, Captain!")
    elif st.session_state.score >= 3:
        st.write("🌟 Well done, Captain!")
    else:
        st.write("🧭 Tough voyage. Try again!")

    if st.button("Play again", type="primary"):
        start_new_game(data)
        st.rerun()

else:
    round_number = st.session_state.round_number
    passenger = st.session_state.passengers.iloc[round_number]
    total_passengers = len(st.session_state.passengers)

    age = float(passenger["Age"])
    fare = float(passenger["Fare"])

    st.progress(round_number / total_passengers)
    st.subheader(f"Passenger {round_number + 1} of {total_passengers}")

    age_column, fare_column = st.columns(2)
    age_column.metric("Age", f"{age:.0f} years")
    fare_column.metric("Ticket fare", f"£{fare:.2f}")

    if st.session_state.feedback is None:
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

            features = np.array([[age, fare]])
            survival_chance = float(model.predict_proba(features)[0][1])

            correct = int(guess_survived) == actual_survived
            st.session_state.feedback = {
                "correct": correct,
                "actual_survived": actual_survived,
                "survival_chance": survival_chance,
            }
            if correct:
                st.session_state.score += 1
            st.rerun()

    else:
        feedback = st.session_state.feedback

        if feedback["correct"]:
            st.success("✅ Correct guess!")
        else:
            st.error("❌ Not quite!")

        outcome = "survived" if feedback["actual_survived"] else "did not survive"
        st.write(f"The real passenger **{outcome}**.")

        # The model's survival chance is revealed only after the guess.
        st.metric(
            "Model's estimated survival chance",
            f"{feedback['survival_chance']:.1%}",
        )
        st.progress(feedback["survival_chance"])

        prediction = (
            "survived"
            if feedback["survival_chance"] >= 0.5
            else "did not survive"
        )
        st.caption(f"After seeing the chance, the model predicted: {prediction}")
        st.write(f"Current score: {st.session_state.score}/{total_passengers}")

        if st.button("Next passenger", type="primary"):
            st.session_state.round_number += 1
            st.session_state.feedback = None

            if st.session_state.round_number >= total_passengers:
                st.session_state.finished = True

            st.rerun()
