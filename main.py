# Lifeboat Challenge
# Install dependencies first:
# pip install kagglehub pandas numpy scikit-learn

KAGGLE_DATASET = "yasserh/titanic-dataset"
FEATURE_COLUMNS = ["Age", "Fare"]
TARGET_COLUMN = "Survived"

import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    """Download and load the Titanic dataset from Kaggle."""
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_path = os.path.join(path, "Titanic-Dataset.csv")
    return pd.read_csv(csv_path)


def train_model(data):
    """Train a survival classifier using passenger age and fare."""
    clean_data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = clean_data[FEATURE_COLUMNS].to_numpy()
    y = clean_data[TARGET_COLUMN].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X, y)

    return model, clean_data


def get_guess():
    """Ask the player to guess whether a passenger survived."""
    while True:
        guess = input("Your guess — survived? (y/n): ").strip().lower()
        if guess in ("y", "yes"):
            return 1
        if guess in ("n", "no"):
            return 0
        print("Please enter y or n.")


def play_game(model, data, rounds=3):
    print("\n🚢 LIFEBOAT CHALLENGE 🚢")
    print("Three passengers need your help.")
    print("Use their age and ticket fare to guess who survived.\n")

    # Sample a few real passengers, without revealing their outcome.
    passengers = data.sample(n=rounds, random_state=None)
    score = 0

    for round_number, (_, passenger) in enumerate(passengers.iterrows(), start=1):
        age = float(passenger["Age"])
        fare = float(passenger["Fare"])

        print(f"Passenger {round_number} of {rounds}")
        print(f"Age: {age:.0f} | Fare: £{fare:.2f}")

        guess = get_guess()
        actual = int(passenger[TARGET_COLUMN])
        prediction = int(model.predict(np.array([[age, fare]]))[0])

        if guess == actual:
            score += 1
            print("✅ Correct!")
        else:
            print("❌ Not this time.")

        print(f"Actual outcome: {'Survived' if actual else 'Did not survive'}")
        print(
            f"Model's guess: {'Survived' if prediction else 'Did not survive'}\n"
        )

    print(f"Game over! You scored {score}/{rounds}.")

    if score == rounds:
        print("🏆 Incredible — you saved everyone!")
    elif score >= 2:
        print("🌟 Nice work, lifeboat captain!")
    else:
        print("🧭 Tough voyage! Try again for a higher score.")


def main():
    print("Loading the Titanic data and preparing the game...")
    data = load_data()
    model, clean_data = train_model(data)

    print(f"Ready! The game has {len(clean_data)} usable passenger records.")
    play_game(model, clean_data)


if __name__ == "__main__":
    main()
