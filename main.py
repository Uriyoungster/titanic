import random

import streamlit as st


PASSENGERS = [
    {"name": "Ada", "age": 8, "fare": 30, "survived": True},
    {"name": "Arthur", "age": 42, "fare": 8, "survived": False},
    {"name": "Beatrice", "age": 25, "fare": 80, "survived": True},
    {"name": "Charles", "age": 55, "fare": 12, "survived": False},
    {"name": "Diana", "age": 19, "fare": 25, "survived": True},
    {"name": "Edward", "age": 36, "fare": 7, "survived": False},
    {"name": "Florence", "age": 31, "fare": 100, "survived": True},
    {"name": "George", "age": 67, "fare": 10, "survived": False},
    {"name": "Hazel", "age": 14, "fare": 50, "survived": True},
    {"name": "Isaac", "age": 29, "fare": 15, "survived": False},
    {"name": "Julia", "age": 40, "fare": 65, "survived": True},
    {"name": "Leonard", "age": 23, "fare": 9, "survived": False},
]

TOTAL_ROUNDS = 5


def start_game():
    st.session_state.passengers = random.sample(PASSENGERS, TOTAL_ROUNDS)
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
st.write("Can you guess which passengers made it to safety?")

if "passengers" not in st.session_state:
    start_game()

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
        start_game()
        st.rerun()

else:
    number = st.session_state.round
    passenger = st.session_state.passengers[number]

    st.progress(number / TOTAL_ROUNDS)
    st.subheader(f"Passenger {number + 1} of {TOTAL_ROUNDS}")
    st.metric("Age", f"{passenger['age']} years")
    st.metric("Ticket fare", f"£{passenger['fare']}")

    st.write("Do you think this passenger survived?")

    choice = st.radio(
        "Choose your answer:",
        ["Survived", "Did not survive"],
        key=f"answer_{number}",
        horizontal=True,
    )

    if st.button("Submit guess", type="primary"):
        guess_survived = choice == "Survived"
        actual_survived = passenger["survived"]
        correct = guess_survived == actual_survived

        if correct:
            st.session_state.score += 1
            st.session_state.feedback = ("correct", actual_survived)
        else:
            st.session_state.feedback = ("incorrect", actual_survived)

    if st.session_state.feedback is not None:
        result, survived = st.session_state.feedback

        if result == "correct":
            st.success("✅ Correct guess!")
        else:
            st.error("❌ Not quite!")

        outcome = "survived" if survived else "did not survive"
        st.write(f"This passenger **{outcome}**.")

        if st.button("Next passenger", type="primary"):
            st.session_state.round += 1
            st.session_state.feedback = None
            if st.session_state.round >= TOTAL_ROUNDS:
                st.session_state.finished = True
            st.rerun()
