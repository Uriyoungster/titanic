import random

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


def survival_guess(age, fare):
    """A simple guess based on age and fare, inspired by the original features."""
    score = 0

    if age < 15:
        score += 2
    elif age < 30:
        score += 1

    if fare >= 50:
        score += 2
    elif fare >= 25:
        score += 1

    return score >= 2


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please type y or n.")


def play():
    print("=" * 42)
    print("       🚢 TITANIC: THE LIFEBOAT GAME")
    print("=" * 42)
    print("A storm is coming! Guess who made it to safety.")
    print("You'll meet 5 passengers. Type y or n for each.\n")

    chosen = random.sample(PASSENGERS, 5)
    points = 0

    for number, passenger in enumerate(chosen, start=1):
        print(f"Passenger {number}/5: {passenger['name']}")
        print(f"Age: {passenger['age']} | Ticket fare: £{passenger['fare']}")

        guess = ask_yes_no("Do you think they survived? (y/n): ")
        model_guess = survival_guess(passenger["age"], passenger["fare"])
        actual = passenger["survived"]

        if guess == actual:
            points += 1
            print("✅ Your guess was right!")
        else:
            print("❌ Your guess was wrong.")

        print("They survived!" if actual else "They did not survive.")
        print(
            "The model guessed correctly.\n"
            if model_guess == actual
            else "The model guessed incorrectly.\n"
        )

    print("=" * 42)
    print(f"Game over! Your score: {points}/5")

    if points == 5:
        print("🏆 Incredible! You saved every passenger.")
    elif points >= 3:
        print("🌟 Nice work, captain!")
    else:
        print("🧭 Rough voyage. Give it another try!")


if __name__ == "__main__":
    play()
