"""First-order autoregressive language model: P(next | current)."""

from collections import Counter, defaultdict
import random

START = "<START>"
END = "<END>"
DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def distribution_from_counts(counts):
    total = sum(counts.values())
    return {} if total == 0 else {word: count / total for word, count in counts.items()}


class FirstOrderModel:
    def __init__(self, sentences):
        self.counts = defaultdict(Counter)
        for sentence in sentences:
            sequence = [START, *sentence.lower().split(), END]
            for current, following in zip(sequence, sequence[1:]):
                self.counts[current][following] += 1

    def distribution(self, current):
        return distribution_from_counts(self.counts.get(current, {}))

    def predict(self, current):
        probabilities = self.distribution(current)
        if not probabilities:
            raise KeyError(f"no transition observed from {current!r}")
        return max(probabilities, key=probabilities.get)

    def generate(self, mode="sample", seed=7, maximum_tokens=20):
        generator = random.Random(seed)
        current = START
        output = []
        for _ in range(maximum_tokens):
            probabilities = self.distribution(current)
            if not probabilities:
                break
            if mode == "greedy":
                following = self.predict(current)
            elif mode == "sample":
                following = generator.choices(
                    list(probabilities), weights=list(probabilities.values())
                )[0]
            else:
                raise ValueError("mode must be 'greedy' or 'sample'")
            if following == END:
                break
            output.append(following)
            current = following
        return " ".join(output)


def main():
    model = FirstOrderModel(DATASET)
    print("FIRST-ORDER MODEL")
    print("INPUT: training sentences")
    for sentence in DATASET:
        print(f"  {START} {sentence} {END}")
    print("\nINPUT: current word = 'the'")
    print("OUTPUT: P(next | the) =", model.distribution("the"))
    print("OUTPUT: most probable next word =", model.predict("the"))
    print("\nOUTPUT: greedy =", model.generate(mode="greedy"))
    print("OUTPUT: sampled =", model.generate(mode="sample", seed=7))
    print("\nNORMALISATION")
    for word, probabilities in model.counts.items():
        print(f"{word}: {sum(model.distribution(word).values()):.2f}")


if __name__ == "__main__":
    main()
