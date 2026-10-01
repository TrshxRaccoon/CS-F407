"""Second-order autoregressive language model: P(next | previous two)."""

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


class SecondOrderModel:
    def __init__(self, sentences):
        self.counts = defaultdict(Counter)
        for sentence in sentences:
            sequence = [START, START, *sentence.lower().split(), END]
            for first, second, following in zip(
                sequence, sequence[1:], sequence[2:]
            ):
                self.counts[(first, second)][following] += 1

    def distribution(self, context):
        return distribution_from_counts(self.counts.get(context, {}))

    def predict(self, context):
        probabilities = self.distribution(context)
        if not probabilities:
            raise KeyError(f"no transition observed from {context!r}")
        return max(probabilities, key=probabilities.get)

    def generate(self, mode="sample", seed=7, maximum_tokens=20):
        generator = random.Random(seed)
        first, second = START, START
        output = []
        for _ in range(maximum_tokens):
            probabilities = self.distribution((first, second))
            if not probabilities:
                break
            if mode == "greedy":
                following = self.predict((first, second))
            elif mode == "sample":
                following = generator.choices(
                    list(probabilities), weights=list(probabilities.values())
                )[0]
            else:
                raise ValueError("mode must be 'greedy' or 'sample'")
            if following == END:
                break
            output.append(following)
            first, second = second, following
        return " ".join(output)


def main():
    model = SecondOrderModel(DATASET)
    print("SECOND-ORDER MODEL")
    print("INPUT: training sentences")
    for sentence in DATASET:
        print(f"  {START} {sentence} {END}")
    for context in [(START, START), ("the", "cat"), ("the", "dog")]:
        print(f"\nINPUT: context = {context}")
        print("OUTPUT: P(next | context) =", model.distribution(context))
        print("OUTPUT: most probable next word =", model.predict(context))
    print("\nOUTPUT: greedy =", model.generate(mode="greedy"))
    print("OUTPUT: sampled =", model.generate(mode="sample", seed=7))
    print("\nNORMALISATION")
    for context in model.counts:
        print(f"{context}: {sum(model.distribution(context).values()):.2f}")


if __name__ == "__main__":
    main()
