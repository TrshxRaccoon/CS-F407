"""PyTorch experiments for binary XOR and its three-class extension.

The binary experiment compares a single affine classifier with a small neural
network. The extension keeps the hidden layer and changes only the output to
three logits so that the sensor states are classified as 0, 1, or 2.

Run with:
    python xor_neural_models.py
"""

import torch
from torch import nn


# Reproducibility makes the printed results easier to inspect and compare.
torch.manual_seed(7)
torch.set_printoptions(precision=4, sci_mode=False)


# The four possible pairs of binary sensor readings and their XOR targets.
inputs = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
)
binary_targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
three_class_targets = torch.tensor([0, 1, 1, 2])


class LinearXORModel(nn.Module):
    """A linear decision function followed by a binary sigmoid."""

    def __init__(self):
        super().__init__()
        self.output_layer = nn.Linear(2, 1)

    def forward(self, features):
        return torch.sigmoid(self.output_layer(features))


class NonlinearXORModel(nn.Module):
    """A 2-2-1 network with a nonlinear hidden layer."""

    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 1)

    def forward(self, features):
        hidden_values = torch.tanh(self.hidden_layer(features))
        return torch.sigmoid(self.output_layer(hidden_values))


class ThreeClassXORModel(nn.Module):
    """A 2-2-3 network that returns raw logits for three classes."""

    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 3)

    def forward(self, features):
        hidden_values = torch.tanh(self.hidden_layer(features))
        return self.output_layer(hidden_values)


class ActivationXORModel(nn.Module):
    """A binary XOR network whose hidden activation can be changed."""

    def __init__(self, activation_name):
        super().__init__()
        self.hidden_layer = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 1)
        self.activation_name = activation_name

    def forward(self, features):
        hidden_values = self.hidden_layer(features)
        if self.activation_name == "sigmoid":
            hidden_values = torch.sigmoid(hidden_values)
        elif self.activation_name == "tanh":
            hidden_values = torch.tanh(hidden_values)
        elif self.activation_name == "relu":
            hidden_values = torch.relu(hidden_values)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
        return torch.sigmoid(self.output_layer(hidden_values))


def train_linear_model():
    """Train the intentionally insufficient linear binary baseline."""
    model = LinearXORModel()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(1, 5001):
        predictions = model(inputs)
        loss = loss_function(predictions, binary_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"[linear] epoch {epoch:4d}, loss = {loss.item():.6f}")

    with torch.no_grad():
        final_probabilities = model(inputs)
        final_labels = (final_probabilities >= 0.5).to(torch.int64)
    print("[linear] final probabilities:", final_probabilities.flatten())
    print("[linear] final predictions:  ", final_labels.flatten())
    print("[linear] correct examples:    ", int((final_labels == binary_targets).sum()), "/ 4")
    print("[linear] expected failure: one affine decision boundary cannot separate XOR.")
    return model


def train_nonlinear_model():
    """Train the Tanh XOR network and expose first-layer gradients."""
    model = NonlinearXORModel()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    for epoch in range(1, 10001):
        predictions = model(inputs)
        loss = loss_function(predictions, binary_targets)

        optimizer.zero_grad()
        loss.backward()

        # This tensor is d(loss)/d(first-layer weights), computed by backward().
        if epoch == 1 or epoch % 1000 == 0:
            print(f"[nonlinear] epoch {epoch:5d}, loss = {loss.item():.6f}")
            print("[nonlinear] first-layer weight gradients:\n", model.hidden_layer.weight.grad)

        optimizer.step()

    with torch.no_grad():
        final_probabilities = model(inputs)
        final_labels = (final_probabilities >= 0.5).to(torch.int64)
    print("[nonlinear] final probabilities:", final_probabilities.flatten())
    print("[nonlinear] final predictions:  ", final_labels.flatten())
    print("[nonlinear] correct examples:    ", int((final_labels == binary_targets).sum()), "/ 4")
    return model


def train_three_class_model():
    """Train the three-class version with logits and multiclass cross-entropy."""
    model = ThreeClassXORModel()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    for epoch in range(1, 5001):
        logits = model(inputs)
        loss = loss_function(logits, three_class_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"[three-class] epoch {epoch:4d}, loss = {loss.item():.6f}")

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = logits.argmax(dim=1)
        probability_sums = probabilities.sum(dim=1)

    print("[three-class] logits:\n", logits)
    print("[three-class] softmax probabilities:\n", probabilities)
    print("[three-class] predicted classes: ", predicted_classes)
    print("[three-class] probability sums:  ", probability_sums)
    print("[three-class] all sums near one:  ", bool(torch.allclose(probability_sums, torch.ones(4))))
    print("[three-class] output weight shape: ", tuple(model.output_layer.weight.shape))
    return model


def run_symmetry_experiment():
    """Show that identical zero-initialized hidden units remain identical."""
    model = NonlinearXORModel()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    print("\n=== Symmetry experiment: all weights and biases initialized to zero ===")
    for step in range(1, 6):
        predictions = model(inputs)
        loss = loss_function(predictions, binary_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rows_are_equal = torch.allclose(
            model.hidden_layer.weight[0], model.hidden_layer.weight[1]
        )
        print(
            f"[symmetry] step {step}, hidden weight rows = {model.hidden_layer.weight.tolist()}, "
            f"identical = {rows_are_equal}"
        )


def run_activation_experiment():
    """Compare sigmoid, Tanh, and ReLU using the same binary task."""
    print("\n=== Hidden activation experiment ===")
    for activation_name in ("sigmoid", "tanh", "relu"):
        torch.manual_seed(7)
        model = ActivationXORModel(activation_name)
        loss_function = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        early_gradient_norm = None

        for epoch in range(1, 10001):
            predictions = model(inputs)
            loss = loss_function(predictions, binary_targets)
            optimizer.zero_grad()
            loss.backward()
            if epoch == 1:
                early_gradient_norm = model.hidden_layer.weight.grad.norm().item()
            optimizer.step()

        with torch.no_grad():
            final_probabilities = model(inputs)
            final_labels = (final_probabilities >= 0.5).to(torch.int64)
        correct_count = int((final_labels == binary_targets).sum())
        print(
            f"[activation] {activation_name:7s}: final loss = {loss.item():.6f}, "
            f"correct = {correct_count}/4, early first-layer gradient norm = "
            f"{early_gradient_norm:.6f}"
        )


def explain_results():
    """Print the mathematical interpretation required by the experiment."""
    print("\nInterpretation")
    print("A stack of affine layers is still affine, so the linear baseline cannot represent XOR.")
    print("The Tanh hidden layer bends the representation, allowing the output layer to separate it.")
    print("Softmax exponentiates each logit and normalizes by their positive sum, so probabilities sum to 1.")
    print("For one-hot target y and softmax probabilities p, cross-entropy has logit gradient p - y.")


if __name__ == "__main__":
    print("=== Binary XOR: linear baseline ===")
    train_linear_model()
    print("\n=== Binary XOR: nonlinear network ===")
    train_nonlinear_model()
    print("\n=== Three-class sensor classifier ===")
    train_three_class_model()
    run_symmetry_experiment()
    run_activation_experiment()
    explain_results()