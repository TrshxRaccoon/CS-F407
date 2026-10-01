# Laboratory Report: Neural Models and XOR

## Task 1: Problem specification

The input space is $X = \{0,1\}^2$ and the binary output space is
$Y = \{0,1\}$. The four labelled examples are:

| Input    | Target |
| -------- | -----: |
| `(0, 0)` |      0 |
| `(0, 1)` |      1 |
| `(1, 0)` |      1 |
| `(1, 1)` |      0 |

The positive examples are diagonal while the negative examples are on the
other diagonal. Therefore, no single straight decision boundary can separate
the two classes. A single affine layer followed by a sigmoid can only make a
linear decision boundary, so it should fail to classify all four examples.

## Task 2: Model design and validation

The nonlinear binary model is a `2 -> 2 -> 1` network. Its hidden layer uses
Tanh, its output uses sigmoid, and it is trained with `BCELoss` and full-batch
SGD. The three-class model keeps the `2 -> 2` hidden representation and
replaces the binary output with three raw logits, trained using
`CrossEntropyLoss`.

Successful binary learning requires a decreasing loss, all four thresholded
predictions to be correct, and a nonzero first-layer gradient during early
backpropagation. The three-class checks are correct predicted classes, logits
with shape `(4, 3)`, and softmax rows whose components sum to approximately 1.

## Task 3: LLM use and verification

### Exact prompt used

> Generate a complete, well-documented standalone PyTorch Python program for
> the XOR dataset `(0,0)->0`, `(0,1)->1`, `(1,0)->1`, `(1,1)->0`. Create the
> dataset with tensors. Implement and train a linear baseline with input size
> 2, output size 1, one Linear layer followed by Sigmoid, BCELoss, SGD, and
> several thousand epochs. Print periodic loss and final predictions, and
> explain why it fails on XOR. Implement a second 2-2-1 network with a Tanh
> hidden layer, sigmoid output, BCELoss, and SGD. Print periodic loss,
> first-layer weight gradients after backward(), and final predictions.
> Compare both models. Then add a three-class 2-2-3 classifier for classes 0:
> `(0,0)`, 1: `(0,1)` and `(1,0)`, and 2: `(1,1)`, using three logits and
> CrossEntropyLoss. Print logits, softmax probabilities, predicted classes,
> verify probability sums, and explain softmax and the `p-y` gradient. Also
> include a zero-initialization symmetry experiment and compare sigmoid, Tanh,
> and ReLU hidden activations. Use clear names, comments, and a fixed seed.

The generated implementation was inspected for the forward pass, scalar loss,
`loss.backward()`, and `optimizer.step()`. One runtime correction was required:
`torch.set_printoptions` does not accept NumPy's `suppress=True` argument, so it
was changed to the PyTorch-compatible `sci_mode=False`. The final program was
then executed successfully.

## Task 4: Results

### Binary baseline and nonlinear model

| Model               | Initial loss | Final loss | Correct |
| ------------------- | -----------: | ---------: | ------: |
| Linear + sigmoid    |     0.699050 |   0.693147 |     2/4 |
| Tanh hidden network |     0.700802 |   0.000309 |     4/4 |

The linear model produced probabilities approximately `[0.5, 0.5, 0.5, 0.5]`
and predictions `[1, 1, 1, 1]`. The nonlinear model produced probabilities
approximately `[0.0002, 0.9996, 0.9996, 0.0002]` and predictions
`[0, 1, 1, 0]`. At the first nonlinear training step, the first-layer gradient
was approximately:

```text
[[ 0.0054, -0.0133],
 [-0.0192, -0.0159]]
```

This is useful learning evidence because the loss fell substantially while all
four predictions became correct; a merely nonzero gradient would not by itself
establish successful learning.

### Symmetry experiment

With all weights and biases set to zero, the two hidden-layer weight rows were
identical after steps 1 through 5. They remained `[[0, 0], [0, 0]]`. Identical
hidden units compute the same activation and receive the same gradient, so an
optimizer update cannot make them specialize. Random initialization breaks
this symmetry.

### Activation experiment

| Hidden activation | Final loss | Correct |  Early ` |     | grad W^(1) |     | \_2` |
| ----------------- | ---------: | ------: | -------: | --- | ---------- | --- | ---- |
| Sigmoid           |   0.001028 |     4/4 | 0.005185 |
| Tanh              |   0.000309 |     4/4 | 0.015719 |
| ReLU              |   0.693147 |     3/4 | 0.004804 |

These results describe this seed, architecture, optimizer, and learning rate;
they do not establish a universally best activation. Sigmoid derivatives can
become small when units saturate. ReLU has zero derivative for negative
pre-activations, which can leave a unit unable to update. Inspecting both
pre-activations and activations distinguishes these mechanisms.

## Task 5: Three-class extension

The output weight matrix has shape `(3, 2)`, giving three logits per input.
The final predicted classes were `[0, 1, 1, 2]`, matching the targets. The
softmax probability rows were approximately:

```text
[[0.9997, 0.0003, 0.0000],
 [0.0001, 0.9998, 0.0001],
 [0.0001, 0.9998, 0.0001],
 [0.0000, 0.0003, 0.9997]]
```

Every row summed to `1.0000`. Softmax exponentiates each logit and divides by
the sum of all exponentials, so its outputs are nonnegative and sum to one.
For one-hot target vector $y$ and softmax vector $p$, differentiating the
cross-entropy $-\sum_i y_i \log p_i$ with respect to the logits gives
$\nabla_z L = p-y$.

## Reflection questions

1. **Depth versus nonlinearity:** Adding affine layers alone does not increase
   the class of functions beyond one affine map. XOR requires a nonlinear
   hidden activation to create a separable representation.
2. **Backpropagation evidence:** The first-layer gradient was nonzero at the
   beginning, the loss decreased from `0.700802` to `0.000309`, and the network
   classified all four examples correctly.
3. **Identical initialization:** Equal hidden parameters produce equal hidden
   outputs and equal gradients. Consequently, every update preserves equality
   and the units cannot learn distinct features.
4. **Activation effect:** Tanh had the largest early gradient norm and lowest
   final loss in this run. Sigmoid also succeeded with a smaller gradient, while
   ReLU reached only `3/4`; these are experimental observations, explained by
   saturation for sigmoid/Tanh or zero slopes for inactive ReLU units.
5. **Output and loss pairing:** Binary classification needs one probability and
   binary cross-entropy, whereas mutually exclusive multiclass classification
   needs multiple logits and multiclass cross-entropy. The pairing determines
   the meaning and gradient of the output.
6. **LLM productivity and verification:** The LLM quickly supplied PyTorch
   boilerplate for models, loops, and diagnostics. Human verification was
   essential for checking the architecture, discovering the invalid print
   option, and measuring whether predictions actually learned XOR.
7. **Scaling tests:** Loss trends, sampled predictions, gradient health, and
   probability normalization remain useful at scale. Exhaustive four-point
   checks and large finite-difference gradient checks become too expensive, so
   they should be replaced by sampled or automated tests.

## Reproduction

Run the submitted program from the `LAB-4` directory:

```text
python xor_neural_models.py
```

The code and all reported values are in `xor_neural_models.py`.
