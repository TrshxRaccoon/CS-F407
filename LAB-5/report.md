# AI Laboratory: Bayesian Networks and Autoregressive Language Models

## Files submitted

- `first_order_model.py`: estimates P(next token | current token).
- `second_order_model.py`: estimates P(next token | previous two tokens).
- `conditional_probability_tables.txt`: selected CPT rows and normalization results.
- `generated_sentences.txt`: reproducible generated examples.

The models use ordinary Python data structures and random sampling. They add
`<START>` and `<END>` to every sentence and do not use a pretrained model or
machine-learning library.

## Question 1

The chain rule changes one whole-sentence probability into a sequence of
next-token probabilities. A generator can therefore start at `<START>`, sample
the next token, and repeatedly sample from the conditional distribution until
`<END>` is produced.

## Question 2

The first-order network assumes that earlier tokens do not matter once the
immediately preceding token is known:

$$P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1}).$$

## Question 3

For the six supplied sentences:

| Current token | Conditional distribution                                         |
| ------------- | ---------------------------------------------------------------- |
| `the`         | `cat`: 3/12, `dog`: 3/12, `mat`: 2/12, `rug`: 2/12, `park`: 2/12 |
| `cat`         | `sat`: 2/3, `ran`: 1/3                                           |
| `dog`         | `sat`: 2/3, `ran`: 1/3                                           |
| `sat`         | `on`: 1                                                          |
| `ran`         | `to`: 1                                                          |

The zero-probability transitions are all word pairs not observed in the
training data. For example, P(`on` | `cat`) = 0.

## Question 4

Transition counts are stored in `FirstOrderModel.counts` and
`SecondOrderModel.counts`. Each value is a `Counter` keyed by the possible
following token.

## Question 5

`distribution()` divides every transition count by the total count for the
current word or context. This computes the conditional probability table row.

## Question 6

Sampling generation chooses a token according to its probability. Greedy
generation chooses `arg max P(next | context)`. Greedy output is deterministic,
while sampling can select less probable observed transitions and creates more
variation.

## Question 7

An unseen context has an empty distribution. The `predict()` methods raise a
`KeyError` for it, while generation stops because there is no transition to
choose.

## Question 8

A row total of 0.87 means the implementation is not a valid normalized
conditional distribution. A count may be missing, the denominator may be
wrong, or probabilities may have been incorrectly rounded.

## Question 9

Not necessarily. The most probable word reflects frequencies in this small
corpus, not broad human linguistic knowledge or meaning. A probability model
estimates its training distribution; it does not automatically have human
expectations.

## Question 10

Greedy generation follows the same highest-probability path every time and can
repeat a cycle in a first-order model. Sampling produces more variation because
it can select any observed continuation according to its probability. A maximum
length protects greedy generation from an endless cycle.

## Question 11

1. First order has one incoming dependency, X(t-1) -> X(t). Second order has
   two, X(t-2) -> X(t) and X(t-1) -> X(t).
2. The first CPT is indexed by one token; the second CPT is indexed by ordered
   pairs of tokens.
3. The second-order model sees two previous tokens instead of one.
4. The second-order model needs more data because it has more possible contexts
   and therefore more sparse or unseen rows.

## Question 12

Additional context can distinguish phrases that have the same final word, which
can improve prediction. It also increases the number of CPT rows and parameters.
With limited data, many rows have few examples or zero examples, making their
probabilities unreliable.

## Question 13

Approach B is preferable because it specifies variables, dependencies, the
probability being estimated, and the generation method before code is written.
That specification makes the result inspectable and lets us test invariants such
as every CPT row summing to 1. The LLM is used as an implementation assistant,
not as a replacement for understanding the model.

## Question 14

Thinking in terms of a Bayesian network provides:

- an explicit representation of dependencies;
- a factorization of the joint distribution;
- an interpretation for each conditional probability;
- a principled sampling procedure;
- a clear independence assumption;
- an explanation of how larger context changes the CPT;
- testable invariants for checking the implementation.

## LLM reflection and validation

The LLM was given a behavioral specification: count adjacent tokens for the
first-order model, count triples for the second-order model, normalize counts,
provide greedy and sampling generation, stop at `<END>`, and display selected
CPT rows. The generated code was inspected rather than accepted blindly.

One correction came from testing the generated first-order implementation. An
initial check incorrectly expected the `the` row to have denominator 6 and
expected greedy generation to finish after one training sentence. Running the
normalization check showed that `the` has 8 outgoing observations in this
corpus, because it occurs both at sentence starts and inside phrases. Greedy
generation also revealed a valid repeating cycle, so a maximum-token limit was
added. The final code uses the observed counts, prints probability totals, and
supports seeded sampling for reproducible output.

The validation checks are visible in both implementation files. Every observed
first-order and second-order context is printed with a total of 1.00. Running
both files directly confirms the CPTs, greedy output, sampled output, and
unknown-context behavior.
