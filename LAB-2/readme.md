# Logical Planning Laboratory

## 1. Planning problem specification

### Initial state

I = {At(Robot, A), At(Package, A)}

### Goal

G = {At(Package, C)}

### Available actions

- Move(A, B): preconditions {At(Robot, A)}; effects {¬At(Robot, A), At(Robot, B)}
- Move(B, A): preconditions {At(Robot, B)}; effects {¬At(Robot, B), At(Robot, A)}
- Move(B, C): preconditions {At(Robot, B)}; effects {¬At(Robot, B), At(Robot, C)}
- Move(C, B): preconditions {At(Robot, C)}; effects {¬At(Robot, C), At(Robot, B)}
- PickUp(Package, A): preconditions {At(Robot, A), At(Package, A)}; effects {¬At(Package, A), Holding(Package)}
- PickUp(Package, B): preconditions {At(Robot, B), At(Package, B)}; effects {¬At(Package, B), Holding(Package)}
- Drop(Package, C): preconditions {At(Robot, C), Holding(Package)}; effects {¬Holding(Package), At(Package, C)}

### Applicability check

- PickUp(Package, A) is applicable in I because At(Robot, A) and At(Package, A) are both true.
- Drop(Package, C) is not applicable in I because the preconditions require both At(Robot, C) and Holding(Package), which are false at the start.

## 2. Manual plan

A valid plan is:

1. Move(A, B)
2. PickUp(Package, B)
3. Move(B, C)
4. Drop(Package, C)

State trace:

- S0 = {At(Robot, A), At(Package, A)}
- After Move(A, B): {At(Robot, B), At(Package, A)}
- After PickUp(Package, B): {At(Robot, B), Holding(Package)}
- After Move(B, C): {At(Robot, C), Holding(Package)}
- After Drop(Package, C): {At(Robot, C), At(Package, C)}

This satisfies the goal G = {At(Package, C)}.

## 3. LLM prompt used

I want to implement a simple planning agent in Python.
Represent a state as a set of logical propositions.
Each action should contain:

- a name;
- positive preconditions;
- negative preconditions;
- positive effects;
- negative effects.

An action is applicable if all of its preconditions are satisfied by the current state.
When an action is applied:

1. remove its negative effects from the state;
2. add its positive effects to the state.

Use breadth-first search to find a sequence of actions that achieves a specified goal.
The program should also:

- detect when no plan exists;
- print the resulting sequence of actions;
- print the states reached after each action.

Explain the implementation and identify any assumptions you make.
Run the generated program on the warehouse problem.

## 4. Generated Python program

The working planner is implemented in [planner.py](planner.py). It includes:

- Action data model
- applicability test
- state transition function
- BFS search for a valid plan
- explicit no-plan detection
- console output for the action sequence and intermediate states

## 5. Tests and results

The validation suite is in [tests/test_planner.py](tests/test_planner.py).

### Test A: Solvable problem

- Initial state: {At(Robot, A), At(Package, A)}
- Goal: {At(Package, C)}
- Result: plan found
- Valid plan: Move(A, B) -> PickUp(Package, B) -> Move(B, C) -> Drop(Package, C)
- Status: valid

### Test B: Impossible problem

- Removed the pickup action
- Initial state: {At(Robot, A), At(Package, A)}
- Goal: {At(Package, C)}
- Result: no plan found
- Status: correct

### Test C: Irrelevant action

- Added an action Move(A, C) that moves the robot but not the package
- Goal: {At(Package, C)}
- Result: planner still required the package to reach C via the correct drop action
- Status: correct

### Verification

Command run:
`py -3 -m pytest -q -s`

Result:
`5 passed in 0.06s`

## 6. Answers to the “Think About It” questions

### Q: What is the relationship between logic and search?

Logic determines whether an action is applicable and how the state changes. Search decides which sequence of applicable actions to try next.

### Q: How do logical reasoning and BFS work together?

The planner checks preconditions in the current state. If they are satisfied, it generates a successor state. BFS then explores alternative action sequences until the goal is reached or all possibilities are exhausted.

### Q: Why should we not trust an LLM explanation more than independently executed state transitions?

Because the explanation is a generated narrative; the actual state transitions are objective and computed by the program. Independent execution is a stronger form of verification.

## 7. Reflection on the use of the LLM

The LLM helped produce the initial structure of the planner quickly, but the final system still needed to be checked and validated by execution. The key lesson was: Understand -> Specify -> Generate -> Execute -> Verify.

This illustrates that LLM-generated code can accelerate engineering, but it does not replace reasoning, testing, or independent validation.

## 8. Optional Prolog extension

In Prolog, facts and rules provide logical inference. For example:

- connected(a,b).
- connected(b,c).
- can_move(X,Y) :- connected(X,Y).

A query such as `?- can_move(a,b).` succeeds because a and b are directly connected, while `?- can_move(a,c).` fails because there is no direct connection between a and c in the knowledge base. This demonstrates logical entailment through facts and rules.

## 9. Summary

This lab demonstrates the core AI pattern:
Logical reasoning + Search = Planning

The planner checks whether actions are valid in a state, updates the world according to the action effects, and searches through alternatives until it reaches the goal or proves none exists.
