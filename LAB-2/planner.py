from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Action:
    name: str
    positive_preconditions: Set[str] = field(default_factory=set)
    negative_preconditions: Set[str] = field(default_factory=set)
    positive_effects: Set[str] = field(default_factory=set)
    negative_effects: Set[str] = field(default_factory=set)


def is_applicable(action: Action, state: Set[str]) -> bool:
    """Check whether all positive preconditions hold and no negative preconditions hold."""
    return action.positive_preconditions.issubset(state) and action.negative_preconditions.isdisjoint(state)


def apply_action(state: Set[str], action: Action) -> Set[str]:
    """Apply an action to a state, removing negative effects and adding positive effects."""
    new_state = set(state)
    new_state -= action.negative_effects
    new_state |= action.positive_effects
    return new_state


def solve_planning_problem(
    initial_state: Set[str],
    actions: Iterable[Action],
    goal: Set[str],
) -> Tuple[Optional[List[Action]], List[Set[str]]]:
    """Use breadth-first search to find a plan reaching the goal.

    Returns a tuple of (plan, states_reached). When no plan exists, returns (None, []).
    """
    queue = deque([(frozenset(initial_state), [])])
    visited = {frozenset(initial_state)}
    states_reached: List[Set[str]] = []

    while queue:
        current_state, plan = queue.popleft()
        current_state_set = set(current_state)
        states_reached.append(current_state_set.copy())

        if goal.issubset(current_state_set):
            return plan, states_reached

        for action in actions:
            if not is_applicable(action, current_state_set):
                continue

            successor = apply_action(current_state_set, action)
            successor_frozen = frozenset(successor)
            if successor_frozen in visited:
                continue

            visited.add(successor_frozen)
            queue.append((successor_frozen, plan + [action]))

    return None, []


if __name__ == "__main__":
    initial = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}
    actions = [
        Action("Move(A,B)", {"At(Robot,A)"}, set(), {"At(Robot,B)"}, {"At(Robot,A)"}),
        Action("Move(B,A)", {"At(Robot,B)"}, set(), {"At(Robot,A)"}, {"At(Robot,B)"}),
        Action("Move(B,C)", {"At(Robot,B)"}, set(), {"At(Robot,C)"}, {"At(Robot,B)"}),
        Action("Move(C,B)", {"At(Robot,C)"}, set(), {"At(Robot,B)"}, {"At(Robot,C)"}),
        Action("PickUp(Package,A)", {"At(Robot,A)", "At(Package,A)"}, set(), {"Holding(Package)"}, {"At(Package,A)"}),
        Action("PickUp(Package,B)", {"At(Robot,B)", "At(Package,B)"}, set(), {"Holding(Package)"}, {"At(Package,B)"}),
        Action("Drop(Package,C)", {"At(Robot,C)", "Holding(Package)"}, set(), {"At(Package,C)"}, {"Holding(Package)"}),
    ]

    plan, states = solve_planning_problem(initial, actions, goal)
    print("Plan found:", plan is not None)
    if plan is not None:
        print("Plan:", [a.name for a in plan])
        print("States reached:")
        for i, state in enumerate(states):
            print(f"S{i}: {sorted(state)}")
    else:
        print("No plan found")
