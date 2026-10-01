from planner import Action, apply_action, is_applicable, solve_planning_problem


def build_warehouse_actions():
    return [
        Action(
            name="Move(A,B)",
            positive_preconditions={"At(Robot,A)"},
            negative_preconditions=set(),
            positive_effects={"At(Robot,B)"},
            negative_effects={"At(Robot,A)"},
        ),
        Action(
            name="Move(B,A)",
            positive_preconditions={"At(Robot,B)"},
            negative_preconditions=set(),
            positive_effects={"At(Robot,A)"},
            negative_effects={"At(Robot,B)"},
        ),
        Action(
            name="Move(B,C)",
            positive_preconditions={"At(Robot,B)"},
            negative_preconditions=set(),
            positive_effects={"At(Robot,C)"},
            negative_effects={"At(Robot,B)"},
        ),
        Action(
            name="Move(C,B)",
            positive_preconditions={"At(Robot,C)"},
            negative_preconditions=set(),
            positive_effects={"At(Robot,B)"},
            negative_effects={"At(Robot,C)"},
        ),
        Action(
            name="PickUp(Package,A)",
            positive_preconditions={"At(Robot,A)", "At(Package,A)"},
            negative_preconditions=set(),
            positive_effects={"Holding(Package)"},
            negative_effects={"At(Package,A)"},
        ),
        Action(
            name="PickUp(Package,B)",
            positive_preconditions={"At(Robot,B)", "At(Package,B)"},
            negative_preconditions=set(),
            positive_effects={"Holding(Package)"},
            negative_effects={"At(Package,B)"},
        ),
        Action(
            name="Drop(Package,C)",
            positive_preconditions={"At(Robot,C)", "Holding(Package)"},
            negative_preconditions=set(),
            positive_effects={"At(Package,C)"},
            negative_effects={"Holding(Package)"},
        ),
    ]


def test_action_applicability_checks_preconditions():
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    pickup = Action(
        name="PickUp(Package,A)",
        positive_preconditions={"At(Robot,A)", "At(Package,A)"},
        negative_preconditions=set(),
        positive_effects={"Holding(Package)"},
        negative_effects={"At(Package,A)"},
    )
    drop = Action(
        name="Drop(Package,C)",
        positive_preconditions={"At(Robot,C)", "Holding(Package)"},
        negative_preconditions=set(),
        positive_effects={"At(Package,C)"},
        negative_effects={"Holding(Package)"},
    )

    print("\nCASE: Action applicability")
    print(f"Initial state: {sorted(initial_state)}")
    print(f"Check: PickUp(Package,A) applicable? {is_applicable(pickup, initial_state)}")
    print(f"Check: Drop(Package,C) applicable? {is_applicable(drop, initial_state)}")

    assert is_applicable(pickup, initial_state)
    assert not is_applicable(drop, initial_state)


def test_solvable_problem_finds_valid_plan():
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}
    actions = build_warehouse_actions()

    print("\nCASE: Solvable problem")
    print(f"Initial state: {sorted(initial_state)}")
    print(f"Goal: {sorted(goal)}")

    plan, states = solve_planning_problem(initial_state, actions, goal)
    print("Output:")
    if plan is None:
        print("No plan found")
        print(f"States reached: {states}")
    else:
        print(f"Plan: {[action.name for action in plan]}")
        print("States reached:")
        for i, state in enumerate(states):
            print(f"  S{i}: {sorted(state)}")

    assert plan is not None
    assert goal.issubset(states[-1])
    assert any(action.name == "Drop(Package,C)" for action in plan)


def test_impossible_problem_reports_no_plan():
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}
    actions = [
        Action(
            name="Move(A,B)",
            positive_preconditions={"At(Robot,A)"},
            negative_preconditions=set(),
            positive_effects={"At(Robot,B)"},
            negative_effects={"At(Robot,A)"},
        ),
        Action(
            name="Move(B,C)",
            positive_preconditions={"At(Robot,B)"},
            negative_preconditions=set(),
            positive_effects={"At(Robot,C)"},
            negative_effects={"At(Robot,B)"},
        ),
    ]

    print("\nCASE: Impossible problem")
    print(f"Initial state: {sorted(initial_state)}")
    print(f"Goal: {sorted(goal)}")

    plan, states = solve_planning_problem(initial_state, actions, goal)
    print("Output:")
    if plan is None:
        print("No plan found")
        print(f"States reached: {states}")
    else:
        print(f"Plan: {[action.name for action in plan]}")
        print("States reached:")
        for i, state in enumerate(states):
            print(f"  S{i}: {sorted(state)}")

    assert plan is None
    assert states == []


def test_irrelevant_actions_do_not_count_as_goal_completion():
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}
    actions = build_warehouse_actions() + [
        Action(
            name="Move(A,C)",
            positive_preconditions={"At(Robot,A)"},
            negative_preconditions=set(),
            positive_effects={"At(Robot,C)"},
            negative_effects={"At(Robot,A)"},
        )
    ]

    print("\nCASE: Irrelevant action")
    print(f"Initial state: {sorted(initial_state)}")
    print(f"Goal: {sorted(goal)}")

    plan, states = solve_planning_problem(initial_state, actions, goal)
    print("Output:")
    if plan is None:
        print("No plan found")
        print(f"States reached: {states}")
    else:
        print(f"Plan: {[action.name for action in plan]}")
        print("States reached:")
        for i, state in enumerate(states):
            print(f"  S{i}: {sorted(state)}")

    assert plan is not None
    assert "At(Robot,C)" in states[-1]
    assert "At(Package,C)" in states[-1]
    assert plan[-1].name == "Drop(Package,C)"


def test_apply_action_updates_state_correctly():
    state = {"At(Robot,A)", "At(Package,A)"}
    action = Action(
        name="Move(A,B)",
        positive_preconditions={"At(Robot,A)"},
        negative_preconditions=set(),
        positive_effects={"At(Robot,B)"},
        negative_effects={"At(Robot,A)"},
    )

    new_state = apply_action(state, action)

    assert "At(Robot,A)" not in new_state
    assert "At(Robot,B)" in new_state
    assert "At(Package,A)" in new_state
