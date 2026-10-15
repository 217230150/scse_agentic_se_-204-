# ============================================================================
# AI Usage Statement (course requirement: all AI usage must be documented)
# Model environment: this project was developed and tested with Qwen
# models served locally by Ollama (pipeline model qwen2.5:1.5b, revision
# model qwen3:8b; the Analyst stage uses the DashScope Qwen API). No paid
# model services were used, and no prohibited agent frameworks (Dify /
# MetaGPT etc.) -- the agents are plain Python implementations calling the
# Ollama API directly.
# Reason for using AI: to keep operations convenient, the code concise and
# readable, and to help later reviewers understand and check the code
# logic. Prompts, raw model outputs and human revisions are recorded in
# trace.log and the artifacts/ directory, and in the per-file note below.
# AI involvement: the overall structure and core logic of this file were
# written by the team; prompt engineering and the validation / revision-
# loop sections (model-response extraction, generated-test validation,
# final version adoption) were drafted with AI assistance and then
# reviewed by hand and verified by real runs. Runtime artifacts produced
# by Qwen (navigation logic, test module, plan, etc.) are stored under
# generated/ and artifacts/.
# ============================================================================
"""
test_generated_navigation_logic.py

Level 2 - Behavioral testing of the generated navigation logic.

What it does:
    1. loads decide_next_move(state) from the navigation logic artifact
       (generated/navigation_logic.py),
    2. builds test cases for every known robot state - the state space is
       six booleans (goal_ahead, goal_on_left, goal_on_right, front_blocked,
       left_blocked, right_blocked), so all 2**6 = 64 states are tested,
    3. for each state, passes the state to decide_next_move, checks that the
       result is a valid action and that it is also the correct expected
       move,
    4. also keeps explicit sample test cases in the style of the shared
       test_generated_navigation_logic.py.

Expected behaviour (from the brief + the plan):
    - prefer moving toward the goal whenever that direction is not blocked,
    - never move into a blocked direction,
    - if no safe direction exists, STOP.

Usage:
    python test_generated_navigation_logic.py
"""

import importlib.util
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_PATH = os.path.join(HERE, "generated", "navigation_logic.py")

ALLOWED_ACTIONS = {"FORWARD", "LEFT", "RIGHT", "STOP"}
STATE_KEYS = (
    "goal_ahead",
    "goal_on_left",
    "goal_on_right",
    "front_blocked",
    "left_blocked",
    "right_blocked",
)


def load_decide_next_move(path):
    """Load the decide_next_move function from the artifact file."""
    if not os.path.exists(path):
        raise RuntimeError("Navigation logic artifact not found: {}".format(path))
    spec = importlib.util.spec_from_file_location("navigation_logic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, "decide_next_move", None)
    if func is None:
        raise RuntimeError(
            "Artifact {} does not define decide_next_move(state).".format(path)
        )
    return func


def expected_move(state):
    """Reference implementation of the expected behaviour used by the tests.

    This is NOT the code under test - it is an independent expectation so the
    test can tell the generated logic apart from the intended behaviour.
    """
    if state["goal_ahead"] and not state["front_blocked"]:
        return "FORWARD"
    if state["goal_on_left"] and not state["left_blocked"]:
        return "LEFT"
    if state["goal_on_right"] and not state["right_blocked"]:
        return "RIGHT"
    if not state["front_blocked"]:
        return "FORWARD"
    if not state["left_blocked"]:
        return "LEFT"
    if not state["right_blocked"]:
        return "RIGHT"
    return "STOP"


def all_states():
    """Every known robot state: all combinations of the six booleans."""
    for values in itertools.product([True, False], repeat=len(STATE_KEYS)):
        yield dict(zip(STATE_KEYS, values))


def sample_test_cases():
    """Explicit test cases in the style of the shared sample file.

    Each case is a (state, expected action) pair.
    """
    return [
        (  # typical sample case from the shared file
            {
                "goal_ahead": True,
                "goal_on_left": False,
                "goal_on_right": False,
                "front_blocked": False,
                "left_blocked": False,
                "right_blocked": False,
            },
            "FORWARD",
        ),
        (  # goal ahead but blocked: avoid left
            {
                "goal_ahead": True,
                "goal_on_left": False,
                "goal_on_right": False,
                "front_blocked": True,
                "left_blocked": False,
                "right_blocked": True,
            },
            "LEFT",
        ),
        (  # goal on left and free: go left
            {
                "goal_ahead": False,
                "goal_on_left": True,
                "goal_on_right": False,
                "front_blocked": False,
                "left_blocked": False,
                "right_blocked": False,
            },
            "LEFT",
        ),
        (  # goal on right and free: go right
            {
                "goal_ahead": False,
                "goal_on_left": False,
                "goal_on_right": True,
                "front_blocked": False,
                "left_blocked": False,
                "right_blocked": False,
            },
            "RIGHT",
        ),
        (  # no goal visible, front free: keep moving forward
            {
                "goal_ahead": False,
                "goal_on_left": False,
                "goal_on_right": False,
                "front_blocked": False,
                "left_blocked": True,
                "right_blocked": True,
            },
            "FORWARD",
        ),
        (  # everything blocked: safe stop
            {
                "goal_ahead": False,
                "goal_on_left": False,
                "goal_on_right": False,
                "front_blocked": True,
                "left_blocked": True,
                "right_blocked": True,
            },
            "STOP",
        ),
        (  # goal on left but left blocked, goal elsewhere: avoid right
            {
                "goal_ahead": False,
                "goal_on_left": True,
                "goal_on_right": False,
                "front_blocked": True,
                "left_blocked": True,
                "right_blocked": False,
            },
            "RIGHT",
        ),
    ]


def main():
    decide_next_move = load_decide_next_move(NAV_PATH)
    print("=== Navigation logic behavioral test ===")
    print("Testing decide_next_move(state) from: {}".format(NAV_PATH))
    print()

    failures = []
    tested = 0

    # 1) Exhaustive: every known robot state (all 64 combinations).
    for state in all_states():
        action = decide_next_move(state)
        tested += 1

        if action not in ALLOWED_ACTIONS:
            failures.append(
                "invalid action {!r} for state {}".format(action, state)
            )
            continue
        expected = expected_move(state)
        if action != expected:
            failures.append(
                "state {} -> {!r}, expected {!r}".format(state, action, expected)
            )

    # 2) Explicit sample test cases.
    for state, expected in sample_test_cases():
        action = decide_next_move(state)
        tested += 1
        if action not in ALLOWED_ACTIONS or action != expected:
            failures.append(
                "sample state {} -> {!r}, expected {!r}".format(state, action, expected)
            )

    if failures:
        print("FAILED: {} of {} checks did not match.".format(len(failures), tested))
        for failure in failures[:20]:
            print("  - " + failure)
        sys.exit(1)

    print("PASSED: {} checks passed (all 64 known states + {} sample cases).".format(
        tested, len(sample_test_cases())
    ))


if __name__ == "__main__":
    main()
