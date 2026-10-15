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
test_agent.py

The Test Agent - the fourth agent of the integrated system.

Following the multi-agent principle, it receives the validated requirements
(Analyst), the validated plan (Planner), and the navigation code produced by
the Developer Agent, and it writes pytest-style tests that exercise the
decide_next_move(state) function across every meaningful scenario.

Like the other agents, the test code MUST come from the Qwen model (served
through Ollama), not from this Python program.  This module only:
    1. builds the prompt (role, task, input structure, coverage, format),
    2. sends requirements + plan + navigation code to Qwen,
    3. extracts the test code from the model response,
    4. validates the code (validate_test_code),
    5. returns the validated test code as a string.

Public API:
    run_test_agent(requirements, plan, navigation_code) -> str
    validate_test_code(code) -> str
"""

import json
import os
import re

from ollama import chat

# Qwen model served by the local Ollama installation.
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

SYSTEM_PROMPT = """You are a senior software test engineer specialising in
robot navigation. You receive:
    1. the validated software requirements (JSON),
    2. the validated navigation plan (JSON),
    3. the navigation code under test, which defines decide_next_move(state).

The function under test has this signature and contract:
    def decide_next_move(state): ...
    # state is a dict with EXACTLY six boolean keys:
    #   goal_ahead, goal_on_left, goal_on_right,
    #   front_blocked, left_blocked, right_blocked
    # it returns exactly one of: "FORWARD", "LEFT", "RIGHT", "STOP".

Write a SINGLE pytest test module that covers ALL of the following scenarios:
    - goal ahead and the front is clear            -> FORWARD
    - goal on the left and the left is clear       -> LEFT
    - goal on the right and the right is clear     -> RIGHT
    - goal ahead but front blocked, left clear     -> LEFT
    - goal ahead but front and left blocked,
      right clear                                  -> RIGHT
    - goal on the left but left blocked,
      front clear                                  -> FORWARD
    - goal on the right but right blocked,
      front clear                                  -> FORWARD
    - front blocked, left and right clear          -> LEFT
    - every direction blocked                      -> STOP
    - goal ahead and every direction blocked       -> STOP
    - no goal flag set and the front is clear      -> FORWARD
    - every returned value is one of the allowed
      actions FORWARD, LEFT, RIGHT, STOP.

Hard requirements for the test module:
    1. Plain, valid Python 3 using pytest. Do NOT use unittest classes.
    2. The module MUST import the function under test exactly like this:

           import os
           import sys
           sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
           from generated.navigation_logic import decide_next_move

    3. Use several small functions whose names start with "test_" and build
       the state dict explicitly in each test, for example:
           state = {
               "goal_ahead": True, "goal_on_left": False,
               "goal_on_right": False, "front_blocked": False,
               "left_blocked": False, "right_blocked": False,
           }
           assert decide_next_move(state) == "FORWARD"
       Every test function MUST be declared with an EMPTY parameter list
       (def test_name():). Never give a test function parameters or pytest
       fixtures - build the state dict inside the function body instead.
    4. Every test function must contain at least one assert that compares
       decide_next_move(state) to an expected action string.
    5. Only the Python standard library and pytest may be imported.
    6. Output ONLY the Python code - no explanations, no markdown code
       fences, no extra text before or after the code."""


def _strip_thinking(text):
    """Remove qwen3-style reasoning blocks from a model response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code_fence(text):
    """Remove a wrapping markdown code fence, if the model adds one."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def call_qwen(requirements, plan, navigation_code, feedback=""):
    """Send the system prompt together with the requirements, the plan and
    the navigation code to Qwen through Ollama and return the raw response
    text (which should be pytest code). When feedback is given it is prepended
    to the user message so the model can correct a previous invalid answer."""
    user_content = (
        (feedback + "\n\n" if feedback else "")
        + "VALIDATED REQUIREMENTS:\n"
        + json.dumps(requirements, indent=2, ensure_ascii=False)
        + "\n\nVALIDATED PLAN:\n"
        + json.dumps(plan, indent=2, ensure_ascii=False)
        + "\n\nNAVIGATION CODE UNDER TEST:\n"
        + navigation_code
        + "\n\nWrite the pytest module now."
    )

    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": 0.2, "num_predict": 3000},
    )

    # ollama returns a ChatResponse object (pydantic model).
    if hasattr(response, "message"):
        message = response.message
        if hasattr(message, "content"):
            return message.content
    # Fallback for dict-style responses.
    if isinstance(response, dict):
        try:
            return response["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected response from Qwen: {}".format(response)
            ) from exc
    raise RuntimeError("Unexpected response from Qwen: {}".format(response))


def validate_test_code(code):
    """Validate the pytest code produced by Qwen.

    Checks that:
    - the code is a non-empty string,
    - it contains no markdown code fences,
    - it imports the function under test,
    - it defines pytest test functions with assertions,
    - it compiles as valid Python 3.

    Returns the validated code string on success, otherwise raises
    ValueError with a clear message.
    """
    if not isinstance(code, str):
        raise ValueError(
            "Generated test code must be a string, got {}.".format(
                type(code).__name__
            )
        )

    if not code.strip():
        raise ValueError("Generated test code is empty.")

    # No markdown fences allowed in the saved module.
    if "```" in code:
        raise ValueError("Generated test code must not contain markdown fences.")

    # Must import the function under test.
    if "decide_next_move" not in code:
        raise ValueError(
            "Generated test code must reference decide_next_move."
        )
    if not re.search(r"^\s*(import|from)\s+", code, flags=re.MULTILINE):
        raise ValueError("Generated test code must contain import statements.")

    # Must define pytest-style test functions.
    if not re.search(r"def\s+test_[A-Za-z0-9_]+\s*\(", code):
        raise ValueError(
            "Generated test code must define functions named test_*."
        )

    # Must contain assertions.
    if "assert " not in code:
        raise ValueError("Generated test code must contain assert statements.")

    # Test functions must take no parameters (pytest would treat any
    # parameter as a fixture and error at collection time).
    for match in re.finditer(r"def\s+(test_[A-Za-z0-9_]+)\s*\(([^)]*)\)", code):
        if match.group(2).strip():
            raise ValueError(
                "Test function '{}' must be declared with an empty parameter "
                "list (def {}()): build the state dict inside the body "
                "instead.".format(match.group(1), match.group(1))
            )

    # Any stdlib module used via attribute access must actually be imported.
    for name in ("sys", "os"):
        if re.search(r"\b{}\.".format(name), code) and not re.search(
            r"^\s*import\s+{}\b".format(name), code, flags=re.MULTILINE
        ):
            raise ValueError(
                "The module uses '{}' but never imports it.".format(name)
            )

    # Must compile as valid Python 3.
    try:
        compile(code, "<generated-tests>", "exec")
    except SyntaxError as exc:
        raise ValueError(
            "Generated test code is not valid Python ({}).".format(exc)
        ) from exc

    return code


def run_test_agent(requirements, plan, navigation_code, feedback=""):
    """Accept the validated requirements, the validated plan and the
    navigation code as parameters, send them to the Test Agent (Qwen via
    Ollama), extract the pytest code from the response, validate it, and
    return the validated test code as a string.

    The model is asked up to three times; a response that fails validation
    triggers one more attempt with the validation error fed back to it. An
    externally supplied feedback string (for example, pytest failures from a
    previous test-code generation) is forwarded on the first attempt."""
    last_error = None

    for _attempt in range(1, 4):
        response_text = call_qwen(requirements, plan, navigation_code, feedback)
        cleaned = _strip_code_fence(_strip_thinking(response_text))

        try:
            return validate_test_code(cleaned)
        except ValueError as exc:
            last_error = exc
            feedback = (
                "Your previous answer was invalid: {}. "
                "Output ONLY valid pytest Python code, with no markdown "
                "fences, importing decide_next_move and using test_* "
                "functions with asserts.".format(exc)
            )

    raise ValueError(
        "Test Agent failed to produce valid pytest code after 3 attempts: "
        "{}".format(last_error)
    )


if __name__ == "__main__":
    # Small self-test using the artifacts already present in the project.
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "artifacts", "requirements.json"), "r",
              encoding="utf-8") as f:
        sample_requirements = json.load(f)
    with open(os.path.join(here, "artifacts", "plan.json"), "r",
              encoding="utf-8") as f:
        sample_plan = json.load(f)
    with open(os.path.join(here, "generated", "navigation_logic.py"), "r",
              encoding="utf-8") as f:
        sample_code = f.read()

    print("Running the Test Agent...")
    print(run_test_agent(sample_requirements, sample_plan, sample_code))
