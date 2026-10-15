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
developer_agent.py

The Developer Agent - the third agent of the system.

Following the multi-agent principle, its input is ONLY the validated
navigation plan produced by the Planner Agent (context isolation: it never
sees the Analyst conversation or the raw requirements, just the plan
artifact).  It turns the plan into actual Python code that makes the robot
navigate.

Like the other agents, the code MUST come from the Qwen model (served through
Ollama), not from this Python program.  This module only:
    1. builds the prompt (role, task, constraints, output format),
    2. sends the plan to Qwen via the Ollama client,
    3. extracts the Python code from the model's response,
    4. validates the code (validate_code),
    5. returns the validated code as a string.

Public API:
    run_developer(plan) -> str
    validate_code(code) -> str
    run_developer_revision(original_code, review_notes, benchmark_report,
                           requirements, plan) -> str
"""

import json
import os
import re

from ollama import chat

# Qwen model served by the local Ollama installation.
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

# Model used for the post-review revision. A larger model is used by default
# because the revision must reconcile code, reviews and benchmark data; it can
# be overridden with the REVISION_MODEL environment variable.
REVISION_MODEL = os.environ.get("REVISION_MODEL", "qwen3:8b")

ALLOWED_ACTIONS = {"FORWARD", "LEFT", "RIGHT", "STOP"}

SYSTEM_PROMPT = """You are a senior robot software engineer. You receive a
validated navigation plan (a JSON object) and you write the actual Python code
that makes the robot navigate according to that plan.

The robot's sensor input "state" is a dictionary with EXACTLY these six
boolean keys:
    goal_ahead, goal_on_left, goal_on_right,
    front_blocked, left_blocked, right_blocked

Requirements for the code you produce:
1. It must be plain, valid Python 3 code that can be executed as-is.
2. It MUST define the main entry function decide_next_move(state) that takes
   the sensor state dictionary and returns one of the allowed actions. Other
   programs will call ONLY decide_next_move(state) - they will not call any
   helper function inside it.
3. It MUST also retain the original navigation logic: keep the original
   decision function (decide_next_action) AND add the decide_next_move(state)
   wrapper. Use the reference implementation below - you may rename/extend
   helper functions and add docstrings, but the core if/elif logic MUST be
   kept:

   def decide_next_action(blocked_ahead, blocked_left, blocked_right,
                          goal_reached):
       if goal_reached:
           return "STOP"
       if not blocked_ahead:
           return "FORWARD"
       if not blocked_left:
           return "LEFT"
       if not blocked_right:
           return "RIGHT"
       return "STOP"

   def decide_next_move(state):
       goal_ahead = state["goal_ahead"]
       goal_on_left = state["goal_on_left"]
       goal_on_right = state["goal_on_right"]
       front_blocked = state["front_blocked"]
       left_blocked = state["left_blocked"]
       right_blocked = state["right_blocked"]
       # Prefer moving toward the goal whenever it is safe.
       if goal_ahead and not front_blocked:
           return "FORWARD"
       if goal_on_left and not left_blocked:
           return "LEFT"
       if goal_on_right and not right_blocked:
           return "RIGHT"
       # Otherwise avoid obstacles and keep moving (original logic).
       if not front_blocked:
           return "FORWARD"
       if not left_blocked:
           return "LEFT"
       if not right_blocked:
           return "RIGHT"
       # No safe direction: stop.
       return "STOP"

4. The code may ONLY use the actions listed in the plan's "allowed_actions"
   (FORWARD, LEFT, RIGHT, STOP) - nothing else.
5. Use only the Python standard library. Do not import any external library.
6. Do not read files, do not open network connections, do not call eval/exec,
   do not ask for user input. Keep it a pure, deterministic logic module.
7. Output ONLY the Python code - no explanations, no markdown code fences, no
   extra text before or after.

You will receive the plan as the user message. Write the code now."""


REVISION_SYSTEM_PROMPT = """You are a senior robot software engineer revising
an existing navigation module after testing and review. You receive:
    1. the ORIGINAL navigation code (it defines decide_next_move(state)),
    2. the reviewer notes from the logical simulation,
    3. the IR-SIM benchmark report (per-world reached_goal, collision_or_done,
       steps_taken, efficiency_status, final_pose) and its review,
    4. the validated requirements and the validated navigation plan.

Your job is to produce a CORRECTED version of decide_next_move(state) that
fixes every concrete problem listed in the reviews/benchmark.

Hard requirements for the revised code:
1. Plain, valid Python 3, standard library only. No imports, no file or
   network access, no eval/exec, no user input. It must stay a pure,
   deterministic logic module.
2. It MUST define decide_next_move(state) taking a dict with EXACTLY these
   six boolean keys:
       goal_ahead, goal_on_left, goal_on_right,
       front_blocked, left_blocked, right_blocked
   and return ONLY one of: "FORWARD", "LEFT", "RIGHT", "STOP".
3. Correct behaviour (the full navigation contract):
   - Never return a direction that is blocked.
   - When the goal lies in a direction that is not blocked, move that way
     (goal ahead + clear front => FORWARD; goal left + clear left => LEFT;
     goal right + clear right => RIGHT).
   - When no goal flag is set, keep moving to explore: if the front is not
     blocked return FORWARD; otherwise LEFT if clear, otherwise RIGHT.
   - When the front is blocked, choose a side that is not blocked; bias the
     choice toward the side the goal is on, and do NOT alternate sides on
     consecutive equivalent states (this prevents oscillation).
   - Return STOP ONLY when every direction (front, left and right) is
     blocked. The caller detects goal reaching; STOP must never be used as
     a substitute for moving when any direction is clear.
4. Keep the implementation small and explicit; helper functions are allowed
   but decide_next_move(state) must be fully self-contained in its decisions.
5. Output ONLY the Python code - no explanations, no markdown code fences,
   no thinking text, no extra words before or after the code."""


def _message_text(response):
    """Extract the answer text from an Ollama chat response.

    Reasoning models (e.g. qwen3) can put the whole answer into the
    'thinking' field and leave 'content' empty.  Fall back to 'thinking'
    in that case so the pipeline never receives an empty string."""
    if hasattr(response, "message"):
        message = response.message
        content = getattr(message, "content", None)
        if content and content.strip():
            return content
        thinking = getattr(message, "thinking", None)
        if thinking and thinking.strip():
            return thinking
        if content is not None:
            return content
    if isinstance(response, dict):
        message = response.get("message", {})
        content = message.get("content") or ""
        if content.strip():
            return content
        return message.get("thinking") or content
    raise RuntimeError("Unexpected response from Qwen: {}".format(response))


def call_qwen(plan):
    """Send the system prompt + the validated plan to Qwen through Ollama and
    return the raw response text (which should be Python code)."""
    user_content = json.dumps(plan, indent=2, ensure_ascii=False)

    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        think=False,
        # temperature 0 makes code generation deterministic so repeated
        # runs of the pipeline produce the same navigation logic.
        options={"temperature": 0},
    )

    return _message_text(response)
    # Fallback for dict-style responses.
    if isinstance(response, dict):
        try:
            return response["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected response from Qwen: {}".format(response)
            ) from exc
    raise RuntimeError("Unexpected response from Qwen: {}".format(response))


def _strip_code_fence(text):
    """Remove qwen3-style reasoning blocks and a wrapping markdown code
    fence, if the model adds either of them."""
    # Reasoning models may emit <think>...</think> before the answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def validate_code(code):
    """Validate the Python code produced by Qwen.

    Checks that:
    - the code is a string,
    - it compiles as valid Python 3 (compile with mode='exec'),
    - it defines the mandatory decide_next_move(state) entry function,
    - it defines at least one function,
    - it does not contain obviously dangerous constructs.

    Returns the validated (cleaned) code string on success, otherwise raises
    ValueError with a clear message.
    """
    if not isinstance(code, str):
        raise ValueError(
            "Generated code must be a string, got {}.".format(type(code).__name__)
        )

    if not code.strip():
        raise ValueError("Generated code is empty.")

    # (Simple sanity check: no obviously forbidden constructs.)
    forbidden = ["eval(", "exec(", "__import__", "subprocess", "socket", "open("]
    lowered = code.lower()
    for token in forbidden:
        if token.lower() in lowered:
            raise ValueError(
                "Generated code contains forbidden construct {!r}.".format(token)
            )

    # Must compile as valid Python 3.
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        snippet = code[:300]
        raise ValueError(
            "Generated code is not valid Python ({}). Code starts with: {!r}".format(
                exc, snippet
            )
        ) from exc

    # Must define the mandatory decide_next_move(state) entry function.
    if "def decide_next_move" not in code:
        raise ValueError(
            "Generated code must define the entry function decide_next_move(state)."
        )

    # Must define at least one function meaningful for navigation.
    if "def " not in code:
        raise ValueError(
            "Generated code must define at least one function (e.g. "
            "decide_next_move)."
        )

    return code


def run_developer(plan):
    """Accept the validated plan artifact as a parameter, send it to the
    Developer Agent (Qwen via Ollama), accept Qwen's response, extract the
    Python code, validate it, and return the validated code as a string."""
    response_text = call_qwen(plan)

    cleaned = _strip_code_fence(response_text)
    return validate_code(cleaned)


def _as_text(value):
    """Render an artifact (dict/list) as readable text; strings pass through."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def call_qwen_revision(original_code, review_notes, benchmark_report,
                       requirements, plan, feedback=""):
    """Send the original code together with the review notes, benchmark
    report, requirements and plan to the revision model and return the raw
    response text (which should be the revised Python code)."""
    user_content = (
        (feedback + "\n\n" if feedback else "")
        + "ORIGINAL NAVIGATION CODE:\n"
        + original_code
        + "\n\nREVIEW NOTES (logical simulation):\n"
        + _as_text(review_notes)
        + "\n\nIR-SIM BENCHMARK REPORT AND REVIEW:\n"
        + _as_text(benchmark_report)
        + "\n\nVALIDATED REQUIREMENTS:\n"
        + _as_text(requirements)
        + "\n\nVALIDATED PLAN:\n"
        + _as_text(plan)
        + "\n\nWrite the FULL revised navigation module now."
    )

    response = chat(
        model=REVISION_MODEL,
        messages=[
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        think=False,
        # temperature 0 keeps the revision deterministic and reproducible.
        options={"temperature": 0, "num_predict": 2500},
    )

    return _message_text(response)


def run_developer_revision(original_code, review_notes, benchmark_report,
                           requirements, plan, out_path=None):
    """Produce a revised navigation module after testing and review.

    Receives the original navigation code, the review notes, the benchmark
    report (the benchmark review may be embedded), the requirements and the
    plan. Asks the revision LLM for a new navigation logic, validates the
    answer (up to three attempts, feeding validation errors back), and saves
    it as a NEW file (generated/navigation_logic_revised.py by default).
    Returns the validated revised code as a string."""
    if out_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(
            here, "generated", "navigation_logic_revised.py"
        )

    last_error = None
    feedback = ""

    for _attempt in range(1, 4):
        response_text = call_qwen_revision(
            original_code, review_notes, benchmark_report, requirements,
            plan, feedback
        )
        cleaned = _strip_code_fence(response_text)

        try:
            revised = validate_code(cleaned)
        except ValueError as exc:
            last_error = exc
            feedback = (
                "Your previous revised code was invalid: {}. Output ONLY "
                "valid plain Python code defining decide_next_move(state), "
                "with no thinking text and no markdown fences.".format(exc)
            )
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(revised)
            if not revised.endswith("\n"):
                f.write("\n")
        return revised

    raise ValueError(
        "Developer revision failed to produce valid code after 3 attempts: "
        "{}".format(last_error)
    )


if __name__ == "__main__":
    # Small self-test against a known-good plan artifact.
    sample_plan = {
        "goal": "Navigate the robot toward its goal while avoiding blocked paths.",
        "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
        "safe_stop": True,
        "avoid.obstacles": True,
        "strategy": (
            "Move FORWARD whenever the path ahead is clear; if an obstacle "
            "blocks the path ahead, try LEFT then RIGHT; stop when the goal "
            "is reached or when no direction is safe."
        ),
        "decisions": [
            "If the path ahead is clear, go FORWARD.",
            "If an obstacle blocks the path ahead and the left is clear, go LEFT.",
            "If the left is blocked too, try RIGHT.",
            "If every direction is blocked, STOP.",
        ],
        "stop_condition": "Stop when the goal is reached, or when no direction "
        "is safe to move into.",
    }
    print("Running the Developer Agent on the sample plan...")
    code = run_developer(sample_plan)
    print(code)
