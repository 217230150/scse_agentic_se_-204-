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
planner_agent.py

The Planner Agent - the second agent of the system.

In a multi-agent system the output of one agent becomes the context for the
next agent.  The Planner Agent therefore receives ONLY the validated
requirements produced by the Analyst Agent (context isolation: it never sees
the whole Analyst conversation, just the final artifact) and turns them into a
concrete navigation plan for the robot.

Like the Analyst Agent, the final plan MUST come from the Qwen model (served
through Ollama), not from this Python program.  This module only:
    1. builds the prompt (role, task, constraints, output structure),
    2. sends the requirements to Qwen via the Ollama client,
    3. parses the model's JSON text into Python data (json.loads),
    4. validates the data (validate_plan),
    5. returns the validated plan.

Public API:
    run_planner(requirement) -> dict
    validate_plan(data) -> dict
"""

import json
import os

from ollama import chat

# Qwen model served by the local Ollama installation.
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

REQUIRED_KEYS = {"strategy", "decisions", "stop_condition"}
ALLOWED_ACTIONS = {"FORWARD", "LEFT", "RIGHT", "STOP"}

SYSTEM_PROMPT = """You are a senior robotics navigation planner. Your job is to
read the validated requirements produced by the requirements engineer (the
Analyst Agent) and turn them into a concrete, machine-readable navigation plan
for a robot.

The robot can only perform these actions: FORWARD, LEFT, RIGHT, STOP.

What the plan must contain:
- "strategy": a short paragraph describing the overall navigation strategy
  (how the robot moves toward its goal while avoiding obstacles, and when it
  must stop safely).
- "decisions": a list of short decision rules the robot can apply while
  navigating (for example "if an obstacle blocks the path ahead, turn LEFT",
  "if the left side is blocked, turn RIGHT", "if no direction is safe, STOP").
  Each item is a string.
- "stop_condition": the exact point at which the robot must stop, as a string
  (for example reaching the goal, or when no direction is safe to move into).

Rules you MUST follow:
1. Base every part of the plan ONLY on the given requirements. Do not invent
   new capabilities or new actions (the robot can only FORWARD, LEFT, RIGHT,
   STOP).
2. Your entire reply must be a single valid JSON object with EXACTLY these
   three keys: strategy, decisions, stop_condition. No missing keys, no extra
   keys.
3. "strategy" must be a non-empty string.
4. "decisions" must be a non-empty list of non-empty strings.
5. "stop_condition" must be a non-empty string.
6. Output ONLY the JSON object - no explanations, no markdown code fences, no
   extra text before or after.

Required output structure:
{
  "strategy": "...",
  "decisions": ["...", "..."],
  "stop_condition": "..."
}"""


def call_qwen(requirement):
    """Send the system prompt + the validated requirements to Qwen through
    Ollama and return the raw response text (which should be a JSON object)."""
    user_content = json.dumps(requirement, indent=2, ensure_ascii=False)

    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": 0.2},
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


def validate_plan(data):
    """Validate the plan produced by Qwen (already converted to a Python data
    structure with json.loads).

    Checks that:
    - the data is a valid dictionary,
    - all the necessary keys exist (strategy, decisions, stop_condition),
    - "strategy" is a non-empty string,
    - "decisions" is a non-empty list of non-empty strings,
    - "stop_condition" is a non-empty string,
    - optionally, if Qwen also copied requirement fields, "goal" is a
      non-empty string and "allowed_actions" contains only valid actions
      (FORWARD/LEFT/RIGHT/STOP).

    Returns the validated dictionary on success, otherwise raises
    ValueError with a clear message.
    """
    if not isinstance(data, dict):
        raise ValueError(
            "Plan must be a JSON object (dictionary), got {}.".format(
                type(data).__name__
            )
        )

    # No missing keys.
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError("Missing required key(s): {}.".format(sorted(missing)))

    # "strategy" is a non-empty string.
    if not isinstance(data["strategy"], str) or not data["strategy"].strip():
        raise ValueError("'strategy' must be a non-empty string.")

    # "decisions" is a non-empty list of non-empty strings.
    decisions = data["decisions"]
    if not isinstance(decisions, list):
        raise ValueError(
            "'decisions' must be a list, got {}.".format(type(decisions).__name__)
        )
    if not decisions:
        raise ValueError("'decisions' must not be empty.")
    for decision in decisions:
        if not isinstance(decision, str) or not decision.strip():
            raise ValueError(
                "Each item in 'decisions' must be a non-empty string, got {!r}.".format(
                    decision
                )
            )

    # "stop_condition" is a non-empty string.
    if not isinstance(data["stop_condition"], str) or not data["stop_condition"].strip():
        raise ValueError("'stop_condition' must be a non-empty string.")

    # If Qwen copied any requirement fields as well, keep them honest.
    if "goal" in data and (not isinstance(data["goal"], str) or not data["goal"].strip()):
        raise ValueError("'goal' must be a non-empty string.")
    if "allowed_actions" in data:
        actions = data["allowed_actions"]
        if not isinstance(actions, list) or not actions:
            raise ValueError("'allowed_actions' must be a non-empty list.")
        for action in actions:
            if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
                raise ValueError(
                    "Invalid action {!r}. Allowed actions are only: {}.".format(
                        action, sorted(ALLOWED_ACTIONS)
                    )
                )
    if "safe_stop" in data and not isinstance(data["safe_stop"], bool):
        raise ValueError("'safe_stop' must be a boolean.")
    if "avoid.obstacles" in data and not isinstance(data["avoid.obstacles"], bool):
        raise ValueError("'avoid.obstacles' must be a boolean.")

    return data


def _strip_code_fence(text):
    """Remove a wrapping markdown code fence, if Qwen adds one."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def run_planner(requirement):
    """Accept the validated requirements artifact as a parameter, send it to
    the Planner Agent (Qwen via Ollama), accept Qwen's response, convert the
    JSON text into Python data (json.loads), validate the data, and return
    the validated plan."""
    response_text = call_qwen(requirement)

    cleaned = _strip_code_fence(response_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        snippet = cleaned[:300]
        raise ValueError(
            "Qwen did not return valid JSON ({}). Raw response starts with: {!r}".format(
                exc, snippet
            )
        ) from exc

    return validate_plan(data)


if __name__ == "__main__":
    # Small self-test against a known-good requirements artifact.
    sample = {
        "goal": "Navigate the robot toward its goal while avoiding blocked paths.",
        "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
        "safe_stop": True,
        "avoid.obstacles": True,
    }
    print("Running the Planner Agent on the sample requirements...")
    result = run_planner(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
