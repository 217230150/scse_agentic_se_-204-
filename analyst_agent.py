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
analyst_agent.py

The Analyst Agent - the first (almost) agent of the system.

Its ONLY responsibility is to analyse the brief text and convert it into
explicit software requirements.  It behaves like a human requirements
engineer: it takes the customer's requirements text, extracts the specific
requirements from it, and returns them as validated JSON with exactly the
structure that the rest of the system consumes.

The final output MUST come from the Qwen model, not from this Python
program.  This module only:
    1. builds the prompt (role, task, constraints, output structure),
    2. sends the prompt + brief text to Qwen,
    3. parses the model's JSON text into Python data (json.loads),
    4. validates the data (validate_requirements),
    5. returns the validated requirements.

Public API:
    run_analyst(brief_text) -> dict
    validate_requirements(requirements) -> dict
"""

import json
import os
import urllib.error
import urllib.request

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

REQUIRED_KEYS = {"goal", "allowed_actions", "safe_stop", "avoid.obstacles"}
ALLOWED_ACTIONS = {"FORWARD", "LEFT", "RIGHT", "STOP"}

SYSTEM_PROMPT = """You are a senior requirements engineer. Your job is to read a
human-language brief for a software system and turn it into a small set of
explicit, machine-readable software requirements.

What you must do:
- Read the brief text carefully.
- Extract the navigation objective and write it into the "goal" field as a
  short string.
- List the valid steps the robot can take in "allowed_actions".

Rules you MUST follow:
1. The robot's actions are limited to exactly: FORWARD, LEFT, RIGHT, STOP.
   Do not invent or add any other action.
2. Set "safe_stop" to true if the robot must stop safely when no direction
   is safe to move into.
3. Set "avoid.obstacles" to true if the robot must avoid anything that
   blocks its path.
4. Your entire reply must be a single valid JSON object with EXACTLY the
   keys shown in the structure below: no missing keys, no extra keys.
5. "goal" must be a string, "allowed_actions" must be a list,
   "safe_stop" and "avoid.obstacles" must be booleans.
6. Output ONLY the JSON object - no explanations, no markdown code fences,
   no extra text before or after.

Required output structure:
{
  "goal": "string",
  "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
  "safe_stop": true,
  "avoid.obstacles": true
}"""


def call_qwen(brief_text):
    """Send the system prompt + brief text to Qwen and return the raw
    response text (which should be a JSON object)."""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Environment variable DASHSCOPE_API_KEY is not set. "
            "Get a key from https://bailian.console.aliyun.com/ and set it, e.g. in "
            "PowerShell:  $env:DASHSCOPE_API_KEY = \"sk-...\"  "
            "or in CMD:  set DASHSCOPE_API_KEY=sk-..."
        )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": brief_text},
        ],
        "temperature": 0.2,
    }

    request = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("Qwen API error {}: {}".format(exc.code, detail)) from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected response from Qwen API: {}".format(body)) from exc


def validate_requirements(requirements):
    """Validate the data produced by Qwen (already converted to a Python
    data structure with json.loads).

    Checks that:
    - the data is a valid dictionary,
    - all the necessary keys exist (goal, allowed_actions, safe_stop,
      avoid.obstacles) and NO new key was introduced by Qwen,
    - "goal" is a string,
    - "allowed_actions" is a list of valid actions (FORWARD/LEFT/RIGHT/STOP),
    - "safe_stop" and "avoid.obstacles" are booleans.

    Returns the validated dictionary on success, otherwise raises
    ValueError with a clear message.
    """
    if not isinstance(requirements, dict):
        raise ValueError(
            "Requirements must be a JSON object (dictionary), got {}.".format(
                type(requirements).__name__
            )
        )

    # No missing keys.
    missing = REQUIRED_KEYS - set(requirements.keys())
    if missing:
        raise ValueError(
            "Missing required key(s): {}.".format(sorted(missing))
        )

    # No new keys introduced by Qwen.
    extra = set(requirements.keys()) - REQUIRED_KEYS
    if extra:
        raise ValueError(
            "Unexpected key(s) introduced by Qwen: {}.".format(sorted(extra))
        )

    # "goal" is a string.
    if not isinstance(requirements["goal"], str) or not requirements["goal"].strip():
        raise ValueError("'goal' must be a non-empty string.")

    # "allowed_actions" is a list of valid actions.
    actions = requirements["allowed_actions"]
    if not isinstance(actions, list):
        raise ValueError("'allowed_actions' must be a list, got {}.".format(
            type(actions).__name__
        ))
    if not actions:
        raise ValueError("'allowed_actions' must not be empty.")
    for action in actions:
        if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
            raise ValueError(
                "Invalid action {!r}. Allowed actions are only: {}.".format(
                    action, sorted(ALLOWED_ACTIONS)
                )
            )

    # "safe_stop" and "avoid.obstacles" are booleans.
    if not isinstance(requirements["safe_stop"], bool):
        raise ValueError("'safe_stop' must be a boolean.")
    if not isinstance(requirements["avoid.obstacles"], bool):
        raise ValueError("'avoid.obstacles' must be a boolean.")

    return requirements


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


def run_analyst(brief_text):
    """Accept the brief text as a parameter, send appropriate prompts to the
    Qwen model, accept Qwen's response, convert the JSON text into Python
    data (json.loads), validate the data, and return the validated
    requirements."""
    response_text = call_qwen(brief_text)

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

    return validate_requirements(data)
