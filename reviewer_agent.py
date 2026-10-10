"""
reviewer_agent.py

The Reviewer Agent - reviews the simulation report produced by runner.py.

It receives the simulation report (scenario results, totals, success rate,
list of failures) and returns a strict JSON object with the keys:
    issue, likely_cause, recommendation, needs_revision.

The pass/fail decision (needs_revision) is computed deterministically from
the measured report so that it never depends on how the model phrases its
answer; the Qwen model (via Ollama) is used to phrase the issue, likely
cause and recommendation, with a deterministic fallback when the model
output cannot be validated.

Public API:
    run_reviewer(simulation_report) -> dict
    build_deterministic_review(simulation_report) -> dict
    validate_review(review) -> dict
"""

import json
import os
import re

from ollama import chat

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

REQUIRED_KEYS = {"issue", "likely_cause", "recommendation", "needs_revision"}

SYSTEM_PROMPT = """You are a senior software review engineer. You receive a
JSON simulation report for a robot navigation function decide_next_move(state).
The report contains total_scenarios, passed_scenarios, failed_scenarios,
success_rate and a "failures" array; each failure gives the scenario name,
the sensor state, the expected action and the actual action.

Analyse the report and output ONE JSON object with EXACTLY these keys:
{
  "issue": "short description of the concrete problem found, or 'No issues found' when every scenario passed",
  "likely_cause": "the most likely reason the problem occurs",
  "recommendation": "a concrete, actionable improvement",
  "needs_revision": true
}

Rules:
1. "needs_revision" MUST be a JSON boolean: true when failed_scenarios is
   greater than 0, false when the success_rate is 1.0.
2. The other three values MUST be non-empty strings.
3. Base the review only on the report. Name the failing scenario(s) and the
   mismatched actions in the "issue" field.
4. Output ONLY the JSON object - no explanations, no markdown code fences,
   no extra text before or after it."""


def _strip_thinking(text):
    """Remove qwen3-style reasoning blocks from a model response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text):
    """Extract the first balanced-looking JSON object from a model response."""
    text = text.strip()
    # Fast path.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the text between the first '{' and the last '}'.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("No JSON object found in response", text, 0)


def validate_review(review):
    """Validate the reviewer output.

    Checks that it is a JSON object with exactly the required keys,
    "needs_revision" is a boolean and the other three fields are non-empty
    strings.  Returns the validated dictionary, otherwise raises
    ValueError.
    """
    if not isinstance(review, dict):
        raise ValueError(
            "Review must be a JSON object (dictionary), got {}.".format(
                type(review).__name__
            )
        )

    missing = REQUIRED_KEYS - set(review.keys())
    if missing:
        raise ValueError("Missing required key(s): {}.".format(sorted(missing)))

    extra = set(review.keys()) - REQUIRED_KEYS
    if extra:
        raise ValueError("Unexpected key(s): {}.".format(sorted(extra)))

    for key in ("issue", "likely_cause", "recommendation"):
        if not isinstance(review[key], str) or not review[key].strip():
            raise ValueError("'{}' must be a non-empty string.".format(key))

    if not isinstance(review["needs_revision"], bool):
        raise ValueError("'needs_revision' must be a boolean.")

    return review


def build_deterministic_review(simulation_report):
    """Build the review directly from the measured simulation report.

    needs_revision is true exactly when at least one scenario failed. The
    issue text names the failing scenarios; when everything passes the
    review reports no issues."""
    failed = simulation_report.get("failed_scenarios", 0)
    success_rate = simulation_report.get("success_rate", 0.0)
    failures = simulation_report.get("failures", [])

    if failed <= 0:
        return {
            "issue": "No issues found",
            "likely_cause": "All simulated scenarios produced the expected action.",
            "recommendation": "No revision needed; keep the current navigation logic.",
            "needs_revision": False,
        }

    names = ", ".join(
        sorted({item.get("name", "unknown") for item in failures})
    ) or "unknown scenarios"
    details = "; ".join(
        "{}: expected {}, got {}".format(
            item.get("name"), item.get("expected"), item.get("actual")
        ) for item in failures[:10]
    )

    return {
        "issue": "{} scenario(s) failed (success rate {:.2%}): {}".format(
            failed, float(success_rate), names
        ),
        "likely_cause": (
            "The decide_next_move function disagrees with the required "
            "behaviour in these states: {}".format(details)
        ),
        "recommendation": (
            "Revise decide_next_move so that it moves toward the goal when "
            "that direction is clear, otherwise chooses a clear side, and "
            "stops only when every direction is blocked; add the failing "
            "states as regression tests."
        ),
        "needs_revision": True,
    }


def call_qwen(simulation_report, deterministic_review, feedback=""):
    """Send the simulation report and the deterministic review to Qwen and
    return the raw response text (which should be the strict JSON object)."""
    user_content = (
        (feedback + "\n\n" if feedback else "")
        + "SIMULATION REPORT:\n"
        + json.dumps(simulation_report, indent=2, ensure_ascii=False)
        + "\n\nMEASURED DETERMINISTIC VERDICT (needs_revision must match):\n"
        + json.dumps(deterministic_review, indent=2, ensure_ascii=False)
    )

    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": 0.2, "num_predict": 800},
    )

    if hasattr(response, "message") and hasattr(response.message, "content"):
        return response.message.content
    if isinstance(response, dict):
        return response["message"]["content"]
    raise RuntimeError("Unexpected response from Qwen: {}".format(response))


def run_reviewer(simulation_report):
    """Accept the simulation report, build the deterministic review, ask
    Qwen to phrase the review, validate the model output, and return it.

    The boolean needs_revision is always taken from the deterministic
    review so the measured verdict is authoritative. If the model output
    cannot be validated after two attempts, the deterministic review is
    returned instead."""
    deterministic = build_deterministic_review(simulation_report)

    feedback = ""
    for _attempt in range(2):
        try:
            raw = call_qwen(simulation_report, deterministic, feedback)
            review = _extract_json(_strip_thinking(raw))
            review = validate_review(review)
            # The measured boolean always wins.
            review["needs_revision"] = deterministic["needs_revision"]
            return validate_review(review)
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            feedback = (
                "Your previous answer was invalid ({}). Reply with one JSON "
                "object only, using the exact four keys, with a boolean "
                "needs_revision.".format(exc)
            )

    # Deterministic fallback - always valid.
    return validate_review(dict(deterministic))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(here, "simulation_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    print(json.dumps(run_reviewer(report), indent=2, ensure_ascii=False))
