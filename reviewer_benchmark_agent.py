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
reviewer_benchmark_agent.py

The Benchmark Reviewer Agent - reviews irsim_benchmark_report.json.

It measures three quality requirements:
    safety            - no world ended in collision/terminal failure without
                        reaching the goal,
    goal completion   - every world reached its goal (success_rate == 1.0),
    efficiency        - every successful run finished within the efficient
                        step limit (efficiency_rate == 1.0, no
                        inefficient-success worlds).

It returns a strict JSON object with keys:
    issue, likely_cause, recommendation,
    safety_passed, goal_completion_passed, efficiency_passed,
    needs_revision.

The four booleans are computed deterministically from the measured benchmark
fields (build_deterministic_review); the Qwen model (via Ollama) phrases the
issue, likely cause and recommendation, with the deterministic review used
directly when the model output cannot be validated.

Public API:
    run_benchmark_reviewer(benchmark_report) -> dict
    build_deterministic_review(benchmark_report) -> dict
    validate_benchmark_review(review) -> dict
"""

import json
import os
import re

from ollama import chat

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

REQUIRED_KEYS = {
    "issue",
    "likely_cause",
    "recommendation",
    "safety_passed",
    "goal_completion_passed",
    "efficiency_passed",
    "needs_revision",
}
BOOLEAN_KEYS = {
    "safety_passed",
    "goal_completion_passed",
    "efficiency_passed",
    "needs_revision",
}
STRING_KEYS = {"issue", "likely_cause", "recommendation"}

SYSTEM_PROMPT = """You are a senior robot quality reviewer. You receive a JSON
benchmark report for the navigation logic tested in several IR-SIM worlds.
The report contains total_worlds, goals_reached, success_rate,
efficient_worlds, inefficient_success_worlds, failed_worlds,
efficiency_rate, average_steps_taken and a "results" array. Each result gives
world_name, reached_goal, collision_or_done, steps_taken,
efficiency_status and final_pose.

Analyse it and output ONE JSON object with EXACTLY these keys:
{
  "issue": "short description of the concrete problem found, or 'No issues found' when all quality requirements pass",
  "likely_cause": "the most likely reason for the problem",
  "recommendation": "a precise, actionable fix",
  "safety_passed": true,
  "goal_completion_passed": true,
  "efficiency_passed": true,
  "needs_revision": false
}

Rules:
1. safety_passed, goal_completion_passed, efficiency_passed and
   needs_revision MUST be JSON booleans and MUST match the measured
   deterministic verdict supplied with the report.
2. issue, likely_cause and recommendation MUST be non-empty strings.
3. Name the specific world(s), step counts and failure modes in the issue and
   give a concrete fix in the recommendation.
4. Output ONLY the JSON object - no explanations, no markdown code fences,
   no extra text before or after it."""


def _strip_thinking(text):
    """Remove qwen3-style reasoning blocks from a model response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text):
    """Extract the first JSON object from a model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("No JSON object found in response", text, 0)


def validate_benchmark_review(review):
    """Validate the benchmark reviewer output.

    Checks it is a JSON object with exactly the required keys, the four
    quality booleans are actual booleans and the three text fields are
    non-empty strings.  Raises ValueError otherwise."""
    if not isinstance(review, dict):
        raise ValueError(
            "Benchmark review must be a JSON object (dictionary), got {}.".format(
                type(review).__name__
            )
        )

    missing = REQUIRED_KEYS - set(review.keys())
    if missing:
        raise ValueError("Missing required key(s): {}.".format(sorted(missing)))

    extra = set(review.keys()) - REQUIRED_KEYS
    if extra:
        raise ValueError("Unexpected key(s): {}.".format(sorted(extra)))

    for key in STRING_KEYS:
        if not isinstance(review[key], str) or not review[key].strip():
            raise ValueError("'{}' must be a non-empty string.".format(key))

    for key in BOOLEAN_KEYS:
        if not isinstance(review[key], bool):
            raise ValueError("'{}' must be a boolean.".format(key))

    return review


def build_deterministic_review(benchmark_report):
    """Analyse the measured benchmark report and determine the four quality
    booleans plus phrased issue / likely cause / recommendation.

    Defaults are used when a field is missing:
      - safety is measured per world (collision_or_done true while the goal
        was not reached),
      - goal completion is the aggregate success_rate being 1.0,
      - efficiency is zero inefficient-success worlds and an efficiency
        rate of 1.0.
    """
    results = benchmark_report.get("results", [])
    total = benchmark_report.get("total_worlds", len(results))
    success_rate = benchmark_report.get("success_rate", 0.0)
    efficiency_rate = benchmark_report.get("efficiency_rate", 0.0)
    inefficient_success = benchmark_report.get(
        "inefficient_success_worlds", 0
    )

    unsafe_worlds = [
        r.get("world_name", "unknown")
        for r in results
        if r.get("collision_or_done") and not r.get("reached_goal")
    ]
    failed_worlds = [
        r.get("world_name", "unknown")
        for r in results
        if not r.get("reached_goal")
    ]
    inefficient_worlds = [
        "{} ({} steps)".format(r.get("world_name", "unknown"),
                               r.get("steps_taken"))
        for r in results
        if r.get("reached_goal")
        and r.get("efficiency_status") == "acceptable_but_inefficient"
    ]

    safety_passed = len(unsafe_worlds) == 0
    goal_completion_passed = (
        float(success_rate) >= 1.0 and len(failed_worlds) == 0
    )
    efficiency_passed = (
        int(inefficient_success) == 0
        and float(efficiency_rate) >= 1.0
        and len(inefficient_worlds) == 0
    )
    needs_revision = not (
        safety_passed and goal_completion_passed and efficiency_passed
    )

    issues = []
    causes = []
    recommendations = []

    if not safety_passed:
        issues.append("Unsafe benchmark behavior detected")
        causes.append(
            "the robot ended in collision/terminal failure in world(s): "
            + ", ".join(unsafe_worlds)
        )
        recommendations.append(
            "Add obstacle avoidance for the failing heading(s): never "
            "command a move into a blocked direction and turn away earlier "
            "(increase clearance) in these worlds."
        )

    if not goal_completion_passed:
        issues.append("Goal completion failure")
        causes.append(
            "only {}/{} worlds reached their goals; failed world(s): {}".format(
                benchmark_report.get("goals_reached", 0), total,
                ", ".join(failed_worlds)
            )
        )
        recommendations.append(
            "Revise decide_next_move to detour around obstacles while "
            "continuing to bias movement toward the goal so every world "
            "reaches its goal within the step limit."
        )

    if not efficiency_passed:
        issues.append("Successful but inefficient navigation")
        causes.append(
            "these successful runs exceeded the efficient step limit: "
            + ", ".join(inefficient_worlds)
        )
        recommendations.append(
            "Reduce oscillation: commit to a detour direction, avoid "
            "alternating LEFT/RIGHT, and resume goal-directed FORWARD "
            "motion as soon as the obstacle is cleared."
        )

    if needs_revision:
        return {
            "issue": "; ".join(issues),
            "likely_cause": "; ".join(causes),
            "recommendation": " ".join(recommendations),
            "safety_passed": safety_passed,
            "goal_completion_passed": goal_completion_passed,
            "efficiency_passed": efficiency_passed,
            "needs_revision": needs_revision,
        }

    return {
        "issue": "No issues found",
        "likely_cause": (
            "All worlds reached their goals safely within the efficient "
            "step limit."
        ),
        "recommendation": (
            "No revision needed; keep the current navigation logic."
        ),
        "safety_passed": True,
        "goal_completion_passed": True,
        "efficiency_passed": True,
        "needs_revision": False,
    }


def call_qwen(benchmark_report, deterministic_review, feedback=""):
    """Send the benchmark report and deterministic verdict to Qwen and
    return the raw response text."""
    user_content = (
        (feedback + "\n\n" if feedback else "")
        + "BENCHMARK REPORT:\n"
        + json.dumps(benchmark_report, indent=2, ensure_ascii=False)
        + "\n\nMEASURED DETERMINISTIC VERDICT (all booleans must match):\n"
        + json.dumps(deterministic_review, indent=2, ensure_ascii=False)
    )

    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": 0.2, "num_predict": 1000},
    )

    if hasattr(response, "message") and hasattr(response.message, "content"):
        return response.message.content
    if isinstance(response, dict):
        return response["message"]["content"]
    raise RuntimeError("Unexpected response from Qwen: {}".format(response))


def run_benchmark_reviewer(benchmark_report):
    """Accept the benchmark report, build the deterministic review, ask
    Qwen to phrase it, validate the model output, and return the validated
    dictionary. The measured booleans always overwrite the model's so the
    verdict is grounded in the measurements. The deterministic review is
    returned directly when the model cannot produce valid output."""
    deterministic = build_deterministic_review(benchmark_report)

    feedback = ""
    for _attempt in range(2):
        try:
            raw = call_qwen(benchmark_report, deterministic, feedback)
            review = _extract_json(_strip_thinking(raw))
            review = validate_benchmark_review(review)
            # Measured booleans always win.
            for key in BOOLEAN_KEYS:
                review[key] = deterministic[key]
            return validate_benchmark_review(review)
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            feedback = (
                "Your previous answer was invalid ({}). Reply with one JSON "
                "object only, using the exact seven keys, with booleans "
                "matching the deterministic verdict.".format(exc)
            )

    return validate_benchmark_review(dict(deterministic))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "artifacts", "irsim_benchmark_report.json")
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    print(json.dumps(run_benchmark_reviewer(report), indent=2,
                     ensure_ascii=False))
