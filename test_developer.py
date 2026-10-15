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
test_developer.py

Level 1 - Agent / Pipeline smoke test for the Developer Agent.

What it does:
    1. takes the artifact artifacts/plan.json,
    2. sends it to the Developer Agent (run_developer) and lets the agent run
       as usual - the agent's own validation accepts or rejects its output,
    3. stores the agent output in the generated folder under
       generated/navigation_logic.py,
    4. displays the code after the test ends.

Ensures the plan artifact is turned into runnable navigation code and that the
decide_next_move(state) entry function - the only function other
programs/agents call - is created.

Usage:
    python test_developer.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from developer_agent import run_developer  # noqa: E402

PLAN_PATH = os.path.join(HERE, "artifacts", "plan.json")
GENERATED_DIR = os.path.join(HERE, "generated")
OUT_PATH = os.path.join(GENERATED_DIR, "navigation_logic.py")


def main():
    # 1. Read the plan artifact.
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)
    print("=== Developer Agent smoke test ===")
    print("Plan artifact:\n{}".format(
        json.dumps(plan, indent=2, ensure_ascii=False)
    ))
    print()

    # 2. Let the agent run as usual (its own validation accepts/rejects).
    code = run_developer(plan)

    # 3. Store the code under generated/navigation_logic.py.
    os.makedirs(GENERATED_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(code)
        if not code.endswith("\n"):
            f.write("\n")

    # 4. Display the code after the test.
    print("Validated navigation code:")
    print(code)
    print()

    # The different logic is fine, but decide_next_move(state) must exist.
    if "def decide_next_move" not in code:
        raise RuntimeError(
            "FAILED: generated code does not define decide_next_move(state)."
        )

    print("PASSED: plan was turned into navigation code with decide_next_move(state).")
    print("Saved to: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
