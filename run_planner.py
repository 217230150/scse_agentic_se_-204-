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
run_planner.py

Runner for the Planner Agent.  Its responsibilities:
    1. read the validated requirements from requirements.json (the artifact
       produced by the Analyst Agent at the end of the previous project),
    2. call run_planner(requirements),
    3. receive the final validated navigation plan,
    4. save the plan as plan.json (json.dump) - the artifact that the
       Developer Agent will use.

The saved plan.json combines the validated requirements (goal,
allowed_actions, safe_stop, avoid.obstacles) with the navigation plan
(strategy, decisions, stop_condition), so it is a self-contained artifact for
the Developer Agent.

Usage:
    python run_planner.py
"""

import json
import os

from planner_agent import run_planner

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_PATH = os.path.join(HERE, "requirements.json")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
OUT_PATH = os.path.join(ARTIFACTS_DIR, "plan.json")


def main():
    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
        requirements = json.load(f)

    print("Validated requirements (input to the Planner Agent):")
    print(json.dumps(requirements, indent=2, ensure_ascii=False))
    print()

    plan = run_planner(requirements)

    # Self-contained artifact: requirements context + navigation plan.
    plan_artifact = dict(requirements)
    plan_artifact.update(plan)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(plan_artifact, f, indent=2, ensure_ascii=False)

    print("Validated plan:")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print()
    print("Saved to: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
