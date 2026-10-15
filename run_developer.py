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
run_developer.py

Runner for the Developer Agent.  Its responsibilities:
    1. read the validated navigation plan from plan.json (the artifact
       produced by the Planner Agent; preferred location is artifacts/,
       falling back to the project root),
    2. call run_developer(plan),
    3. receive the validated Python code,
    4. store the generated code in generated/navigation_logic.py.

Usage:
    python run_developer.py
"""

import json
import os

from developer_agent import run_developer

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_PATH = os.path.join(HERE, "artifacts", "plan.json")
if not os.path.exists(PLAN_PATH):
    PLAN_PATH = os.path.join(HERE, "plan.json")
GENERATED_DIR = os.path.join(HERE, "generated")
OUT_PATH = os.path.join(GENERATED_DIR, "navigation_logic.py")


def main():
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)

    print("Validated plan (input to the Developer Agent):")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print()

    code = run_developer(plan)

    os.makedirs(GENERATED_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(code)
        if not code.endswith("\n"):
            f.write("\n")

    print("Validated navigation code:")
    print(code)
    print()
    print("Saved to: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
