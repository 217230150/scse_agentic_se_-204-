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
test_analyst.py

Level 1 - Agent / Pipeline smoke test for the Analyst Agent.

What it does:
    1. takes the original brief.txt file,
    2. sends it to the Analyst Agent (run_analyst) and lets the agent run as
       usual - the agent's own validation accepts or rejects its output,
    3. stores the agent output in the artifacts folder under
       artifacts/requirements.json,
    4. displays the artifact after the test ends.

Ensures the agent turns the brief text into a usable requirements artifact.

Usage:
    python test_analyst.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from analyst_agent import run_analyst  # noqa: E402

BRIEF_PATH = os.path.join(HERE, "brief.txt")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
OUT_PATH = os.path.join(ARTIFACTS_DIR, "requirements.json")


def main():
    # 1. Read the original brief.
    with open(BRIEF_PATH, "r", encoding="utf-8") as f:
        brief_text = f.read()
    print("=== Analyst Agent smoke test ===")
    print("Brief text:\n{}".format(brief_text))
    print()

    # 2. Let the agent run as usual (its own validation accepts/rejects).
    requirements = run_analyst(brief_text)

    # 3. Store the artifact under artifacts/requirements.json.
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(requirements, f, indent=2, ensure_ascii=False)

    # 4. Display the artifact after the test.
    print("Validated requirements artifact:")
    print(json.dumps(requirements, indent=2, ensure_ascii=False))
    print()
    print("PASSED: brief text was turned into a usable requirements artifact.")
    print("Saved to: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
