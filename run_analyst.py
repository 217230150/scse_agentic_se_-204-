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
run_analyst.py

Runner for the Analyst Agent.  Its responsibilities:
    1. read the brief text (brief.txt),
    2. call run_analyst(brief_text),
    3. receive the final validated software requirements,
    4. save the requirements as requirements.json inside the artifacts/
       folder (json.dump).

Usage:
    python run_analyst.py
"""

import json
import os

from analyst_agent import run_analyst

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF_PATH = os.path.join(HERE, "brief.txt")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
OUT_PATH = os.path.join(ARTIFACTS_DIR, "requirements.json")


def main():
    with open(BRIEF_PATH, "r", encoding="utf-8") as f:
        brief_text = f.read()

    requirements = run_analyst(brief_text)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(requirements, f, indent=2, ensure_ascii=False)

    print("Validated requirements:")
    print(json.dumps(requirements, indent=2, ensure_ascii=False))
    print()
    print("Saved to: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
