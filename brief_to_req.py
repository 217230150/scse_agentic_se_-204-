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
brief_to_req.py

Part 1 of the Requirements Engineering exercise:
read the human-language brief (brief.txt), send it to the Qwen model and
have the model produce software requirements from the text.  The model
output is saved to robot_requirements.txt.

Usage:
    python brief_to_req.py                # run once, save to robot_requirements.txt
    python brief_to_req.py --times 3      # run 3 times and examine consistency

Requires an environment variable DASHSCOPE_API_KEY (see README.md).
Uses only the Python standard library (no third-party packages).
"""

import argparse
import json
import os
import urllib.error
import urllib.request

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF_PATH = os.path.join(HERE, "brief.txt")
OUT_PATH = os.path.join(HERE, "robot_requirements.txt")

SYSTEM_PROMPT = (
    "You are a requirements engineer. Read the human-language brief below and "
    "produce a clear set of explicit software requirements for the system. "
    "List the requirements as short, numbered statements. Do not invent "
    "features that are not supported by the brief."
)


def call_qwen(brief_text):
    """Send the brief text to Qwen and return the model's text answer."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Turn brief.txt into software requirements using Qwen."
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        help="How many times to run the model (to check consistency).",
    )
    args = parser.parse_args()
    if args.times < 1:
        parser.error("--times must be at least 1")

    with open(BRIEF_PATH, "r", encoding="utf-8") as f:
        brief_text = f.read()

    outputs = []
    for run_no in range(1, args.times + 1):
        print("--- Run {}/{} ---".format(run_no, args.times))
        text = call_qwen(brief_text)
        print(text)
        print()
        outputs.append(text)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("Qwen output for brief.txt\n")
        f.write("=" * 60 + "\n\n")
        for i, text in enumerate(outputs, start=1):
            f.write("--- Run {} ---\n{}\n\n".format(i, text))

    print("Model output saved to: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
