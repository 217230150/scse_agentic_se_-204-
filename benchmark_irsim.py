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
benchmark_irsim.py

Automated benchmarking for the IR-SIM navigation runs.

For every world file in irsim_worlds it runs test_irsim_with_agent.py as a
subprocess (the same script used by run_visual.sh, but with a non-interactive
matplotlib backend so the benchmark does not block on a window), extracts the
per-world report from the script output, and aggregates the metrics required
by the project brief:

    Success Rate  = goals reached / total environments
    Efficiency    = efficient runs / total environments
    Average steps = total steps / total environments

The aggregated report is saved as
    artifacts/irsim_benchmark_report.json

Usage:
    python benchmark_irsim.py
"""

import ast
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORLDS_DIR = os.path.join(HERE, "irsim_worlds")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
PER_WORLD_REPORT = os.path.join(ARTIFACTS_DIR, "irsim_report.json")
HARNESS = os.path.join(HERE, "test_irsim_with_agent.py")

# The brief lists easy, medium, difficult and custom. The cloned repository
# names the difficult world "world_hard.yaml"; keep the measured order
# explicit so the benchmark is stable regardless of file-system ordering.
WORLD_ORDER = [
    ("easy", "world_easy.yaml"),
    ("medium", "world_medium.yaml"),
    ("hard", "world_hard.yaml"),       # "difficult" in the slide deck
    ("custom", "world_custom.yaml"),
]


def load_world_files():
    """Return (name, absolute path) pairs for all available world files in
    the canonical order. Missing files are skipped; any extra yaml file is
    appended at the end so nothing is silently left untested."""
    worlds = []
    seen = set()

    for name, filename in WORLD_ORDER:
        path = os.path.join(WORLDS_DIR, filename)
        if os.path.exists(path):
            worlds.append((name, path))
            seen.add(filename)

    if os.path.isdir(WORLDS_DIR):
        for filename in sorted(os.listdir(WORLDS_DIR)):
            if filename.endswith((".yaml", ".yml")) and filename not in seen:
                worlds.append((os.path.splitext(filename)[0],
                               os.path.join(WORLDS_DIR, filename)))

    if not worlds:
        raise FileNotFoundError(
            "No IR-SIM world files found in {}.".format(WORLDS_DIR)
        )
    return worlds


def _extract_report_from_output(stdout):
    """Parse the per-world report dict printed by test_irsim_with_agent.py.

    The script prints the report with print(report) (Python dict syntax), so
    the line is parsed with ast.literal_eval. The first dict that contains
    the 'reached_goal' key wins."""
    for line in stdout.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            candidate = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(candidate, dict) and "reached_goal" in candidate:
            return candidate
    return None


def run_world_benchmark(name, world_path):
    """Run test_irsim_with_agent.py for one world file and return the report
    dictionary augmented with the world name and path.

    The matplotlib backend is forced to Agg and IR-SIM is put into headless
    mode in the subprocess so no window opens and plt.show() returns
    immediately instead of blocking on an interactive window."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["IRSIM_HEADLESS"] = "1"

    completed = subprocess.run(
        [sys.executable, HARNESS, world_path],
        cwd=HERE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = completed.stdout or ""

    report = _extract_report_from_output(output)

    # Fallback: the harness always writes artifacts/irsim_report.json.
    if report is None and os.path.exists(PER_WORLD_REPORT):
        with open(PER_WORLD_REPORT, "r", encoding="utf-8") as f:
            report = json.load(f)

    if report is None:
        raise RuntimeError(
            "Could not extract the IR-SIM report for {} (exit code {}). "
            "Output ends with: {}".format(
                name, completed.returncode, output[-1000:]
            )
        )

    report["world_name"] = name
    report["world_file"] = os.path.relpath(world_path, HERE)
    return report


def run_benchmark():
    """Run the benchmark for every world and build the summary report."""
    worlds = load_world_files()
    results = []

    for name, path in worlds:
        print("Benchmarking world: {} ({})".format(
            name, os.path.relpath(path, HERE)))
        results.append(run_world_benchmark(name, path))

    total = len(results)
    reached_goal_count = sum(1 for r in results if r.get("reached_goal"))
    efficient_count = sum(
        1 for r in results
        if r.get("reached_goal") and r.get("efficiency_status") == "efficient"
    )
    inefficient_success_count = sum(
        1 for r in results
        if r.get("reached_goal")
        and r.get("efficiency_status") == "acceptable_but_inefficient"
    )
    failed_count = sum(1 for r in results if not r.get("reached_goal"))
    total_steps = sum(int(r.get("steps_taken", 0)) for r in results)
    average_steps = total_steps / total if total else 0.0

    benchmark_report = {
        "total_worlds": total,
        "goals_reached": reached_goal_count,
        "success_rate": reached_goal_count / total if total else 0.0,
        "efficient_worlds": efficient_count,
        "inefficient_success_worlds": inefficient_success_count,
        "failed_worlds": failed_count,
        "efficiency_rate": efficient_count / total if total else 0.0,
        "average_steps_taken": average_steps,
        "results": results,
    }
    return benchmark_report


def save_report(benchmark_report):
    """Save the benchmark report under artifacts/ as JSON."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, "irsim_benchmark_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)
    return path


def print_report(benchmark_report):
    """Print the benchmark report in a readable format."""
    print("\n" + "=" * 60)
    print("IR-SIM BENCHMARK SUMMARY")
    print("=" * 60)
    print("Total worlds        : {}".format(benchmark_report["total_worlds"]))
    print("Goals reached       : {}".format(benchmark_report["goals_reached"]))
    print("Success rate        : {:.2%}".format(benchmark_report["success_rate"]))
    print("Efficient worlds    : {}".format(benchmark_report["efficient_worlds"]))
    print("Inefficient success : {}".format(
        benchmark_report["inefficient_success_worlds"]))
    print("Failed worlds       : {}".format(benchmark_report["failed_worlds"]))
    print("Efficiency rate     : {:.2%}".format(
        benchmark_report["efficiency_rate"]))
    print("Average steps       : {:.1f}".format(
        benchmark_report["average_steps_taken"]))
    print("-" * 60)
    for result in benchmark_report["results"]:
        print("  {:<8} reached={!s:<5} steps={:<4} status={}".format(
            result.get("world_name"),
            result.get("reached_goal"),
            result.get("steps_taken"),
            result.get("efficiency_status"),
        ))
    print("=" * 60)


def main():
    benchmark_report = run_benchmark()
    path = save_report(benchmark_report)
    print_report(benchmark_report)
    print("Benchmark report saved to: {}".format(path))


if __name__ == "__main__":
    main()
