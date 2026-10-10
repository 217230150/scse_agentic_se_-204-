"""
main.py

The Orchestrator - central entry point of the integrated agent pipeline.

It runs every agent in sequence, stores their artifacts, and records every
action in trace.log:

    1. Load the brief (brief.txt)
    2. Run the Analyst agent        -> artifacts/requirements.json
    3. Run the Planner agent        -> artifacts/plan.json
    4. Run the Developer agent      -> generated/navigation_logic.py
    5. Run the Test agent + pytest  -> generated/test_navigation_logic.py
    6. Run the simulation (runner)  -> simulation_report.json
    7. Run the Reviewer agent       -> artifacts/simulation_review.json
    8. Run the IR-SIM benchmark     -> artifacts/irsim_benchmark_report.json
    9. Run the benchmark reviewer   -> artifacts/benchmark_review.json
    10. If a review requests a revision, run run_developer_revision() and
        re-run the whole test/simulation/benchmark pipeline on the revised
        logic (artifacts/second_pass/).

Usage (with the project virtual environment active):
    python main.py
"""

import json
import logging
import os
import subprocess
import sys

from analyst_agent import run_analyst
from planner_agent import run_planner
from developer_agent import run_developer, run_developer_revision
from test_agent import run_test_agent
from runner import run_simulation
from reviewer_agent import run_reviewer
from benchmark_irsim import run_benchmark
from reviewer_benchmark_agent import run_benchmark_reviewer

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
GENERATED_DIR = os.path.join(HERE, "generated")
SECOND_PASS_DIR = os.path.join(ARTIFACTS_DIR, "second_pass")

BRIEF_PATH = os.path.join(HERE, "brief.txt")
REQUIREMENTS_PATH = os.path.join(ARTIFACTS_DIR, "requirements.json")
PLAN_PATH = os.path.join(ARTIFACTS_DIR, "plan.json")
NAVIGATION_PATH = os.path.join(GENERATED_DIR, "navigation_logic.py")
ORIGINAL_NAVIGATION_PATH = os.path.join(GENERATED_DIR, "navigation_logic_original.py")
REVISED_NAVIGATION_PATH = os.path.join(GENERATED_DIR, "navigation_logic_revised.py")
TEST_FILE_PATH = os.path.join(GENERATED_DIR, "test_navigation_logic.py")
SIM_REPORT_PATH = os.path.join(HERE, "simulation_report.json")
SIM_REVIEW_PATH = os.path.join(ARTIFACTS_DIR, "simulation_review.json")
BENCHMARK_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "irsim_benchmark_report.json")
BENCHMARK_REVIEW_PATH = os.path.join(ARTIFACTS_DIR, "benchmark_review.json")

LOG_PATH = os.path.join(HERE, "trace.log")


def setup_logging():
    """Configure logging to trace.log and to the console."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def save_json(path, data):
    """Save data as JSON, creating the parent directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(path, text):
    """Save text, creating the parent directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def load_brief(logger):
    """Step 1: load the brief."""
    logger.info("Loading the brief from %s", os.path.relpath(BRIEF_PATH, HERE))
    with open(BRIEF_PATH, "r", encoding="utf-8") as f:
        brief_text = f.read()
    logger.info("Brief loaded (%d characters)", len(brief_text))
    return brief_text


def run_analyst_stage(brief_text, logger):
    """Step 2: run the Analyst agent. When no DashScope key is configured the
    already-validated artifacts/requirements.json is reused."""
    if os.environ.get("DASHSCOPE_API_KEY"):
        logger.info("Running Analyst Agent")
        requirements = run_analyst(brief_text)
        save_json(REQUIREMENTS_PATH, requirements)
        logger.info("Analyst Agent created requirements.json")
        return requirements

    logger.warning(
        "DASHSCOPE_API_KEY not set; cannot call the cloud Analyst Agent. "
        "Reusing the previously validated requirements artifact."
    )
    fallback = REQUIREMENTS_PATH
    if not os.path.exists(fallback):
        fallback = os.path.join(HERE, "requirements.json")
    with open(fallback, "r", encoding="utf-8") as f:
        requirements = json.load(f)
    save_json(REQUIREMENTS_PATH, requirements)
    logger.info("Analyst requirements loaded from %s",
                os.path.relpath(fallback, HERE))
    return requirements


def run_planner_stage(requirements, logger):
    """Step 3: run the Planner agent."""
    logger.info("Running Planner Agent")
    plan_part = run_planner(requirements)

    # Self-contained artifact, exactly like run_planner.py builds it.
    plan_artifact = dict(requirements)
    plan_artifact.update(plan_part)

    save_json(PLAN_PATH, plan_artifact)
    logger.info("Planner Agent created plan.json")
    return plan_artifact


def run_developer_stage(plan, logger):
    """Step 4: run the Developer agent."""
    logger.info("Running Developer Agent")
    code = run_developer(plan)
    save_text(NAVIGATION_PATH, code)
    logger.info("Developer Agent created generated/navigation_logic.py")
    return code


def run_pytest_stage(requirements, plan, code, logger):
    """Step 5: run the Test agent, then execute the generated pytest module.

    If pytest fails, the failure is fed back to the Test Agent and the test
    module is regenerated once - the navigation code is verified separately
    by the exhaustive simulation, so a failure here is treated as a defect
    in the generated tests first."""
    feedback = ""
    test_code = None

    for generation_round in range(2):
        logger.info("Running Test Agent (generation %d)", generation_round + 1)
        test_code = run_test_agent(requirements, plan, code, feedback)
        save_text(TEST_FILE_PATH, test_code)
        logger.info(
            "Test Agent created generated/test_navigation_logic.py"
        )

        logger.info("Running pytest on generated/test_navigation_logic.py")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest",
             os.path.relpath(TEST_FILE_PATH, HERE), "-v", "--tb=short"],
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = completed.stdout or ""
        failed_tests = []
        for line in output.splitlines():
            if line.startswith(("PASSED", "FAILED", "ERROR", "test_", "=")):
                logger.info("pytest | %s", line)
            if line.strip().startswith("FAILED"):
                failed_tests.append(line.strip())

        if completed.returncode == 0:
            logger.info("Pytest passed")
            return True, test_code

        logger.error("Pytest failed (return code %d)", completed.returncode)
        feedback = (
            "The test module you generated failed in these test(s): "
            + "; ".join(failed_tests)
            + ". The navigation contract is fixed: goal ahead + clear front "
            "=> FORWARD; goal left + clear left => LEFT; goal right + clear "
            "right => RIGHT; goal direction blocked => use a clear side; "
            "all directions blocked => STOP. Correct the EXPECTATIONS and "
            "state dictionaries in the test code and output the full pytest "
            "module again."
        )

    return False, test_code


def run_simulation_stage(logger, output_dir):
    """Step 6: run the navigation simulation and save its report."""
    logger.info("Running navigation simulation (runner.py logic)")
    report = run_simulation()

    report_path = os.path.join(output_dir, "simulation_report.json")
    save_json(report_path, report)
    # Keep the exact root-level file name requested in the brief.
    save_json(SIM_REPORT_PATH, report)

    logger.info(
        "Simulation finished: %d/%d scenarios passed (success rate %.2f%%)",
        report["passed_scenarios"], report["total_scenarios"],
        report["success_rate"] * 100,
    )
    return report


def run_reviewer_stage(simulation_report, logger, output_dir):
    """Step 7: run the simulation Reviewer agent."""
    logger.info("Running Reviewer Agent on the simulation report")
    review = run_reviewer(simulation_report)

    review_path = os.path.join(output_dir, "simulation_review.json")
    save_json(review_path, review)
    save_json(SIM_REVIEW_PATH, review)

    logger.info("Reviewer Agent produced simulation_review.json "
                "(needs_revision=%s)", review["needs_revision"])
    return review


def run_benchmark_stage(logger, output_dir):
    """Step 8: run the IR-SIM benchmark across all worlds."""
    logger.info("Running IR-SIM benchmark for every world")
    report = run_benchmark()

    report_path = os.path.join(output_dir, "irsim_benchmark_report.json")
    save_json(report_path, report)
    save_json(BENCHMARK_REPORT_PATH, report)

    logger.info(
        "IR-SIM benchmark finished: success rate %.2f%%, efficiency rate "
        "%.2f%%, average steps %.1f",
        report["success_rate"] * 100, report["efficiency_rate"] * 100,
        report["average_steps_taken"],
    )
    return report


def run_benchmark_reviewer_stage(benchmark_report, logger, output_dir):
    """Step 9: run the benchmark Reviewer agent."""
    logger.info("Running Benchmark Reviewer Agent")
    review = run_benchmark_reviewer(benchmark_report)

    review_path = os.path.join(output_dir, "benchmark_review.json")
    save_json(review_path, review)
    save_json(BENCHMARK_REVIEW_PATH, review)

    logger.info(
        "Benchmark reviewer produced benchmark_review.json "
        "(safety=%s, goals=%s, efficiency=%s, needs_revision=%s)",
        review["safety_passed"], review["goal_completion_passed"],
        review["efficiency_passed"], review["needs_revision"],
    )
    return review


def run_revision_stage(original_code, simulation_review, benchmark_report,
                       benchmark_review, requirements, plan, logger):
    """Step 10: generate the revised navigation logic."""
    logger.info("A review requested changes; running Developer revision")

    benchmark_context = dict(benchmark_report)
    benchmark_context["benchmark_review"] = benchmark_review

    revised = run_developer_revision(
        original_code=original_code,
        review_notes=simulation_review,
        benchmark_report=benchmark_context,
        requirements=requirements,
        plan=plan,
        out_path=REVISED_NAVIGATION_PATH,
    )
    logger.info(
        "Developer revision created generated/navigation_logic_revised.py"
    )
    return revised


def run_post_revision_pipeline(revised_code, requirements, plan, logger):
    """Re-test the revised logic and re-run the full test/simulation/
    benchmark pipeline. Outputs of this second pass live under
    artifacts/second_pass/."""
    os.makedirs(SECOND_PASS_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("Starting second pipeline pass with the revised logic")

    # Preserve the v1 module, then install the revised logic where the
    # unchanged harness imports it (generated/navigation_logic.py).
    with open(NAVIGATION_PATH, "r", encoding="utf-8") as f:
        v1_code = f.read()
    save_text(ORIGINAL_NAVIGATION_PATH, v1_code)
    save_text(NAVIGATION_PATH, revised_code)
    logger.info("Original v1 logic preserved as "
                "generated/navigation_logic_original.py")

    pytest_passed, _ = run_pytest_stage(
        requirements, plan, revised_code, logger
    )
    sim_report = run_simulation_stage(logger, SECOND_PASS_DIR)
    sim_review = run_reviewer_stage(sim_report, logger, SECOND_PASS_DIR)
    benchmark_report = run_benchmark_stage(logger, SECOND_PASS_DIR)
    benchmark_review = run_benchmark_reviewer_stage(
        benchmark_report, logger, SECOND_PASS_DIR
    )

    logger.info(
        "Second pass complete: pytest=%s, simulation success %.2f%%, "
        "benchmark success %.2f%%",
        pytest_passed, sim_report["success_rate"] * 100,
        benchmark_report["success_rate"] * 100,
    )
    return {
        "pytest_passed": pytest_passed,
        "simulation_report": sim_report,
        "simulation_review": sim_review,
        "benchmark_report": benchmark_report,
        "benchmark_review": benchmark_review,
    }


def main():
    logger = setup_logging()
    logger.info("Orchestrator started")

    # ---- First pass -----------------------------------------------------
    brief_text = load_brief(logger)
    requirements = run_analyst_stage(brief_text, logger)
    plan = run_planner_stage(requirements, logger)
    original_code = run_developer_stage(plan, logger)
    pytest_passed, _ = run_pytest_stage(
        requirements, plan, original_code, logger
    )
    simulation_report = run_simulation_stage(logger, ARTIFACTS_DIR)
    simulation_review = run_reviewer_stage(
        simulation_report, logger, ARTIFACTS_DIR
    )
    benchmark_report = run_benchmark_stage(logger, ARTIFACTS_DIR)
    benchmark_review = run_benchmark_reviewer_stage(
        benchmark_report, logger, ARTIFACTS_DIR
    )

    # ---- Revision and second pass --------------------------------------
    force_revision = os.environ.get("FORCE_REVISION", "").lower() in (
        "1",
        "true",
        "yes",
    )
    needs_revision = (
        simulation_review.get("needs_revision", False)
        or benchmark_review.get("needs_revision", False)
        or force_revision
    )

    second_pass = None
    if needs_revision:
        if force_revision and not (
            simulation_review.get("needs_revision", False)
            or benchmark_review.get("needs_revision", False)
        ):
            logger.info(
                "Revision stage forced via FORCE_REVISION although both "
                "reviews passed (used to demonstrate the revision loop)"
            )

        revised_code = run_revision_stage(
            original_code, simulation_review, benchmark_report,
            benchmark_review, requirements, plan, logger
        )
        second_pass = run_post_revision_pipeline(
            revised_code, requirements, plan, logger
        )

        # Keep the better of the two logics as the final navigation module.
        # Logical simulation success is weighted alongside benchmark success
        # and efficiency.
        first_score = (
            benchmark_report["success_rate"] * 2
            + benchmark_report["efficiency_rate"]
            + simulation_report["success_rate"]
        )
        second_score = (
            second_pass["benchmark_report"]["success_rate"] * 2
            + second_pass["benchmark_report"]["efficiency_rate"]
            + second_pass["simulation_report"]["success_rate"]
        )
        if second_score >= first_score:
            logger.info(
                "Revised logic kept as generated/navigation_logic.py "
                "(score %.2f vs %.2f)", second_score, first_score
            )
        else:
            save_text(NAVIGATION_PATH, original_code)
            logger.warning(
                "Revision did not improve the benchmark (score %.2f vs "
                "%.2f); restored the original logic. The revised file is "
                "still available at generated/navigation_logic_revised.py",
                second_score, first_score,
            )
    else:
        logger.info("No revision requested by either reviewer")

    logger.info("Orchestrator finished. Trace saved to %s",
                os.path.relpath(LOG_PATH, HERE))


if __name__ == "__main__":
    main()
