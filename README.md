# scse_agentic_se_-204-

Course: Special Course in Software Engineering '26
Projects: Requirements Engineering → Plan and Develop → Testing → **Integration and Simulation**

## The integrated pipeline

`main.py` is the orchestrator. Running it executes every agent in order and
records all actions in `trace.log`:

1. Load the brief (`brief.txt`)
2. Analyst agent → `artifacts/requirements.json`
3. Planner agent → `artifacts/plan.json`
4. Developer agent → `generated/navigation_logic.py`
5. Test agent → `generated/test_navigation_logic.py`, then runs pytest
6. Simulation runner → `simulation_report.json`
7. Reviewer agent → `artifacts/simulation_review.json`
8. IR-SIM benchmark → `artifacts/irsim_benchmark_report.json`
9. Benchmark reviewer → `artifacts/benchmark_review.json`
10. If a review requests it, `run_developer_revision()` generates revised
    logic (`generated/navigation_logic_revised.py`) and the whole
    test/simulation/benchmark pipeline is re-run; the second-pass artifacts
    are stored under `artifacts/second_pass/`.

## Files

| File | Purpose |
|---|---|
| `brief.txt` | The human-language brief (robot navigation). |
| `analyst_agent.py` / `run_analyst.py` | Analyst agent and its standalone runner. |
| `planner_agent.py` / `run_planner.py` | Planner agent and its standalone runner. |
| `developer_agent.py` | Developer agent; also `run_developer_revision()` for the revised logic (the only pre-existing file modified). |
| `test_agent.py` | Test agent: generates pytest tests for `decide_next_move`. |
| `runner.py` | Logical simulation: named scenarios + all 64 boolean states; writes `simulation_report.json`. |
| `reviewer_agent.py` | Reviews `simulation_report.json`; strict JSON verdict. |
| `test_irsim_with_agent.py` | Provided IR-SIM harness (unchanged); runs one world with the generated logic. |
| `benchmark_irsim.py` | Runs every world headless and aggregates success/efficiency/step metrics. |
| `reviewer_benchmark_agent.py` | Reviews the benchmark report against safety, goal completion and efficiency. |
| `run_visual.sh` | Provided visual runner (unchanged). |
| `irsim_worlds/` | `world_easy.yaml`, `world_medium.yaml`, `world_hard.yaml`, `world_custom.yaml`. |
| `generated/navigation_logic.py` | Navigation code under test (`decide_next_move(state)`). |
| `artifacts/` | All reports and reviews. |

## Setup

```powershell
uv venv --python 3.12 .venv
# if the default PyPI mirror is stale, point uv at a synced mirror:
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv pip install ir-sim ollama pytest
```

A local [Ollama](https://ollama.com) installation must be running with the
models used by the agents (default `qwen2.5:1.5b`; the revision uses
`qwen3:8b`, override with `REVISION_MODEL`).

The Analyst agent uses the DashScope Qwen API. Set `DASHSCOPE_API_KEY` to
re-run it; otherwise the orchestrator reuses the validated
`artifacts/requirements.json`.

## Run

```powershell
# Full integrated pipeline (all agents, logging, benchmark, revision + re-run):
.venv\Scripts\python.exe main.py

# Individual stages:
.venv\Scripts\python.exe runner.py
.venv\Scripts\python.exe benchmark_irsim.py

# Visual IR-SIM runs (one per world; use Git Bash on Windows):
./run_visual.sh irsim_worlds/world_easy.yaml
./run_visual.sh irsim_worlds/world_medium.yaml
./run_visual.sh irsim_worlds/world_hard.yaml
./run_visual.sh irsim_worlds/world_custom.yaml
```

## Notes

- The slide deck names the difficult world `world_difficult.yaml`; the cloned
  repository ships it as `world_hard.yaml`. The benchmark uses the actual
  files in the order easy → medium → hard → custom.
- Actions are restricted to exactly: FORWARD, LEFT, RIGHT, STOP.
