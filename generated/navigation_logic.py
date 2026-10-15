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
# AI involvement: 100% of this file's content was generated at runtime by
# a Qwen model through local Ollama (see developer_agent.py / test_agent.py
# for the prompts and calling code; see trace.log for the generation
# process and timestamps). The team only reviewed its structure and ran
# the tests; the logic inside was not manually rewritten.
# ============================================================================
def decide_next_move(state):
    goal_ahead = state["goal_ahead"]
    goal_on_left = state["goal_on_left"]
    goal_on_right = state["goal_on_right"]
    front_blocked = state["front_blocked"]
    left_blocked = state["left_blocked"]
    right_blocked = state["right_blocked"]
    
    # Move toward the goal if it's in a clear direction
    if goal_ahead and not front_blocked:
        return "FORWARD"
    if goal_on_left and not left_blocked:
        return "LEFT"
    if goal_on_right and not right_blocked:
        return "RIGHT"
    
    # If no goal direction is clear, prioritize moving forward if possible
    if not front_blocked:
        return "FORWARD"
    
    # If front is blocked, choose between left and right based on goal position
    if not left_blocked:
        return "LEFT"
    if not right_blocked:
        return "RIGHT"
    
    # All directions are blocked
    return "STOP"
