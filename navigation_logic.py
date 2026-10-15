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
# AI involvement: none. This file is course-provided original code (the
# reference navigation implementation / course test scaffold) with no AI
# generation or AI modification.
# ============================================================================
def decide_next_action(blocked_ahead, blocked_left, blocked_right, goal_reached):
    if goal_reached:
        return "STOP"
    if not blocked_ahead:
        return "FORWARD"
    if not blocked_left:
        return "LEFT"
    if not blocked_right:
        return "RIGHT"
    return "STOP"
