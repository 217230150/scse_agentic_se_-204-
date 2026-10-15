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
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generated.navigation_logic import decide_next_move

def test_goal_ahead_front_clear():
    state = {
        "goal_ahead": True,
        "goal_on_left": False,
        "goal_on_right": False,
        "front_blocked": False,
        "left_blocked": False,
        "right_blocked": False,
    }
    assert decide_next_move(state) == "FORWARD"

def test_goal_on_left_left_clear():
    state = {
        "goal_ahead": False,
        "goal_on_left": True,
        "goal_on_right": False,
        "front_blocked": False,
        "left_blocked": False,
        "right_blocked": False,
    }
    assert decide_next_move(state) == "LEFT"

def test_goal_on_right_right_clear():
    state = {
        "goal_ahead": False,
        "goal_on_left": False,
        "goal_on_right": True,
        "front_blocked": False,
        "left_blocked": False,
        "right_blocked": False,
    }
    assert decide_next_move(state) == "RIGHT"

def test_goal_ahead_front_blocked_left_clear():
    state = {
        "goal_ahead": True,
        "goal_on_left": False,
        "goal_on_right": False,
        "front_blocked": True,
        "left_blocked": False,
        "right_blocked": False,
    }
    assert decide_next_move(state) == "LEFT"

def test_goal_ahead_front_and_left_blocked_right_clear():
    state = {
        "goal_ahead": True,
        "goal_on_left": False,
        "goal_on_right": False,
        "front_blocked": True,
        "left_blocked": True,
        "right_blocked": False,
    }
    assert decide_next_move(state) == "RIGHT"

def test_goal_on_left_left_blocked_front_clear():
    state = {
        "goal_ahead": False,
        "goal_on_left": True,
        "goal_on_right": False,
        "front_blocked": False,
        "left_blocked": True,
        "right_blocked": False,
    }
    assert decide_next_move(state) == "FORWARD"

def test_goal_on_right_right_blocked_front_clear():
    state = {
        "goal_ahead": False,
        "goal_on_left": False,
        "goal_on_right": True,
        "front_blocked": False,
        "left_blocked": False,
        "right_blocked": True,
    }
    assert decide_next_move(state) == "FORWARD"

def test_front_blocked_left_and_right_clear():
    state = {
        "goal_ahead": False,
        "goal_on_left": False,
        "goal_on_right": False,
        "front_blocked": True,
        "left_blocked": False,
        "right_blocked": False,
    }
    assert decide_next_move(state) == "LEFT"

def test_every_direction_blocked():
    state = {
        "goal_ahead": False,
        "goal_on_left": False,
        "goal_on_right": False,
        "front_blocked": True,
        "left_blocked": True,
        "right_blocked": True,
    }
    assert decide_next_move(state) == "STOP"

def test_no_goal_flag_set_front_clear():
    state = {
        "goal_ahead": False,
        "goal_on_left": False,
        "goal_on_right": False,
        "front_blocked": False,
        "left_blocked": False,
        "right_blocked": False,
    }
    assert decide_next_move(state) == "FORWARD"

def test_every_returned_value_is_one_of_the_allowed_actions():
    assert decide_next_move({"goal_ahead": False, "goal_on_left": False, "goal_on_right": False, "front_blocked": False, "left_blocked": False, "right_blocked": False}) in ["FORWARD", "LEFT", "RIGHT", "STOP"]
