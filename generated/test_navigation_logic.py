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

def test_goal_ahead_and_every_direction_blocked():
    state = {
        "goal_ahead": True,
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
    assert decide_next_move({"goal_ahead": True, "goal_on_left": False, "goal_on_right": False, "front_blocked": False, "left_blocked": False, "right_blocked": False}) in ["FORWARD", "LEFT", "RIGHT", "STOP"]
    assert decide_next_move({"goal_ahead": False, "goal_on_left": True, "goal_on_right": False, "front_blocked": False, "left_blocked": False, "right_blocked": False}) in ["LEFT", "FORWARD", "RIGHT", "STOP"]
    assert decide_next_move({"goal_ahead": False, "goal_on_left": False, "goal_on_right": True, "front_blocked": False, "left_blocked": False, "right_blocked": False}) in ["RIGHT", "FORWARD", "LEFT", "STOP"]
    assert decide_next_move({"goal_ahead": False, "goal_on_left": False, "goal_on_right": False, "front_blocked": False, "left_blocked": False, "right_blocked": False}) in ["FORWARD", "LEFT", "RIGHT", "STOP"]
