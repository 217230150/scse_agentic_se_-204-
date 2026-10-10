def decide_next_move(state):
    goal_ahead = state["goal_ahead"]
    goal_on_left = state["goal_on_left"]
    goal_on_right = state["goal_on_right"]
    front_blocked = state["front_blocked"]
    left_blocked = state["left_blocked"]
    right_blocked = state["right_blocked"]
    
    # Prefer moving toward the goal whenever it is safe.
    if goal_ahead and not front_blocked:
        return "FORWARD"
    if goal_on_left and not left_blocked:
        return "LEFT"
    if goal_on_right and not right_blocked:
        return "RIGHT"
    
    # Otherwise avoid obstacles and keep moving (original logic).
    if not front_blocked:
        return "FORWARD"
    if not left_blocked:
        return "LEFT"
    if not right_blocked:
        return "RIGHT"
    
    # No safe direction: stop.
    return "STOP"
