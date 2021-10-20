def reward_function(rew: dict) -> float:
    return rew["found_cheese"] * 2
