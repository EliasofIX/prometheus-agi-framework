# cognition/goal_autonomy_engine.py
class GoalAutonomyEngine:
    """
    Generates self-directed, intrinsic goals.
    """
    def __init__(self):
        self.goal_history = []

    def propose_new_goal(self, novelty_score):
        if novelty_score > 0.5:
            new_goal = f"Explore phenomenon_{novelty_score:.2f}"
            self.goal_history.append(new_goal)
            return new_goal
        return None

