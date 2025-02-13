# planning/universal_planner.py
class UniversalPlanner:
    """
    A hybrid planner combining symbolic reasoning with RL-based refinement.
    """
    def __init__(self, knowledge_graph, symbolic_engine=None, rl_policy=None):
        self.kg = knowledge_graph
        self.symbolic_engine = symbolic_engine
        self.rl_policy = rl_policy

    def plan(self, goal):
        skeleton = self._symbolic_plan(goal)
        final_plan = self._rl_refine(skeleton)
        return final_plan

    def _symbolic_plan(self, goal):
        if self.symbolic_engine:
            self.symbolic_engine.verify_logic_consistency()
            return [f"symbolic_step_for_{goal}"]
        return [f"naive_plan_for_{goal}"]

    def _rl_refine(self, skeleton):
        if not self.rl_policy:
            return skeleton
        return [step + "_RLRefined" for step in skeleton]

