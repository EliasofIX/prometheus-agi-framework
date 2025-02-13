# interaction/theory_of_mind_module.py
class TheoryOfMindModule:
    """
    Models the beliefs and intentions of other agents.
    """
    def __init__(self, multi_agent_manager):
        self.mgr = multi_agent_manager

    def update_agent_beliefs(self, agent_id, new_beliefs):
        if agent_id in self.mgr.agents:
            self.mgr.agents[agent_id]["beliefs"].update(new_beliefs)

    def predict_intention(self, agent_id):
        if agent_id not in self.mgr.agents:
            return "unknown"
        beliefs = self.mgr.agents[agent_id]["beliefs"]
        return "cooperative" if beliefs.get("ally", False) else "neutral"

