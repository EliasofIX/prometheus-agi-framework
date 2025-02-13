# interaction_multi_agent/multi_agent_manager.py
class MultiAgentManager:
    """
    Manages multiple agents for social interaction.
    """
    def __init__(self):
        self.agents = {}

    def register_agent(self, agent_id, beliefs=None, goals=None):
        self.agents[agent_id] = {"beliefs": beliefs or {}, "goals": goals or []}

    def update_belief(self, agent_id, key, value):
        if agent_id in self.agents:
            self.agents[agent_id]["beliefs"][key] = value

    def get_agent_info(self, agent_id):
        return self.agents.get(agent_id, {})

