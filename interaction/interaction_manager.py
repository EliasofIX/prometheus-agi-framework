# interaction/interaction_manager.py
class InteractionManager:
    """
    Orchestrates social interactions and dialogue.
    """
    def __init__(self, tom_module, social_cognition):
        self.tom = tom_module
        self.social = social_cognition

    def process_event(self, agent_id, event):
        self.social.interpret_signal(agent_id, "emotion", 0.7)
        new_beliefs = {"ally": True} if "team" in event else {}
        self.tom.update_agent_beliefs(agent_id, new_beliefs)
        return self.tom.predict_intention(agent_id)

