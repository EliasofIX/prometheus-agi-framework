# interaction/social_cognition.py
class SocialCognition:
    """
    Processes social signals and generates contextually appropriate responses.
    """
    def __init__(self, knowledge_graph, multi_agent_manager):
        self.kg = knowledge_graph
        self.mgr = multi_agent_manager

    def interpret_signal(self, agent_id, signal_type, intensity):
        self.kg.update_concept_properties(agent_id, {signal_type: intensity})

    def generate_social_response(self, agent_id):
        info = self.mgr.get_agent_info(agent_id)
        if info.get("beliefs", {}).get("frustration", 0) > 0.5:
            return "I detect high frustration; providing additional clarification."
        return "Proceeding as planned."

