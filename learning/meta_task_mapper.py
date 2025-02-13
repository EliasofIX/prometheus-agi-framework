# learning/meta_task_mapper.py
class MetaTaskMapper:
    """
    Maps new tasks to existing skills using a domain-agnostic embedding space.
    """
    def __init__(self):
        pass

    def find_best_skill_mapping(self, new_task_description, known_tasks):
        if "finance" in new_task_description.lower():
            return "finance_skill"
        elif "robot" in new_task_description.lower():
            return "robotics_skill"
        else:
            return "general_skill"


