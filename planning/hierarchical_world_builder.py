# planning/hierarchical_world_builder.py
class HierarchicalWorldBuilder:
    """
    Constructs multi-scale representations of the environment.
    """
    def __init__(self):
        self.levels = {}

    def add_level(self, level_name, representation):
        self.levels[level_name] = representation

    def refine_level(self, level_name, new_data):
        print(f"HierarchicalWorldBuilder: Refining level {level_name} (stub).")
# planning/hierarchical_world_builder.py
class HierarchicalWorldBuilder:
        """
            Constructs multi-scale representations of the environment.
            """
        def __init__(self):
                self.levels = {}

            def add_level(self, level_name, representation):
                self.levels[level_name] = representation

            def refine_level(self, level_name, new_data):
                print(f"HierarchicalWorldBuilder: Refining level {level_name} (stub).")

