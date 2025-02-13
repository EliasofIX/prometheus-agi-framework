# knowledge/knowledge_graph.py
import networkx as nx
import time

class KnowledgeGraph:
    """
    A dynamic knowledge graph for storing concepts and relationships.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_concept(self, concept_name, properties=None):
        if concept_name not in self.graph:
            self.graph.add_node(concept_name, props=properties or {}, created=time.time())

    def link_concepts(self, source, target, relation_type="association", confidence=1.0):
        if source not in self.graph:
            self.add_concept(source)
        if target not in self.graph:
            self.add_concept(target)
        self.graph.add_edge(source, target, relation=relation_type, confidence=confidence, updated=time.time())

    def update_concept_properties(self, concept_name, new_props):
        if concept_name in self.graph:
            props = self.graph.nodes[concept_name].get("props", {})
            props.update(new_props)
            self.graph.nodes[concept_name]["props"] = props

    def trace_causal_path(self, concept_name):
        """
        Returns a list of causal edges leading to the given concept.
        """
        visited = set()
        stack = [concept_name]
        path_edges = []
        while stack:
            current = stack.pop()
            visited.add(current)
            for pred in self.graph.predecessors(current):
                edge_data = self.graph[pred][current]
                if edge_data.get("relation") == "causal":
                    path_edges.append((pred, current))
                    if pred not in visited:
                        stack.append(pred)
        return path_edges
# knowledge/knowledge_graph.py
import networkx as nx
import time

class KnowledgeGraph:
        """
            A dynamic knowledge graph for storing concepts and relationships.
            """
        def __init__(self):
                self.graph = nx.DiGraph()

            def add_concept(self, concept_name, properties=None):
                if concept_name not in self.graph:
                        self.graph.add_node(concept_name, props=properties or {}, created=time.time())

                def link_concepts(self, source, target, relation_type="association", confidence=1.0):
                    if source not in self.graph:
                            self.add_concept(source)
                        if target not in self.graph:
                            self.add_concept(target)
                        self.graph.add_edge(source, target, relation=relation_type, confidence=confidence, updated=time.time())

                def update_concept_properties(self, concept_name, new_props):
                    if concept_name in self.graph:
                            props = self.graph.nodes[concept_name].get("props", {})
                            props.update(new_props)
                            self.graph.nodes[concept_name]["props"] = props

                    def trace_causal_path(self, concept_name):
                        """
                            Returns a list of causal edges leading to the given concept.
                            """
                        visited = set()
                        stack = [concept_name]
                        path_edges = []
                        while stack:
                                current = stack.pop()
                                visited.add(current)
                                for pred in self.graph.predecessors(current):
                                        edge_data = self.graph[pred][current]
                                        if edge_data.get("relation") == "causal":
                                                path_edges.append((pred, current))
                                                if pred not in visited:
                                                        stack.append(pred)
                                        return path_edges
# knowledge/knowledge_graph.py
import networkx as nx
import time

class KnowledgeGraph:
        """
            A dynamic knowledge graph for storing concepts and relationships.
            """
        def __init__(self):
                self.graph = nx.DiGraph()

            def add_concept(self, concept_name, properties=None):
                if concept_name not in self.graph:
                        self.graph.add_node(concept_name, props=properties or {}, created=time.time())

                def link_concepts(self, source, target, relation_type="association", confidence=1.0):
                    if source not in self.graph:
                            self.add_concept(source)
                        if target not in self.graph:
                            self.add_concept(target)
                        self.graph.add_edge(source, target, relation=relation_type, confidence=confidence, updated=time.time())

                def update_concept_properties(self, concept_name, new_props):
                    if concept_name in self.graph:
                            props = self.graph.nodes[concept_name].get("props", {})
                            props.update(new_props)
                            self.graph.nodes[concept_name]["props"] = props

                    def trace_causal_path(self, concept_name):
                        """
                            Returns a list of causal edges leading to the given concept.
                            """
                        visited = set()
                        stack = [concept_name]
                        path_edges = []
                        while stack:
                                current = stack.pop()
                                visited.add(current)
                                for pred in self.graph.predecessors(current):
                                        edge_data = self.graph[pred][current]
                                        if edge_data.get("relation") == "causal":
                                                path_edges.append((pred, current))
                                                if pred not in visited:
                                                        stack.append(pred)
                                        return path_edges

