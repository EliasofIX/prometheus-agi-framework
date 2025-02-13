# debug_utils.py
import networkx as nx
import matplotlib.pyplot as plt

class DebugUtils:
    """
    Utility functions for visualizing and debugging internal states.
    """
    @staticmethod
    def visualize_knowledge_graph(kg):
        plt.figure(figsize=(8,6))
        pos = nx.spring_layout(kg.graph)
        nx.draw(kg.graph, pos, with_labels=True, node_color='lightblue', font_size=8)
        edge_labels = {(u, v): d.get("relation", "") for u, v, d in kg.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(kg.graph, pos, edge_labels=edge_labels, font_color="red")
        plt.title("PROMETHEUS 10.0 Knowledge Graph")
        plt.show()

    @staticmethod
    def print_simulation_results(results):
        print("Simulation Results:")
        for res in results:
            print(f" Step {res.get('step')}: next_state: {res.get('next_state')}, info: {res.get('info')}")

    @staticmethod
    def debug_meta_reflection(reflection_text):
        print("=== Meta-Reflection ===")
        print(reflection_text)
        print("=== End Meta-Reflection ===")

