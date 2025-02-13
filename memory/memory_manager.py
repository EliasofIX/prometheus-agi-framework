# memory/memory_manager.py
from episodic_memory_store import EpisodicMemoryStore
from semantic_knowledge_bank import SemanticKnowledgeBank
from advanced_memory import AdvancedMemory

class MemoryManager:
    """
    Integrates episodic, semantic, and consolidated memory.
    """
    def __init__(self, base_model):
        self.episodic_store = EpisodicMemoryStore()
        self.semantic_bank = SemanticKnowledgeBank()
        self.advanced_mem = AdvancedMemory(base_model)

    def log_episode(self, episode_data):
        self.episodic_store.add_episode(episode_data)

    def store_fact(self, subj, relation, obj, confidence=1.0):
        self.semantic_bank.add_fact(subj, relation, obj, confidence)

    def train_batch(self, x, y, loss_fn):
        return self.advanced_mem.train_on_batch(x, y, loss_fn)

    def consolidate_models(self):
        self.advanced_mem.periodic_compress()

    def infer(self, x):
        return self.advanced_mem.infer(x)

