# knowledge/ontology_builder.py
import numpy as np
from sklearn.cluster import KMeans
import hashlib
import torch
from transformers import BertModel, BertTokenizer

class ConceptAbstractionLayer:
    """
    Uses a pretrained transformer (BERT) to encode raw observations into abstract embeddings.
    """
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.transformer = BertModel.from_pretrained(model_name)
        self.transformer.eval()

    def encode(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = self.transformer(**inputs)
        return outputs.last_hidden_state[:, 0, :]

class OntologyBuilder:
    """
    Generates abstract concept embeddings from textual observations.
    """
    def __init__(self, knowledge_graph, n_clusters=3):
        self.kg = knowledge_graph
        self.n_clusters = n_clusters
        self.abstractor = ConceptAbstractionLayer()

    def infer_new_concept(self, observations, domain_label=None):
        if len(observations) < self.n_clusters:
            return
        embeddings = self.abstractor.encode(observations).numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(embeddings)
        for i in range(self.n_clusters):
            center = kmeans.cluster_centers_[i]
            concept_hash = hashlib.md5(center.tobytes()).hexdigest()[:6]
            concept_name = f"Concept_{concept_hash}"
            self.kg.add_concept(concept_name, properties={"abstract_center": center.tolist(), "domain": domain_label})

