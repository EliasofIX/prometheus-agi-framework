# knowledge/cross_domain_transfer.py
import torch
import torch.nn as nn

class CrossDomainTransfer:
    """
    Computes similarity between abstract concept embeddings for transfer learning.
    """
    def __init__(self, knowledge_graph, embed_dim=8):
        self.kg = knowledge_graph
        self.embed_dim = embed_dim
        self.projector = nn.Linear(embed_dim, embed_dim)

    def identify_transfer_candidates(self, domainA, domainB):
        domainA_nodes = []
        domainB_nodes = []
        for n, data in self.kg.graph.nodes(data=True):
            props = data.get("props", {})
            if props.get("domain") == domainA and "abstract_center" in props:
                domainA_nodes.append((n, torch.tensor(props["abstract_center"], dtype=torch.float32)))
            if props.get("domain") == domainB and "abstract_center" in props:
                domainB_nodes.append((n, torch.tensor(props["abstract_center"], dtype=torch.float32)))
        matches = []
        for nameA, embA in domainA_nodes:
            projA = self.projector(embA)
            for nameB, embB in domainB_nodes:
                projB = self.projector(embB)
                similarity = nn.functional.cosine_similarity(projA.unsqueeze(0), projB.unsqueeze(0)).item()
                if similarity > 0.8:
                    matches.append((nameA, nameB, similarity))
        return matches

