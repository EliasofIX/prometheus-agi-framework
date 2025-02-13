# memory/advanced_memory.py
import torch
import copy

class AdvancedMemory:
    """
    Implements progress & compress memory consolidation.
    """
    def __init__(self, model, lr=1e-4):
        self.stable_model = copy.deepcopy(model)
        self.active_model = model
        self.optimizer = torch.optim.Adam(self.active_model.parameters(), lr=lr)
        self.consolidation_count = 0

    def train_on_batch(self, x, y, loss_fn):
        self.active_model.train()
        self.optimizer.zero_grad()
        pred = self.active_model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def periodic_compress(self):
        for stable_param, active_param in zip(self.stable_model.parameters(), self.active_model.parameters()):
            stable_param.data = 0.5 * stable_param.data + 0.5 * active_param.data
        self.consolidation_count += 1

    def infer(self, x):
        self.stable_model.eval()
        with torch.no_grad():
            return self.stable_model(x)

