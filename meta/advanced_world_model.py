# meta/advanced_world_model.py
import torch.nn as nn
import torch

class AdvancedWorldModel(nn.Module):
    """
    A recurrent latent world model (Dreamer-like) for counterfactual simulation.
    """
    def __init__(self, state_dim=10, action_dim=4, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.rnn = nn.GRU(input_size=state_dim+action_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_std = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state, action, hidden):
        inp = torch.cat([state, action], dim=-1).unsqueeze(1)
        output, hidden = self.rnn(inp, hidden)
        mean = self.fc_mean(output.squeeze(1))
        std = torch.exp(self.fc_std(output.squeeze(1)))
        return mean, std, hidden

    def simulate_rollout(self, initial_state, actions, steps=5):
        hidden = None
        state = torch.tensor([initial_state], dtype=torch.float32)
        results = []
        for i in range(steps):
            act = torch.tensor([actions[i]], dtype=torch.float32)
            mean, std, hidden = self.forward(state, act, hidden)
            next_state = mean.detach().numpy().tolist()
            results.append({"step": i, "next_state": next_state, "info": "Predicted by AdvancedWorldModel"})
            state = mean
        return results

