# memory/unsupervised_memory.py
import torch
import torch.nn as nn

class MemoryAutoencoder(nn.Module):
    """
    An autoencoder for compressing episodic memory.
    """
    def __init__(self, input_dim=100, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

class UnsupervisedMemoryConsolidator:
    """
    Consolidates episodic memory using an autoencoder.
    """
    def __init__(self, input_dim=100, latent_dim=32, lr=1e-4):
        self.autoencoder = MemoryAutoencoder(input_dim, latent_dim)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def consolidate(self, memory_tensor):
        self.autoencoder.train()
        z, recon = self.autoencoder(memory_tensor)
        loss = self.loss_fn(recon, memory_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return z, loss.item()

