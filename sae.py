from llm import Llm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.optim as optim
from tqdm import tqdm  
import numpy as np

class Sae(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = Llm(model_name="gpt2")
        self.encoder = nn.Linear(768, 12288)
        self.decoder = nn.Linear(12288, 768)
        self.k = 20

    def collect_activations(self):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        activations = []
        for sample in tqdm(dataset, desc="Coletando ativações"):
            text = sample["text"]
            token_acts = self.llm.get_activations(text)
            for token_act in token_acts:
                activations.append(token_act)
        return torch.stack(activations)
   
    def apply_sparsity(self, latents):
        #vou ser fiel ao paper e usar top_k, mas posso usar uma relu + regu dps
        values, indices = torch.topk(latents, k=20, dim=-1)  
        mask = torch.zeros_like(latents).scatter_(-1, indices, 1.0)
        return latents * mask
    
    def forward(self, x):
        latents = self.encoder(x)
        latents_sparse = self.apply_sparsity(latents)
        recons = self.decoder(latents_sparse)
        return recons
    
    def loss(self, recons, target):
        return F.mse_loss(recons, target)
        
    def train(self, epochs=5, batch_size=64, lr=1e-3, device="cuda"):
        self.to(device)
        activations = self.collect_activations()
        dataset = torch.utils.data.TensorDataset(activations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(device)
                recons = self(x)
                loss = self.loss(recons, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            media_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {media_loss:.6f}")
            
sae = Sae()
sae.train(epochs=10, batch_size=64, lr=1e-3, device="cuda")
        
        