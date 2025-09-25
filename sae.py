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
        processed_count = 0
        error_count = 0
        empty_count = 0
        
        for sample in tqdm(dataset, desc="Coletando ativações"):
            text = sample["text"]
            if not text or not text.strip():
                empty_count += 1
                continue
            try:
                # Debug: print first few texts that aren't empty
                if processed_count == 0:
                    print(f"First valid text sample: '{text[:100]}...'")
                
                token_acts = self.llm.get_activations(text)
                print(f"Got token_acts: {type(token_acts)}, length: {len(token_acts) if token_acts is not None else 'None'}")
                
                if token_acts is not None and len(token_acts) > 0:
                    for token_act in token_acts:
                        activations.append(token_act)
                    processed_count += 1
                
                # Limit to avoid memory issues - collect only first 10000 activations
                if len(activations) >= 10000:
                    break
                    
            except RuntimeError as e:
                # Debug: print the actual error for the first few
                if error_count < 5:
                    print(f"RuntimeError {error_count + 1}: {str(e)}")
                    print(f"Text causing error: '{text[:50]}...'")
                error_count += 1
                continue
            except Exception as e:
                # Catch any other errors
                if error_count < 5:
                    print(f"Other error: {type(e).__name__}: {str(e)}")
                error_count += 1
                continue
        
        print(f"Processed: {processed_count}, Empty: {empty_count}, Errors: {error_count}")
        print(f"Total activations collected: {len(activations)}")
        
        if len(activations) == 0:
            raise ValueError("No activations collected. Check your LLM implementation.")
        
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
        self.llm.model.to(device)
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
        
        