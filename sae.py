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
        samples_processed = 0
        non_empty_samples = 0
        
        for sample in tqdm(dataset, desc="Coletando ativações"):
            text = sample["text"]
            samples_processed += 1
            
            if not text or not text.strip():
                continue
            
            token_acts = self.llm.get_activations(text)
            if token_acts.size(0) == 0:
                continue
                
            non_empty_samples += 1
            if non_empty_samples == 1:
                print(f"\nprimeiro texto não vazio: '{text[:100]}...'")
                print(f"shape das ativações: {token_acts.shape}")
            
            if non_empty_samples % 1000 == 0:
                print(f"processadas {non_empty_samples} amostras não vazias, {len(activations)} ativações coletadas")
            
            for token_act in token_acts:
                activations.append(token_act)
            
            if samples_processed >= len(dataset) * 0.3:
                print(f"Parando em 30% do dataset...")
                break
                
        print(f"processadas: {samples_processed}")
        print(f"não vazias: {non_empty_samples}")
        print(f"ativações coletadas: {len(activations)}")
        
        if activations:
            stacked = torch.stack(activations)
            print(f"shape final: {stacked.shape}")
            return stacked
        else:
            raise ValueError("nenhuma ativação foi coletada")
   
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
    
    def evaluate_metrics(self, test_data, device="cuda"):
        """
        Calcula métricas importantes para avaliar a qualidade do SAE
        """
        nn.Module.eval(self)
        self.to(device)
        
        with torch.no_grad():
            test_data = test_data.to(device)
            latents = self.encoder(test_data)
            latents_sparse = self.apply_sparsity(latents)
            recons = self.decoder(latents_sparse)
            recon_loss = F.mse_loss(recons, test_data).item()
            active_neurons = (latents_sparse != 0).float().mean().item()
            sparsity_pct = active_neurons * 100
            l0_norm = (latents_sparse != 0).float().sum(dim=-1).mean().item()
            ss_res = torch.sum((test_data - recons) ** 2)
            ss_tot = torch.sum((test_data - test_data.mean()) ** 2)
            r2_score = 1 - (ss_res / ss_tot).item()
            cos_sim = F.cosine_similarity(test_data, recons, dim=-1).mean().item()
            feature_usage = (latents_sparse != 0).float().mean(dim=0)
            dead_features = (feature_usage == 0).sum().item()
            active_features = (feature_usage > 0).sum().item()
            per_dim_error = ((test_data - recons) ** 2).mean(dim=0)
            max_error_dim = per_dim_error.max().item()
            min_error_dim = per_dim_error.min().item()
            
            metrics = {
                'reconstruction_loss': recon_loss,
                'sparsity_percent': sparsity_pct,
                'l0_norm': l0_norm,
                'r2_score': r2_score,
                'cosine_similarity': cos_sim,
                'active_features': active_features,
                'dead_features': dead_features,
                'total_features': latents_sparse.shape[-1],
                'max_error_dim': max_error_dim,
                'min_error_dim': min_error_dim,
                'feature_usage_std': feature_usage.std().item()
            }
            
        return metrics
    
    def print_metrics(self, metrics):
        print("\n" + "="*50)
        print("         MÉTRICAS DO SAE")
        print("="*50)
        print(f"Reconstruction Loss (MSE): {metrics['reconstruction_loss']:.6f}")
        print(f"R² Score:                  {metrics['r2_score']:.4f}")
        print(f"Cosine Similarity:         {metrics['cosine_similarity']:.4f}")
        print()
        print("--- SPARSITY ---")
        print(f"Sparsity Level:            {metrics['sparsity_percent']:.2f}%")
        print(f"L0 Norm (avg active):      {metrics['l0_norm']:.1f}")
        print()
        print("--- FEATURES ---")
        print(f"Active Features:           {metrics['active_features']}/{metrics['total_features']}")
        print(f"Dead Features:             {metrics['dead_features']}")
        print(f"Feature Usage Std:         {metrics['feature_usage_std']:.4f}")
        print()
        print("--- RECONSTRUCTION ---")
        print(f"Max Error (dimension):     {metrics['max_error_dim']:.6f}")
        print(f"Min Error (dimension):     {metrics['min_error_dim']:.6f}")
        print("="*50)
    
    def analyze_top_features(self, test_data, top_k=10, device="cuda"):
        """Analisa as features mais ativas"""
        nn.Module.eval(self)
        self.to(device)
        
        with torch.no_grad():
            test_data = test_data.to(device)
            latents = self.encoder(test_data)
            latents_sparse = self.apply_sparsity(latents)

            feature_activations = latents_sparse.abs().mean(dim=0)  
            feature_frequency = (latents_sparse != 0).float().mean(dim=0)
            
            top_activation_idx = feature_activations.topk(top_k).indices
            
            top_frequency_idx = feature_frequency.topk(top_k).indices
            
            print(f"\n{'='*60}")
            print(f"           TOP {top_k} FEATURES ANALYSIS")
            print(f"{'='*60}")
            
            print(f"\n--- TOP {top_k} POR ATIVAÇÃO MÉDIA ---")
            for i, idx in enumerate(top_activation_idx):
                print(f"feature {idx:4d}: ativação={feature_activations[idx]:.4f}, frequência={feature_frequency[idx]:.4f}")
            
            print(f"\n--- TOP {top_k} POR FREQUÊNCIA ---")
            for i, idx in enumerate(top_frequency_idx):
                print(f"feature {idx:4d}: frequência={feature_frequency[idx]:.4f}, ativação={feature_activations[idx]:.4f}")
            
            return {
                'feature_activations': feature_activations,
                'feature_frequency': feature_frequency,
                'top_activation_features': top_activation_idx,
                'top_frequency_features': top_frequency_idx
            }
        
    def train_sae(self, epochs=5, batch_size=64, lr=1e-3, device="cuda", eval_split=0.1):
        torch.cuda.empty_cache()
        
        self.to(device)
        activations = self.collect_activations()
        n_test = int(len(activations) * eval_split)
        n_train = len(activations) - n_test
        
        train_data, test_data = torch.utils.data.random_split(
            activations, [n_train, n_test]
        )
        
        print(f"treino: {n_train}, teste: {n_test}")
        test_tensor = torch.stack([test_data[i] for i in range(len(test_data))])
        
        train_dataset = torch.utils.data.TensorDataset(torch.stack([train_data[i] for i in range(len(train_data))]))
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        print(f"batches por época: {len(dataloader)}")

        for epoch in range(epochs):
            print(f"\nÉpoca {epoch+1}/{epochs}")
            total_loss = 0.0
            batch_count = 0
            
            for batch in dataloader:
                x = batch[0].to(device)
                recons = self(x)
                loss = self.loss(recons, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                
                if batch_count % 1000 == 0:
                    current_loss = total_loss / batch_count
                    progress = (batch_count / len(dataloader)) * 100
                    print(f"batch {batch_count}/{len(dataloader)} ({progress:.1f}%) - Loss atual: {current_loss:.6f}")
                
            media_loss = total_loss / len(dataloader)
            print(f"epoca {epoch+1} completa - Loss final: {media_loss:.6f}")
        
        print("\n métricas no conjunto de teste")
        metrics = self.evaluate_metrics(test_tensor, device)
        self.print_metrics(metrics)
        feature_analysis = self.analyze_top_features(test_tensor, top_k=15, device=device)
        
        return metrics, feature_analysis
            
sae = Sae()
metrics, features = sae.train_sae(epochs=10, batch_size=32, lr=1e-3, device="cuda")
        
        