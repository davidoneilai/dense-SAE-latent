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
        self.default_dataloader_config = {
            'shuffle': True
        }
    
    def create_dataloader(self, dataset, batch_size, **kwargs):
        config = self.default_dataloader_config.copy()
        config.update(kwargs)
        return DataLoader(dataset, batch_size=batch_size, **config)

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
        k = min(self.k, latents.shape[-1])
        values, indices = torch.topk(latents, k=k, dim=-1)  
        mask = torch.zeros_like(latents).scatter_(-1, indices, 1.0)
        return latents * mask
    
    def forward(self, x):
        latents = self.encoder(x)
        latents_sparse = self.apply_sparsity(latents)
        recons = self.decoder(latents_sparse)
        return recons
    
    def loss(self, recons, target):
        return F.mse_loss(recons, target)
    
    @torch.no_grad()
    def recon_loss(self, data_tensor: torch.Tensor, device="cuda"):
        self.eval().to(device)
        x = data_tensor.to(device)
        recons = self(x)
        return F.mse_loss(recons, x).item()
    
    # Importância do neurônio ---
    def neuron_importance_structural(self):
        with torch.no_grad():
            enc = self.encoder.weight.data      # [out=12288, in=768]
            dec = self.decoder.weight.data      # [out=768, in=12288]
            enc_col = torch.norm(enc, dim=1)    # [12288]
            dec_row = torch.norm(dec, dim=0)    # [12288]
            imp = enc_col * dec_row             # [12288]
        return imp

    def neuron_importance_usage(self, sample_acts: torch.Tensor, device="cuda"):
        self.eval()
        with torch.no_grad():
            z = self.encoder(sample_acts.to(device))      # [N, 12288]
            z_sparse = self.apply_sparsity(z)
            freq = (z_sparse != 0).float().mean(dim=0)    # [12288]
            act  = z_sparse.abs().mean(dim=0)             # [12288]
            imp = torch.sqrt((freq + 1e-8) * (act + 1e-8))
        return imp, freq, act

    def neuron_importance_saliency(self, sample_acts: torch.Tensor, device="cuda"):
        self.train().to(device)
        x = sample_acts.to(device)
        z = self.encoder(x)                                # [N, 12288]
        z.retain_grad()
        z_sparse = self.apply_sparsity(z)
        recons = self.decoder(z_sparse)
        loss = F.mse_loss(recons, x)
        self.zero_grad(set_to_none=True)
        loss.backward()
        sal = (z.grad.abs() * z.abs()).mean(dim=0).detach()   # [12288]
        return sal
    
    # Combinação e ranking ---
    def _zscore(self, t: torch.Tensor):
        m, s = t.mean(), t.std().clamp_min(1e-8)
        return (t - m) / s

    def rank_neurons(self, sample_acts: torch.Tensor, use_saliency=True, device="cuda",
                    weights=(0.4, 0.4, 0.2)):
        w_struct, w_usage, w_sal = weights
        s_struct = self.neuron_importance_structural().to(device)      # ↑ melhor
        s_usage, _, _ = self.neuron_importance_usage(sample_acts, device=device)
        if use_saliency:
            s_sal = self.neuron_importance_saliency(sample_acts, device=device)
        else:
            s_sal = torch.zeros_like(s_struct)

        score = w_struct * self._zscore(s_struct) + w_usage * self._zscore(s_usage) + w_sal * self._zscore(s_sal)
        # menor score = menos importante → candidatos a poda
        order = torch.argsort(score)  # ascendente
        return order, score
    
    # pruning e recuperação
    def prune_neurons(self, prune_idx: torch.Tensor):
        with torch.no_grad():
            self.encoder.weight.data[prune_idx, :] = 0.0
            if self.encoder.bias is not None:
                self.encoder.bias.data[prune_idx] = 0.0
            self.decoder.weight.data[:, prune_idx] = 0.0

    def recover_after_prune(self, train_dataset, batch_size=64, epochs=2, lr=3e-4, device="cuda"):
        loader = self.create_dataloader(train_dataset, batch_size)
        opt = optim.Adam(self.parameters(), lr=lr)
        self.train().to(device)
        for _ in range(epochs):
            for (x,) in loader:
                x = x.to(device)
                recons = self(x)
                loss = self.loss(recons, x)
                opt.zero_grad()
                loss.backward()
                opt.step()

        
    def train_sae(self, epochs=5, batch_size=64, lr=1e-3, device="cuda",
              eval_split=0.1, prune_pct=0.0, saliency_sample_size=0,
              recovery_epochs=2, recovery_lr=3e-4):
        torch.cuda.empty_cache()
        
        self.to(device)
        activations = self.collect_activations()
        n_test = int(len(activations) * eval_split)
        n_train = len(activations) - n_test
        train_data, test_data = torch.utils.data.random_split(activations, [n_train, n_test])
        
        print(f"treino: {n_train}, teste: {n_test}")
        test_tensor = torch.stack([test_data[i] for i in range(len(test_data))])
        train_tensor = torch.stack([train_data[i] for i in range(len(train_data))])
        
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        dataloader = self.create_dataloader(train_dataset, batch_size)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        print(f"batches por época: {len(dataloader)}")

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for bi, (x,) in enumerate(dataloader, 1):
                x = x.to(device)
                recons = self(x)
                loss = self.loss(recons, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if bi % 1000 == 0:
                    print(f"epoch {epoch+1} | batch {bi}/{len(dataloader)} - loss {total_loss/bi:.6f}")

            print(f"época {epoch+1} completa - loss médio: {total_loss/len(dataloader):.6f}")
        
        pre_loss = self.recon_loss(test_tensor, device=device)
        print(f"\n[CHECK] Recon MSE (pré-pruning): {pre_loss:.6f}")

        results = {
            "pre_loss": pre_loss,
            "post_loss": None,
            "num_pruned": 0,
            "total_neurons": self.encoder.weight.shape[0],
            "prune_idx": None
        }

        # Pruning opcional
        if prune_pct > 0.0:
            sample_acts = test_tensor
            if 0 < saliency_sample_size < len(test_tensor):
                idx = torch.randperm(len(test_tensor))[:saliency_sample_size]
                sample_acts = test_tensor[idx]

            print(f"\n[RANK] calculando importâncias (saliency={'on' if saliency_sample_size>0 else 'off'})...")
            order, _ = self.rank_neurons(sample_acts, use_saliency=(saliency_sample_size>0), device=device)
            num_prune = int(order.numel() * prune_pct)
            prune_idx = order[:num_prune]
            print(f"[PRUNE] removendo {num_prune}/{order.numel()} neurônios ({prune_pct*100:.1f}%)")
            self.prune_neurons(prune_idx)

            print(f"[RECOVERY] {recovery_epochs} épocas @ lr={recovery_lr}")
            self.recover_after_prune(
                torch.utils.data.TensorDataset(train_tensor),
                batch_size=batch_size, epochs=recovery_epochs, lr=recovery_lr, device=device
            )

            post_loss = self.recon_loss(test_tensor, device=device)
            print(f"[CHECK] Recon MSE (pós-pruning): {post_loss:.6f}")

            results.update({
                "post_loss": post_loss,
                "num_pruned": num_prune,
                "prune_idx": prune_idx.detach().cpu()
            })

        return results
            
if __name__ == "__main__":
    sae = Sae()
    res = sae.train_sae(
        epochs=10, batch_size=64, lr=1e-3, device="cuda",
        prune_pct=0.20,             # 20% menos importantes
        saliency_sample_size=0,     # 0 - sem saliency
        recovery_epochs=2, recovery_lr=3e-4
    )
    print("\nResumo:", {k: (v if k != "prune_idx" else f"tensor[{len(v)}]") for k, v in res.items()})
        
        