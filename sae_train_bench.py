import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from transformer_lens import HookedTransformer
from sae_bench.custom_saes.base_sae import BaseSAE

class TrainableSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        k: int = 20,
        hook_name: str | None = None,
    ):
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)
        self.k = k
        self.default_dataloader_config = {
            'shuffle': True
        }

    def encode(self, x: torch.Tensor):
        # encoder: x -> latents
        latents = x @ self.W_enc + self.b_enc
        # aplicando sparsity (top-k)
        latents_sparse = self.apply_sparsity(latents)
        return latents_sparse

    def decode(self, feature_acts: torch.Tensor):
        # decoder: latents -> reconstruction
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        latents_sparse = self.encode(x)
        recons = self.decode(latents_sparse)
        return recons

    def apply_sparsity(self, latents):
        # top-k sparsity
        k = min(self.k, latents.shape[-1])
        values, indices = torch.topk(latents, k=k, dim=-1)
        mask = torch.zeros_like(latents).scatter_(-1, indices, 1.0)
        return latents * mask

    def create_dataloader(self, dataset, batch_size, **kwargs):
        config = self.default_dataloader_config.copy()
        config.update(kwargs)
        return DataLoader(dataset, batch_size=batch_size, **config)

    def collect_activations(self, model, dataset_name="wikitext", split="train", max_samples=None):
        dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
        activations = []
        samples_processed = 0
        non_empty_samples = 0

        hook_name = self.cfg.hook_name

        for sample in tqdm(dataset, desc="Coletando ativações"):
            text = sample["text"]
            samples_processed += 1

            if not text or not text.strip():
                continue

            _, cache = model.run_with_cache(
                text,
                prepend_bos=True,
                names_filter=[hook_name],
                stop_at_layer=self.cfg.hook_layer + 1,
            )
            token_acts = cache[hook_name]

            if token_acts.size(1) == 0:
                continue

            non_empty_samples += 1
            if non_empty_samples == 1:
                print(f"\nprimeiro texto não vazio: '{text[:100]}...'")
                print(f"shape das ativações: {token_acts.shape}")

            if non_empty_samples % 1000 == 0:
                print(f"processadas {non_empty_samples} amostras não vazias, {len(activations)} ativações coletadas")

            for token_act in token_acts.squeeze(0):
                activations.append(token_act)

            if max_samples and non_empty_samples >= max_samples:
                print(f"Parando em {max_samples} amostras...")
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

    def loss(self, recons, target):
        return F.mse_loss(recons, target)

    @torch.no_grad()
    def quick_kpis(self, data_tensor: torch.Tensor):
        self.eval()
        x = data_tensor.to(self.device)
        recons = self(x)

        mse = F.mse_loss(recons, x).item()
        cos = F.cosine_similarity(x, recons, dim=-1).mean().item()
        ss_res = torch.sum((x - recons) ** 2)
        ss_tot = torch.sum((x - x.mean()) ** 2)
        r2 = (1 - ss_res / ss_tot).item()

        z_sparse = self.encode(x)
        feature_usage_frac = (z_sparse != 0).any(dim=0).float().mean().item()
        l0_avg = (z_sparse != 0).sum(dim=-1).float().mean().item()

        return {
            "mse": mse,
            "cos": cos,
            "r2": r2,
            "feature_usage_frac": feature_usage_frac,
            "l0_avg": l0_avg
        }

    def train_sae(
        self,
        model,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        eval_split=0.1,
        max_train_samples=None,
        device="cuda"
    ):
        torch.cuda.empty_cache()

        self.to(device)

        activations = self.collect_activations(model, max_samples=max_train_samples)
        n_test = int(len(activations) * eval_split)
        n_train = len(activations) - n_test
        train_data, test_data = torch.utils.data.random_split(
            activations, [n_train, n_test], generator=torch.Generator().manual_seed(42)
        )

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
                    print(f"Época {epoch+1}, Batch {bi}/{len(dataloader)}, Loss: {loss.item():.6f}")

            print(f"Época {epoch+1}/{epochs}, Loss média: {total_loss / len(dataloader):.6f}")

        # Final evaluation
        pre_kpis = self.quick_kpis(test_tensor)
        print(f"\n[FINAL] MSE: {pre_kpis['mse']:.6f} | Cos: {pre_kpis['cos']:.4f} | R²: {pre_kpis['r2']:.4f} | Uso-features: {pre_kpis['feature_usage_frac']:.3f} | L0: {pre_kpis['l0_avg']:.1f}")

        return pre_kpis

    @torch.no_grad()
    def normalize_decoder(self):
        original_dtype = self.W_dec.dtype
        self.to(dtype=torch.float32)
        tolerance = 1e-1  

        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        print("decoder vectors are not normalized. Normalizing.")

        test_input = torch.randn(10, self.cfg.d_in).to(
            dtype=self.dtype, device=self.device
        )
        initial_output = self(test_input)

        self.W_dec.data /= norms[:, None]

        new_norms = torch.norm(self.W_dec, dim=1)

        if not torch.allclose(new_norms, torch.ones_like(new_norms), atol=tolerance):
            max_norm_diff = torch.max(torch.abs(new_norms - torch.ones_like(new_norms)))
            print(f"max diff in norms: {max_norm_diff.item()}")
            print("warning: Decoder weights may not be perfectly normalized, but continuing...")

        self.W_enc *= norms
        self.b_enc *= norms

        new_output = self(test_input)

        max_diff = torch.abs(initial_output - new_output).max()
        print(f"max difference in output: {max_diff}")

        if not torch.allclose(initial_output, new_output, atol=tolerance):
            print(f"warning: Output difference ({max_diff.item():.6f}) is larger than tolerance ({tolerance})")
            print("this might affect some evaluations, but SAE should still work.")
        else:
            print("normalization successful!")

        self.to(dtype=original_dtype)

if __name__ == "__main__":
    model_name = "EleutherAI/pythia-70m-deduped"
    hook_layer = 4
    d_in = 512  # pythia-70m
    d_sae = 2048  # example size
    k = 20  # sparsity level

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    model = HookedTransformer.from_pretrained(model_name, device=device)

    sae = TrainableSAE(
        d_in=d_in,
        d_sae=d_sae,
        model_name=model_name,
        hook_layer=hook_layer,
        device=device,
        dtype=dtype,
        k=k
    )
    
    nn.init.xavier_uniform_(sae.W_enc)
    nn.init.xavier_uniform_(sae.W_dec)

    results = sae.train_sae(
        model=model,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        max_train_samples=10000,
        device=device
    )

    print("treinamento concluído")
    print(f"resultados: {results}")

    checkpoint_path = "sae_checkpoint.pt"
    torch.save(sae.state_dict(), checkpoint_path)
    print(f"SAE salvo em {checkpoint_path}")

    sae.normalize_decoder()
    sae.test_sae(model_name)