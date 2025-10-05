import torch
import os
from sae_train_bench import TrainableSAE
import sae_bench.custom_saes.custom_sae_config as custom_sae_config
from transformer_lens import HookedTransformer

def load_trained_sae(checkpoint_path=None):
    model_name = "pythia-70m-deduped"
    hook_layer = 4
    d_in = 512  # pythia-70m
    d_sae = 2048
    k = 20
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    str_dtype = "float32"

    # SAE
    sae = TrainableSAE(
        d_in=d_in,
        d_sae=d_sae,
        model_name=model_name,
        hook_layer=hook_layer,
        device=device,
        dtype=dtype,
        k=k
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Carregando SAE de {checkpoint_path}")
        sae.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        sae.to(device)
        
        print("Normalizando decoder weights")
        try:
            sae.normalize_decoder()
            print("norm concluída!")
        except Exception as e:
            print(f"aplicando normalização manual")
            with torch.no_grad():
                norms = torch.norm(sae.W_dec, dim=1, keepdim=True)
                sae.W_dec.data /= norms
                sae.W_enc.data *= norms.T
                sae.b_enc.data *= norms.squeeze()
                print("normalização manual aplicada!")
    else:
        raise FileNotFoundError(f"checkpoint {checkpoint_path} não encontrado")

    sae.cfg = custom_sae_config.CustomSAEConfig(
        model_name, d_in=d_in, d_sae=d_sae,
        hook_name=f"blocks.{hook_layer}.hook_resid_post",
        hook_layer=hook_layer
    )
    sae.cfg.dtype = str_dtype
    sae.cfg.architecture = "custom_trained_topk"

    return sae, model_name, device, str_dtype

def run_individual_evaluations(sae, model_name, device, str_dtype):
    selected_saes = [("meu_sae_treinado", sae)]
    
    print("executando avaliação CORE")
    try:
        from sae_bench.evals.core.main import multiple_evals
        
        core_results = multiple_evals(
            selected_saes=selected_saes,
            n_eval_reconstruction_batches=100,
            n_eval_sparsity_variance_batches=200,
            eval_batch_size_prompts=8,
            compute_featurewise_density_statistics=True,
            compute_featurewise_weight_based_metrics=True,
            output_folder="eval_results/core",
            dtype=str_dtype,
            device="cpu",
            force_rerun=True,
            verbose=True
        )
        print("core evaluation concluída!")
        
    except Exception as e:
        print(f"erro na avaliação core: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nexecutando avaliação SPARSE PROBING")
    try:
        from sae_bench.evals.sparse_probing.main import run_eval
        from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
        
        config = SparseProbingEvalConfig(
            model_name=model_name,
            random_seed=42,
            llm_batch_size=32,
            llm_dtype=str_dtype,
        )
        
        sparse_results = run_eval(
            config=config,
            selected_saes=selected_saes,
            device="cpu",
            output_folder="eval_results/sparse_probing",
            force_rerun=True
        )
        print("sparse probing evaluation concluída!")
        
    except Exception as e:
        print(f"erro na avaliação sparse probing: {e}")
        import traceback
        traceback.print_exc()

def run_benchmark_evaluation(sae, model_name, device, str_dtype, eval_types=None):
    print("executando avaliações individuais")
    run_individual_evaluations(sae, model_name, device, str_dtype)
def main():
    print("avaliando SAE no SAEBench")
    
    checkpoint_path = "sae_checkpoint.pt"

    sae, model_name, device, str_dtype = load_trained_sae(checkpoint_path)

    run_benchmark_evaluation(sae, model_name, device, str_dtype)
    
    print("\nprocesso de avaliação concluído!")
if __name__ == "__main__":
    main()