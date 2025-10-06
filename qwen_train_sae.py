from tqdm import tqdm
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from sae_train_bench import TrainableSAE

def collect_activations_hf(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    layer_index: int = 8,
    max_samples: int | None = 20000,
    device: torch.device = torch.device("cuda"),
    batch_tokenize_size: int = 8,
    max_length: int = 512,
    show_progress: bool = True,
):
    ds = load_dataset(dataset_name, dataset_config, split=split)
    collected = []
    total_tokens = 0
    model.eval()

    iterator = ds
    if show_progress:
        iterator = tqdm(ds, desc="Coletando ativações HF")

    for example in iterator:
        text = example.get("text") or example.get("content") or ""
        if not text or not text.strip():
            continue
        tokens = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        
        for i in range(0, len(tokens), max_length):
            window = tokens[i : i + max_length]
            inputs = tokenizer.prepare_for_model(window, return_tensors="pt")
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor) and v.dim() == 1:
                    inputs[k] = v.unsqueeze(0)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = out.hidden_states
            if layer_index < 0 or layer_index >= len(hidden_states):
                raise ValueError(f"layer_index {layer_index} inválido; hidden_states tem {len(hidden_states)} camadas")
            token_acts = hidden_states[layer_index]  

            if token_acts.dim() == 2:
                token_acts = token_acts.unsqueeze(0)

            b, seq_len, hid = token_acts.shape
            for bi in range(b):
                for si in range(seq_len):
                    vec = token_acts[bi, si].cpu()
                    collected.append(vec)
                    total_tokens += 1
                    if max_samples and total_tokens >= max_samples:
                        if show_progress:
                            print(f"Alcançado max_samples={max_samples}")
                        stacked = torch.stack(collected)
                        return stacked

    if not collected:
        raise RuntimeError("Nenhuma ativação coletada do dataset/modelo")

    stacked = torch.stack(collected)
    return stacked

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device escolhido: {device}")
    dev_str = args.device.lower()
    if dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA não disponível, mas --device pede cuda")
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"Usando device: {device}")

    print(f"Carregando tokenizer/model: {args.model_name} (fp16={args.fp16})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    model_kwargs = {}
    if args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
        
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
    if hf_token:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)

    acts = collect_activations_hf(
        model,
        tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        layer_index=args.hook_layer,
        max_samples=args.max_train_samples,
        device=device,
        max_length=args.max_length,
        show_progress=True,
    )
    if acts.dtype != torch.float32:
        acts = acts.to(dtype=torch.float32)

    print(f"Ativações coletadas shape: {acts.shape}")

    if args.dump_acts:
        os.makedirs(os.path.dirname(args.dump_acts) or ".", exist_ok=True)
        torch.save(acts, args.dump_acts)
        print(f"Ativações salvas em {args.dump_acts}")

    d_in = acts.shape[-1]
    sae = TrainableSAE(
        d_in=d_in,
        d_sae=args.d_sae,
        model_name=args.model_name,
        hook_layer=args.hook_layer,
        device=device,
        dtype=torch.float32,
        k=args.k,
    )

    torch.nn.init.xavier_uniform_(sae.W_enc)
    torch.nn.init.xavier_uniform_(sae.W_dec)

    def _collect_override(model_unused, dataset_name=None, dataset_config=None, split=None, max_samples=None, **kwargs):
        if max_samples is None or max_samples >= acts.shape[0]:
            return acts.to(device)
        else:
            return acts[:max_samples].to(device)

    sae.collect_activations = _collect_override

    results = sae.train_sae(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        device=device,
    )

    print("Treinamento concluído")
    print(results)

    if args.checkpoint_path:
        torch.save(sae.state_dict(), args.checkpoint_path)
        print(f"SAE salvo em {args.checkpoint_path}")

    sae.normalize_decoder()
    
    def hf_test_sae(sae_obj, hf_model, hf_tokenizer, hook_layer, device):
        sae_obj.eval()
        test_input = "The scientist named the population, after their distinctive horn, Ovid’s Unicorn. These four-horned, silver-white unicorns were previously unknown to science"
        inputs = hf_tokenizer(test_input, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = hf_model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states
        if hook_layer < 0 or hook_layer >= len(hidden_states):
            raise ValueError(f"hook_layer {hook_layer} inválido para hidden_states com {len(hidden_states)} camadas")
        acts = hidden_states[hook_layer]  

        encoded_acts = sae_obj.encode(acts)
        decoded_acts = sae_obj.decode(encoded_acts)

        flattened_acts = acts.reshape(-1, acts.shape[-1])
        reconstructed = sae_obj(flattened_acts).reshape(acts.shape)

        if not torch.allclose(reconstructed, decoded_acts, atol=1e-4):
            max_diff = torch.abs(reconstructed - decoded_acts).max().item()
            print(f"Aviso: reconstrução e decode diferem. max diff: {max_diff}")
        else:
            print("Teste HF: reconstrução e decode batem")

        l0 = (encoded_acts != 0).float().sum(-1).mean().item()
        print(f"average l0: {l0}")

    try:
        hf_test_sae(sae, model, tokenizer, args.hook_layer, device)
    except Exception as e:
        print("Falha no teste HF de consistência:", e)
        print("Tentando fallback para BaseSAE.test_sae (HookedTransformer) — pode falhar se o modelo não existir lá.")
        try:
            sae.test_sae(args.model_name)
        except Exception as e2:
            print("Fallback falhou também:", e2)
            print("Você pode testar manualmente fornecendo um model_name suportado por transformer_lens ou verificar o token/identificador HF.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina SAE a partir de ativações do Qwen via HF Transformers")
    parser.add_argument("--model_name", type=str, default="qwen/qwen-3", help="nome do modelo HF")
    parser.add_argument("--hook_layer", type=int, default=8, help="índice da camada hidden_states a coletar (0=embeddings)")
    parser.add_argument("--d_sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--checkpoint_path", type=str, default="sae_qwen_checkpoint.pt")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--fp16", action="store_true", help="carregar modelo em float16 (atenção: requer GPU)")
    parser.add_argument("--hf_token", type=str, default="", help="token de autenticação Hugging Face para repositórios privados (opcional)")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dump_acts", type=str, default="", help="caminho para salvar tensor de ativações (.pt)")
    parser.add_argument("--max_length", type=int, default=512, help="janela máxima de tokens ao cortar textos longos")

    args = parser.parse_args()
    main(args)
