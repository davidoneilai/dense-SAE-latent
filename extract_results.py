import json
import pandas as pd
from pathlib import Path

def extract_metrics_to_csv():    
    results_file = "eval_results/core/meu_sae_treinado_custom_sae_eval_results.json"
    
    if not Path(results_file).exists():
        print(f"arquivo {results_file} não encontrado!")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    metrics = {}
    metrics['sae_name'] = 'meu_sae_treinado'
    metrics['model'] = data['eval_config']['model_name']
    metrics['eval_type'] = data['eval_type_id']
    recon = data['eval_result_metrics']['reconstruction_quality']
    metrics['explained_variance'] = recon['explained_variance']
    metrics['mse'] = recon['mse']
    metrics['cosine_similarity'] = recon['cossim']
    sparsity = data['eval_result_metrics']['sparsity']
    metrics['l0'] = sparsity['l0']
    metrics['l1'] = sparsity['l1']
    perf = data['eval_result_metrics']['model_performance_preservation']
    metrics['ce_loss_score'] = perf['ce_loss_score']
    shrink = data['eval_result_metrics']['shrinkage']
    metrics['l2_ratio'] = shrink['l2_ratio']
    
    try:
        dead_neurons = data['eval_result_metrics']['sparsity_metrics']['dead_neurons']
        metrics['dead_neurons_frac'] = dead_neurons.get('frac_alive', 'N/A')
    except KeyError:
        metrics['dead_neurons_frac'] = 'N/A'
    
    df = pd.DataFrame([metrics])
    csv_file = "eval_results/core/sae_metrics_summary.csv"
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(df.to_string(index=False, float_format='%.4f'))
    print(f"\n csv ta em: {csv_file}")
    
    return df

if __name__ == "__main__":
    extract_metrics_to_csv()