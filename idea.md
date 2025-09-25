beleza, minha ideia vai ser seguir essa arquitetura:

1. Entrada (ativação do LLM) -> seria o vetor de dimensão do modelo em si (d_model, poderia ser uma residual stream)
- input_dim = d_model

2. Encoder (denso -> latente) -> uma projeção linear que leva o input_dim para latent_dim, umas 16x maior
- Isso porque, o SAE consiste em ter mais neurônios latentes do que de entrada (Linear(input_dim, latent_dim)) 

3. Ativação + Sparsidade -> depois do encoder, eu aplico uma função que zera quase todos os latentes, deixando ativo só os K maiores
- Posso usar aqui uma ReLU + regularização de sparsidade também, mas o paper principal usa o TopK diretamente
- latents = topk(encoder_output, k)

4. Decoder (latente -> reconstrução) -> uma projeção linear que vai levar o latent_dim de volta para o input_dim 
- reconstruar a ativação original, mas usando apenas os latentes "ativos"
- Linear(latent_dim, input_dim)

5. Loss (diff entre o input e output - MSE)
- no paper que eu vi, eles usam um viés inicial negativo para incentivar a sparsidade, isso porque se através da f_ativ eu setar monte de negativos como zeros, maioria dos neurônios ficaram inativos 
- loss = mse(x_reconstructed, x_original) + reg_terms

6. Análise/uso pós-treino
- a ideia é ver quais neurônios mais ativam para certos inputs
- agrupar esses inputs pelos latentes acionados
- manipular esses neuronios latentes para ver como fica
- analise(latentes, exemple)
