import torch
from transformers import GPT2Tokenizer, GPT2Model

class Llm:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
        self.model.to("cuda")
        
    def get_activations(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states[-1]
        return hidden_states.squeeze(0)