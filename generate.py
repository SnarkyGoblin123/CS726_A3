import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        max_output_len: int = 10,
        tau: int = 1,
        k: int = 10,
        p: int = 0.5
    ) -> None:
        '''
            Initialize the TextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            tau: Temperature parameter for random sampling
            k: Top-k parameter for top-k sampling
            p: Cumulative probability threshold for nucleus sampling
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]: 
        '''
            Implement Greedy decoding technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        generated_tokens = []
        curr_inp = input_ids
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(curr_inp)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                if next_token.item() == self.eos_token_id:
                    break
                generated_tokens.append(next_token.item())
                curr_inp = torch.cat([curr_inp, next_token], dim=1)
        
        return torch.tensor(generated_tokens, device=input_ids.device).squeeze(0)
        
    def random_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Random sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
        generated_tokens = []
        curr_inp = input_ids
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(curr_inp)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                scaled_probs = probs**(1/self.tau)
                scaled_probs /= torch.sum(scaled_probs, dim=-1, keepdim=True)
                next_token = torch.multinomial(scaled_probs, num_samples=1)
                if next_token.item() == self.eos_token_id:
                    break
                generated_tokens.append(next_token.item())
                curr_inp = torch.cat([curr_inp, next_token], dim=1)
        
        return torch.tensor(generated_tokens, device=input_ids.device).squeeze(0)
    
    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Top-k sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        generated_tokens = []
        curr_inp = input_ids
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(curr_inp)
                logits = outputs.logits[:, -1, :]
                top_k_values, top_k_indices = torch.topk(logits, self.k, dim=-1)
                probs = torch.softmax(top_k_values, dim=-1)
                next_token = top_k_indices.gather(-1, torch.multinomial(probs, num_samples=1))
                if next_token.item() == self.eos_token_id:
                    break
                generated_tokens.append(next_token.item())
                curr_inp = torch.cat([curr_inp, next_token], dim=1)
        return torch.tensor(generated_tokens, device=input_ids.device).squeeze(0)
    
    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Nucleus sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
        generated_tokens = []
        curr_inp = input_ids
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(curr_inp)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_prob = torch.cumsum(sorted_probs, dim=-1)
                mask_index = cum_prob > self.p
                mask_index[..., 1:] = mask_index[..., :-1].clone()
                mask_index[..., 0] = 0
                rem_index = sorted_indices[mask_index]
                probs[0, rem_index] = 0

                next_token = torch.multinomial(probs, num_samples=1)
                if next_token.item() == self.eos_token_id:
                    break

                generated_tokens.append(next_token.item())
                curr_inp = torch.cat([curr_inp, next_token], dim=1)
        return torch.tensor(generated_tokens, device=input_ids.device).squeeze(0)