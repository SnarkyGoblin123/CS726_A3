import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

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
        current_input = input_ids.clone()
        
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)
                logits = outputs.logits[:, -1, :]  # Get LM head's logits
                
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1)
            
            if next_token.item() == self.eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=-1)
        
        return torch.tensor(generated_tokens, device=input_ids.device)

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

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
        current_input = input_ids.clone()
        while len(generated_tokens) < self.max_output_len:
            #Step1: Find prob of S+1 heads
            with torch.no_grad():
                outputs = self.model.model(current_input)
                last_state = outputs.last_hidden_state[:,-1,:]
                lm_logits = self.model.lm_head(last_state)
                # print("lm_logits",lm_logits.shape)
                medusa_logits = torch.cat([self.model.medusa_head[i](last_state) for i in range(self.no_heads - 1)], dim=0)
                # print("medusa_logits",medusa_logits.shape)
                logits = torch.cat([lm_logits,medusa_logits], dim=0)
                # print("logits",logits.shape)
                probs = torch.softmax(logits, dim=-1)
                # print("probs",probs.shape)
            
            #Step2: Perform Beam Search to generate W candidate sequences each of lenght S+1
            candidates = [{"sequence": [], "scores": 0.0}]
            for i in range(self.no_heads):
                new_candidates = []
                for candidate in candidates:
                    log_probs = torch.log(probs[i])
                    top_log_probs, top_tokens = torch.topk(log_probs, self.beam_width)
                    
                    for log_prob, token in zip(top_log_probs, top_tokens):
                        new_candidates.append({
                            "sequence": candidate["sequence"] + [token.item()],
                            "scores": candidate["scores"] + log_prob.item()
                        })
                # print("new_candidates",new_candidates)
                    
                #Keep top W candidates
                candidates = sorted(new_candidates, key=lambda x: x["scores"], reverse=True)[:self.beam_width]

            #Step3: Select the best candidate from these candidate sequences
            best_candidate = []
            best_scores = -float("inf")

            for cand in candidates:
                temp_input = current_input.clone()
                score = 0
                for token in cand["sequence"]:
                    with torch.no_grad():
                        temp_logits = self.model(temp_input).logits[:,-1,:]
                        temp_probs = torch.softmax(temp_logits, dim=-1)
                        score += torch.log(temp_probs[0][token])
                        temp_input = torch.cat([temp_input, torch.tensor([[token]],device=input_ids.device)], dim=1)
                
                if (score > best_scores):
                    best_scores = score
                    best_candidate = cand["sequence"]

            for token in best_candidate:
                if token == self.eos_token_id:
                    generated_tokens.append(token)
                    return torch.tensor(generated_tokens, device=input_ids.device)
                    
                if len(generated_tokens) >= self.max_output_len:
                    return torch.tensor(generated_tokens, device=input_ids.device)
                
                generated_tokens.append(token)
                current_input = torch.cat([current_input,torch.tensor([[token]], device=input_ids.device)], dim=1)
        
        return torch.tensor(generated_tokens, device=input_ids.device)

