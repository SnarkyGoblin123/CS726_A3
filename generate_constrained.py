import torch
import torch.nn as nn
import warnings
from jaxtyping import Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from collections import defaultdict

warnings.filterwarnings("ignore")

# class Trie:
#     """Trie data structure to store tokenized words."""
#     def __init__(self):
#         self.root = {}
#         self.end_token = "__END__"

#     def insert(self, tokenized_word):
#         """Insert a tokenized word into the Trie."""
#         node = self.root
#         for token in tokenized_word:
#             if token not in node:
#                 node[token] = {}
#             node = node[token]
#         node[self.end_token] = True  # Mark end of a word

#     def get_valid_next_tokens(self, prefix_tokens):
#         """Return valid next tokens given the current sequence of tokens."""
#         node = self.root
#         for token in prefix_tokens:
#             if token in node:
#                 node = node[token]
#             else:
#                 return set()  # No valid continuations
#         return set(node.keys()) - {self.end_token}  # Return all valid continuations

# class ConstrainedTextGenerator:
#     def __init__(
#         self, 
#         model: AutoModelForCausalLM, 
#         tokenizer: AutoTokenizer, 
#         eos_id: int, 
#         max_output_len: int = 10,
#     ) -> None:
#         '''
#             Initialize the ConstrainedTextGenerator class.
            
#             model: LLM
#             tokenizer: LLM's tokenizer.
#             eos_id: End-of-sequence token id 
#             max_output_len: Maximum number of tokens to be generated.
            
#             Do not edit.
#         '''
#         self.model = model
#         self.max_output_len = max_output_len
#         self.eos_token_id = eos_id
#         self.tokenizer = tokenizer

#     def build_trie(self, word_list: List[str]) -> Trie:
#         """Build a Trie from a list of words."""
#         trie = Trie()
#         for word in word_list:
#             tokenized_word = self.tokenizer.encode(word, add_special_tokens=False)
#             trie.insert(tokenized_word)
#         return trie

#     def __call__(
#         self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
#     ) -> Int[torch.Tensor, "batch out_seq_len"]:
#         '''
#             Implement Word-Constrained decoding technique.
#         '''   
#         trie = self.build_trie(word_list)  # Build Trie for constraints
#         generated_tokens = []
#         curr_inp = input_ids
        
#         for _ in range(self.max_output_len):
#             with torch.no_grad():
#                 outputs = self.model(curr_inp)
#                 logits = outputs.logits[:, -1, :]

#                 probs = torch.softmax(logits, dim=-1)
#                 top_tokens = torch.argsort(probs, descending=True)[0]  # Rank tokens

#                 # Check which tokens are allowed based on Trie
#                 valid_next_tokens = trie.get_valid_next_tokens(generated_tokens)
                
#                 # Pick the most probable valid token
#                 next_token = None
#                 for token in top_tokens:
#                     if token.item() in valid_next_tokens:
#                         next_token = token
#                         break

#                 # If no valid token, fall back to greedy decoding
#                 if next_token is None:
#                     next_token = torch.argmax(probs, dim=-1)

#                 if next_token.item() == self.eos_token_id:
#                     break

#                 generated_tokens.append(next_token.item())
#                 curr_inp = torch.cat([curr_inp, next_token.view(1, 1)], dim=1)
#         return torch.tensor(generated_tokens, device=input_ids.device).squeeze(0)

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)# token_id -> TrieNode
        self.is_word_end = False

class ConstrainedTextGenerator:
    def __init__(self, model, tokenizer, eos_id, max_output_len):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_id
        self.max_output_len = max_output_len

    def _build_trie(self, word_list):
        trie = TrieNode()
        for word in word_list:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            node = trie
            for token in tokens:
                if token not in node.children:
                    node.children[token] = TrieNode()
                node = node.children[token]
            node.is_word_end = True
        return trie

    def __call__(self, input_ids, word_list):
        trie = self._build_trie(word_list)
        generated_tokens = []
        current_input = input_ids.clone()
        current_nodes = [trie]  # Tracks all possible Trie paths
        
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)
                logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            
            # Mask invalid tokens (not in any current Trie node's children)
            valid_mask = torch.zeros_like(logits, dtype=torch.bool)
            for node in current_nodes:
                for token_id in node.children:
                    valid_mask[0, token_id] = True
            
            # If no valid tokens, break early
            if not valid_mask.any():
                break
            
            # Apply mask and select highest-probability valid token
            masked_logits = logits.masked_fill(~valid_mask, float('-inf'))
            # next_token = torch.argmax(masked_logits, dim=-1)
            repetition_penalty = 1.5  # Adjust based on tuning
            for token in generated_tokens:
                masked_logits[0, token] /= repetition_penalty

            next_token = torch.argmax(masked_logits, dim=-1)
                        
            if next_token.item() == self.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=-1)
            
            new_nodes = []
            for node in current_nodes:
                if next_token.item() in node.children:
                    child_node = node.children[next_token.item()]
                    new_nodes.append(child_node)  # Continue path
                    if child_node.is_word_end:
                        new_nodes.append(trie)  
            
            # If no valid path, force reset (optional)
            current_nodes = new_nodes if new_nodes else [trie]
        
        return torch.tensor(generated_tokens, device=input_ids.device)