## Programming Assignment 3: LLM sampling and decoding techniques

### Setting up the codebase [very crucial]
1. Getting access to Llama weights: https://www.llama.com/docs/getting-the-models/hugging-face/ (Get access to `meta-llama/Llama-2-7b-hf`)
2. Getting access to IN22 dataset: Sign in to your huggingface account and accept the agreement here: https://huggingface.co/datasets/ai4bharat/IN22-Gen
3. Install dependencies: The codebase has a `environment.yml` file, which you can use to create a new envrinoment as follows:
```bash 
conda env create -f environment.yml
```
4. Install Medusa: Next, you need Medusa's codebase which can be installed as follows:
```bash
conda activate cs726_a3
git clone https://github.com/Darshan7575/Medusa.git
cd Medusa
pip install -e .
```
[Only for gpu1.cse users] Fall back to slightly older cuda versions:
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
``` 
4. Creating Huggingface access token: Finally, you need to create huggingface token to access the model weights and dataset. Follow these instructions: https://huggingface.co/docs/hub/en/security-tokens


### How to run?
#### Task 0

1. **Greedy Decoding**: To run the code on a specific GPU, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "greedy"
```
To run the code on CPU, use the following command (*please note that this will be extremely slow*):
```bash
CUDA_VISIBLE_DEVICES=-1 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "greedy"
```
Additionally, if you want to check the input, reference text and your predicted outputs, you can use:
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "greedy" --debug true
```

2. **Random Sampling**: Follow the same steps as above, but with additional arguments
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "random" --tau <tau value>
```

3. **Top-k Sampling**: Follow the same steps as above, but with additional arguments
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "topk" --k <k value>
```

4. **Nucleus Sampling**: Follow the same steps as above, but with additional arguments
```bash
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "<your_hf_token>" --decoding-strategy "nucleus" --p <p value>
```

#### Task 1
Similar to the previous task, you can run the script as follows:
```bash
CUDA_VISIBLE_DEVICES=0 python task1.py --hf-token "<your_hf_token>" --word_list <path to the word_lists.txt file>
```

#### Task 2
Similar to previous task, you can run the script as follows:
1. **Single head decoding**
```bash 
CUDA_VISIBLE_DEVICES=0 python task2.py --hf-token "<your_hf_token>" --decoding-strategy "single-head"
```

2. **Multiple head decoding**
```bash
CUDA_VISIBLE_DEVICES=0 python task2.py --hf-token "<your_hf_token>" --decoding-strategy "multi-head" --beam-width <beam width> --use-no-medusa-heads <no of medusa heads to be used>
```

---

### Code Documentation

#### [generate.py](generate.py)
This file implements various decoding strategies for text generation using a language model.

- **Classes**:
  - `TextGenerator`: A class that supports multiple decoding strategies, including:
    - **Greedy Decoding**: Selects the token with the highest probability at each step.
    - **Random Sampling**: Samples tokens based on their probabilities, scaled by a temperature parameter.
    - **Top-k Sampling**: Samples tokens from the top-k most probable tokens.
    - **Nucleus Sampling**: Samples tokens from the smallest set of tokens whose cumulative probability exceeds a threshold.

- **Key Parameters**:
  - `decoding_strategy`: Specifies the decoding method to use.
  - `eos_id`: End-of-sequence token ID.
  - `max_output_len`: Maximum number of tokens to generate.
  - `tau`, `k`, `p`: Parameters for random, top-k, and nucleus sampling, respectively.

#### [generate_constrained.py](generate_constrained.py)
This file implements word-constrained decoding using a Trie data structure.

- **Classes**:
  - `TrieNode`: Represents a node in the Trie, used to store tokenized words.
  - `ConstrainedTextGenerator`: A class that generates text while ensuring the output adheres to a predefined list of words.

- **Key Features**:
  - Builds a Trie from a list of words to enforce constraints during decoding.
  - Masks invalid tokens at each step to ensure only valid continuations are generated.
  - Supports repetition penalties to discourage repeated tokens.

#### [generate_medusa.py](generate_medusa.py)
This file implements decoding strategies for the Medusa model, which uses multiple heads for text generation.

- **Classes**:
  - `MedusaTextGenerator`: A class that supports:
    - **Single-head Decoding**: Uses only the language model (LM) head for decoding.
    - **Multi-head Decoding**: Combines the LM head and multiple Medusa heads for decoding, using beam search to find the best sequence.

- **Key Parameters**:
  - `decoding_strategy`: Specifies whether to use single-head or multi-head decoding.
  - `use_no_medusa_heads`: Number of Medusa heads to use (up to 5).
  - `beam_width`: Number of candidates to maintain during beam search.
  - `max_output_len`: Maximum number of tokens to generate.

- **Key Features**:
  - Implements beam search to explore multiple candidate sequences.
  - Combines probabilities from multiple heads to improve decoding quality.