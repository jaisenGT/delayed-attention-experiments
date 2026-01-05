# Delayed Attention for LLMs

**Author: Jaisen Soundar**

Current LLMs are trained to predict the next token using only the preceding ones. This unidirectionality is important since, for training, it allows for a single, parallelizable pass over a sequence to calculate loss for all the token predictions (O(n^2) complexity). For inference, it enables the use of a KV cache, making the generation of each new token efficient (O(n) complexity). However, unlike with bidirectional models (e.g., BERT), a token has no information about the text that will follow it. My hypothesis is that introducing a degree of bidirectionality, while keeping the efficiencies listed above, can improve an LLM's coherence and reasoning.


## Architecture:

The core idea of Delayed Attention is to have two parallel embedding streams for each token throughout the model: E1 and E2.

**E1 (Standard Embedding):** This stream behaves like a standard causal decoder. It performs attention with itself and all previous tokens. Specifically, an E1 embedding at position i attends to the E1 embeddings of itself and the last 29 tokens and the E2 embeddings of all tokens before that.

**E2 (Delayed Embedding):** This stream looks back at all previous tokens and looks ahead by a fixed window of 30 tokens. More specifically, an E2 embedding at position i attends to the E1 embeddings of the next 30 tokens and the E2 embeddings of all tokens up to position i.

During inference, the E2 embedding for a token at position i is only computed after i+30 tokens have been processed. This "delay" is because this E2 token needs tokens i to i+30 (in addition to all previous tokens) to perform attention with. The final prediction for the next token is always made by the E1 embedding in the last layer of the very last token in the sequence.

**Technical Details:** The compute cost of this architecture is ~2x compared to the standard architecture. However, I incorporate a hybrid architecture that uses both standard unidirectional blocks as well as delayed blocks to lower this cost. To differentiate the parameters between the E1 and E2 streams, I apply LoRA to the MLP and attention projection matrices. I incorporate learned embedding-type encodings for the model to further differentiate between the two streams. I also incorporate a small overlap region, in which attention is performed with both E1 and E2, for the model to learn a smooth transition between the two.


## Results and Conclusion:

Initial experiments with training ~50m parameter models on the TinyStories dataset yielded negative results. The delayed model performed on par with a standard unidirectional model of the same parameter count that is trained on the same data. However, the delayed model used ~1.5x the training compute of the standard model. 

Further research can be done to scale up delayed attention models to see if their bidirectional capabilities grow with larger model sizes and to test this architecture on other datasets in which bidirectionality might be especially effective (e.g. code).


## Usage
To reproduce these results or experiment with the Delayed Attention architecture:

```
# Install dependencies
pip install torch numpy transformers datasets tiktoken wandb tqdm

# Prepare the TinyStories dataset
python data/tinystories/prepare.py

# Train the Delayed Attention model
python train.py --config=config/train_tinystories_delayed.py
```


**Acknowledgements:** Andrej Karpathy for the [nanoGPT](https://github.com/karpathy/nanoGPT) codebase and TinyStories for the primary dataset used in these experiments.
