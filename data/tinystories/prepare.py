"""
Prepare the TinyStories dataset for character-level or GPT-2 token-level training.
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    # 1. Download the dataset
    # We use the "TinyStories" dataset from huggingface
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset)

    # 2. Tokenize
    # We use the GPT-2 tokenizer (same as standard nanoGPT)
    enc = tiktoken.get_encoding("gpt2")

    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though.
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # 3. Concatenate and write to bin files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        print(f"writing {filename}...")
        idx = 0
        for example in tqdm(dset):
            arr[idx : idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()

    # save the meta information as well, to help us encode/decode later
    import pickle
    meta = {
        'vocab_size': 50257,
        'stoi': {}, # not needed for gpt-2 BPE
        'itos': {}, 
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)