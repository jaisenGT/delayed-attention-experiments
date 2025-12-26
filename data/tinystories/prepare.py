import os
import tqdm
import tiktoken
import numpy as np
from datasets import load_dataset # pip install datasets

# Number of workers for .map() calls
# Good number to use is ~order number of cpu cores // 2
num_proc = 8

# Number of workers for loading the dataset
# generally doesn't need to be high
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    # 1. Download the TinyStories dataset from Hugging Face
    # We use the 'roneneldan/TinyStories' dataset which is the standard source
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset)

    # 2. Get the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>'] # 50256

    # 3. Define the encoding function
    def process(example):
        # We encode the text and append the <|endoftext|> token (50256) 
        # so the model learns to stop or start a new story.
        ids = enc.encode_ordinary(example['text'])
        ids.append(eot)
        return {'ids': ids, 'len': len(ids)}

    # 4. Tokenize the dataset
    # The dataset has 'train' and 'validation' splits by default
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # 5. Concatenate all ids and save to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        
        print(f"writing {filename}...")
        # We create a memory-mapped array to write directly to disk to save RAM
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        for example in tqdm.tqdm(dset, desc=f"writing {split}"):
            arr[idx : idx + example['len']] = example['ids']
            idx += example['len']
        
        arr.flush()
        print(f"{split} has {arr_len:,} tokens")

    # Expected output size (approximate):
    # train.bin: ~900MB - 1.5GB depending on version
    # val.bin: ~10MB