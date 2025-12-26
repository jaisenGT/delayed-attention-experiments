import os
import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets

# --- Configuration ---
num_proc = 8 
shard_size = 10000 
seed = 42 

if __name__ == '__main__':
    # 1. Load Dataset
    print("Loading dataset...")
    # "roneneldan/TinyStories" is the original. 
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc)

    # 2. Shuffle (Crucial for good training dynamics)
    print("Shuffling dataset...")
    dataset = dataset.shuffle(seed=seed)

    # 3. Setup Tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>'] # 50256

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(eot)
        return {'ids': ids, 'len': len(ids)}

    # 4. Tokenize (Parallel)
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing splits",
        num_proc=num_proc,
    )

    # 5. Save to Disk (Batched)
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        
        # Rename 'validation' -> 'val.bin' for nanoGPT compatibility
        filename = f'{split}.bin'
        if split == 'validation':
            filename = 'val.bin'
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # uint16 is sufficient for GPT-2 vocab (50257)
        dtype = np.uint16 
        
        print(f"Writing {filepath}...")
        arr = np.memmap(filepath, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        total_batches = (len(dset) + shard_size - 1) // shard_size
        
        for batch_idx in tqdm.tqdm(range(total_batches), desc=f"Writing {filename}"):
            start = batch_idx * shard_size
            end = min(start + shard_size, len(dset))
            
            batch = dset[start:end]
            
            # Fast numpy concatenation
            batch_ids = np.concatenate(batch['ids'])
            
            arr[idx : idx + len(batch_ids)] = batch_ids
            idx += len(batch_ids)
        
        arr.flush()
        print(f"Saved {filename}: {arr_len / 1e6:.2f}M tokens")

    print("Done.")