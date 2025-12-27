# train_tinystories_standard.py

out_dir = 'out-tinystories-standard'
eval_interval = 200
eval_iters = 100
log_interval = 10

# save checkpoints so we can generate stories later to check coherence
always_save_checkpoint = True

wandb_log = True
wandb_project = 'tinystories'
wandb_run_name = 'standard-25m'

dataset = 'tinystories'
gradient_accumulation_steps = 2 # simulate larger batch size
batch_size = 64
block_size = 512 # 512 gives E2 room to work.

# Model: ~25M Parameters
# (Standard TinyStories sizes: 1M, 3M, 33M.)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1 # TinyStories is large enough that we don't need heavy dropout

# Delayed Attention Parameters
delayed_layers = []

# Optimization
learning_rate = 5e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 5e-5 
beta2 = 0.99
warmup_iters = 200

weight_decay = 1e-1

device = 'cuda' # or 'cpu' / 'mps'
compile = True # good for A100 GPUs