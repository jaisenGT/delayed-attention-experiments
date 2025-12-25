# train_tinystories_delayed.py

out_dir = 'out-tinystories-delayed'
eval_interval = 500
eval_iters = 200
log_interval = 10

# save checkpoints so we can generate stories later to check coherence
always_save_checkpoint = True

wandb_log = True # highly recommended to visualize the loss divergence
wandb_project = 'tinystories-delayed'
wandb_run_name = 'delayed-30m'

dataset = 'tinystories'
gradient_accumulation_steps = 2 # simulate larger batch size
batch_size = 64
block_size = 512 # 256 is too short for a story. 512 gives E2 room to work.

# Model: ~28M Parameters
# (Standard TinyStories sizes: 1M, 3M, 33M. We target the 33M tier)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1 # TinyStories is large enough that we don't need heavy dropout

# Delayed Attention Parameters
# We put delayed attention in the MIDDLE layers.
# Bottom layers (0,1) establish basic grammar (unidirectional is fine).
# Middle layers (2,3,4,5) do the heavy reasoning with future context.
delayed_layers = [2, 3, 4, 5] 
lookahead = 30 # CRITICAL: Needs to be large enough to be useful (30 tokens)
overlap = 10
lora_rank = 16 # Increased rank slightly to give E2 more capacity
lora_alpha = 32

# Optimization
learning_rate = 5e-4 # Lower LR for larger model
max_iters = 10000 # TinyStories needs more iters than shakespeare
lr_decay_iters = 10000
min_lr = 5e-5 
beta2 = 0.99
warmup_iters = 200

weight_decay = 1e-1

device = 'cuda' # or 'cpu' / 'mps'
compile = True # definitely use compile for this size if possible