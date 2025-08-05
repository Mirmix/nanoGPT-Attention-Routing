# Configuration for enwik8 character-level language modeling with learnable attention head routing
# Target: ~44M parameters (same as baseline) + routing overhead

out_dir = 'out-enwik8-routing'
eval_interval = 1000
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'enwik8-char'
wandb_run_name = 'routing'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 1024 # context of up to 1024 previous characters

# Model configuration with routing (~44M params + routing overhead)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# Routing-specific parameters
use_routing = True
top_k_heads = 6  # Use top-6 heads per token (half of total heads for efficiency)
entropy_reg_coef = 0.01  # Entropy regularization to encourage head specialization

learning_rate = 6e-4
max_iters = 200000
lr_decay_iters = 200000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.95

warmup_iters = 1000

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model 