# Ablation study configuration for enwik8 character-level language modeling
# This disables routing to compare against the routing model

out_dir = 'out-enwik8-routing-ablation'
eval_interval = 1000
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'enwik8-char'
wandb_run_name = 'routing-ablation'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 1024 # context of up to 1024 previous characters

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# Routing-specific parameters (disabled for ablation)
use_routing = False  # Disable routing mechanism
top_k_heads = None
entropy_reg_coef = 0.0

learning_rate = 6e-4
max_iters = 200000
lr_decay_iters = 200000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.95

warmup_iters = 1000

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model 