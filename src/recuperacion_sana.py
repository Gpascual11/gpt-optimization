"""
This script handles the training of the corrected GPT model (`gpt_sano.py`)
on the Tiny Shakespeare dataset. It includes data loading, the main training loop,
learning rate scheduling, loss estimation, and checkpointing with early stopping.
"""
import os
import time
import math
import pickle
import numpy as np
import torch
from contextlib import nullcontext
from gpt_sano import GPTConfig, GPT

out_dir        = 'out'
eval_interval  = 250
log_interval   = 10
eval_iters     = 200
eval_only      = False

always_save_checkpoint = False

dataset        = 'tinyshakespeare' 
batch_size     = 64
block_size     = 256

n_layer        = 6
n_head         = 6
n_embd         = 384
dropout        = 0.2
bias           = False

learning_rate  = 1e-3
max_iters      = 5000
weight_decay   = 1e-1
beta1          = 0.9
beta2          = 0.99
grad_clip      = 1.0

decay_lr       = True
warmup_iters   = 100
lr_decay_iters = 5000
min_lr         = 1e-4
patience       = 5

device         = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype          = 'float16'
compile        = True

# -------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype) \
      if device == 'cuda' else nullcontext()

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data   = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16, mode='r')

def get_batch(split):
    """
    Generates a random batch of data for training or validation.

    Args:
        split (str): The data split to use ('train' or 'val').

    Returns:
        tuple: A tuple containing input tensors (x) and target tensors (y).
    """
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
    y    = torch.stack([torch.from_numpy(data[i+1:i+1+block_size]) for i in ix])
    x, y = x.to(device, dtype=torch.long), y.to(device, dtype=torch.long)
    return x, y

with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout,
)
config = GPTConfig(**model_args)
model  = GPT(config)
model  = model.to(device)

if compile:
    model = torch.compile(model)

optimizer = model.configure_optimizers(weight_decay, learning_rate,
                                       (beta1, beta2), device)
scaler = torch.amp.GradScaler(device, enabled=(dtype == 'float16'))


@torch.no_grad()
def estimate_loss():
    """
    Estimates the average loss for both training and validation splits.

    Returns:
        dict: A dictionary containing the mean loss for 'train' and 'val' splits.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    """
    Calculates the learning rate for a given iteration using a cosine decay schedule
    with warmup.

    Args:
        it (int): The current training iteration.

    Returns:
        float: The calculated learning rate.
    """
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


best_val_loss = float('inf')
patience_counter = 0
X, Y = get_batch('train')
t0 = time.time()

for iter_num in range(max_iters + 1):

    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            if iter_num > 0: # Do not save at step 0
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"New best validation loss: {best_val_loss:.4f}. Saving checkpoint to {out_dir}/ckpt.pt")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at step {iter_num}. Validation loss hasn't improved for {patience * eval_interval} steps.")
            break

    if eval_only:
        break

    with ctx:
        logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    X, Y = get_batch('train')

    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.0f}ms")
