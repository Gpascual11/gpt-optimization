import os
import pickle
import torch
from gpt_sano import GPTConfig, GPT

# Load meta to get vocabulary and decoding mapping
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

# Load checkpoint
ckpt_path = 'out/ckpt.pt'
checkpoint = torch.load(ckpt_path, map_location='cpu')

# Re-create model
model_args = checkpoint['model_args']
config = GPTConfig(**model_args)
model = GPT(config)

# Remove '_orig_mod.' prefix if model was compiled
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()

# Generate text
start_text = "\n"
stoi = meta['stoi']
encode = lambda s: [stoi[c] for c in s]
x = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0)

# Generate 500 characters
print("Generating text from checkpoint...")
print("=" * 40)
generated_indices = model.generate(x, max_new_tokens=500, temperature=0.8, top_k=200)[0].tolist()
print(decode(generated_indices))
print("=" * 40)
