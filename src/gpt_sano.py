"""
This module contains the corrected and optimized implementation of a GPT-style model.
It includes the necessary components like multi-head self-attention, MLP blocks,
and the main GPT model class, incorporating modern best practices such as Pre-LN
and Flash Attention.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    """Configuration class for the GPT model."""
    block_size: int = 1024  # Maximum sequence length
    vocab_size: int = 50257 # Vocabulary size
    n_layer: int = 12      # Number of transformer blocks
    n_head: int = 12       # Number of attention heads
    n_embd: int = 768      # Embedding dimension
    dropout: float = 0.1   # Dropout rate
    bias: bool = True      # Whether to use bias in linear layers

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class LayerNorm(nn.Module):
    """A custom LayerNorm module."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        """Forward pass for LayerNorm."""
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Implements a causal self-attention mechanism."""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Quality fix: Combine Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        """
        Forward pass for the Causal Self-Attention block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        B, T, C = x.size()

        # Quality fix: using single projection and scaled dot product attention (Flash Attention)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Critical fix: Added scaling 1.0 / math.sqrt(k.size(-1)) and used Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron block."""
    def __init__(self, config):
        super().__init__()
        # Quality fix: Hidden dimension in MLP is 4 * n_embd
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Quality fix: Use GELU activation function instead of ReLU
        self.act    = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Forward pass for the MLP."""
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """A single Transformer block, combining self-attention and an MLP."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        """Forward pass for the Transformer block."""
        # Critical fix: Use Pre-LN (LayerNorm before attention and MLP) instead of Post-LN
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The main GPT model class."""
    
    def __init__(self, config):
        """
        Initializes the GPT model with the given configuration.
        """
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Quality fix: Weight tying between token embeddings and the LM head
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        
        # scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self):
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        """Initializes the weights of the model's modules."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass for the GPT model.

        Args:
            idx (torch.Tensor): Input tensor of token indices.
            targets (torch.Tensor, optional): Target token indices for loss calculation.

        Returns:
            tuple: A tuple containing logits and the loss (if targets are provided).
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # Critical fix: Vectorized Cross-Entropy instead of nested Python loops
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates a sequence of tokens starting from the given context.

        Args:
            idx (torch.Tensor): The initial context (token indices).
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): Softmax temperature for sampling.
            top_k (int, optional): If specified, samples from the top k most likely tokens.

        Returns:
            torch.Tensor: The generated sequence of token indices.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                       else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configures the AdamW optimizer with weight decay separation.

        Args:
            weight_decay (float): The weight decay value.
            learning_rate (float): The learning rate.
            betas (tuple): AdamW betas.
            device_type (str): The device type ('cuda' or 'cpu').

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        # Quality fix: Separate parameters for weight decay
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
