# Technical Report: Error Analysis and Remediation

During the code review of [gpt_enfermo.py](src/gpt_enfermo.py), 8 distinct errors were identified in accordance with the project requirements. These consisted of 3 critical architectural flaws and 5 efficiency and quality issues. The following report details the nature of these errors, their impact on the model, and the technical solutions implemented in the refactored [gpt_sano.py](src/gpt_sano.py) module.

---

## Part 1: Critical Errors

These errors fundamentally compromised the mathematical stability of the neural network or made practical training unfeasible.

### 1. Missing Scaling Factor in Self-Attention
* **Location:** [CausalSelfAttention.forward](src/gpt_sano.py#L53), during the Query (Q) and Key (K) matrix multiplication: `att = q @ k.transpose(-2, -1)`.
* **Description:** In the standard scaled dot-product attention formulation, the dot product of Q and K must be scaled by `1 / sqrt(d_k)`. Without this scaling factor, as the dimension of the embedding vectors grows, the variance of the dot product increases significantly. When these large values are passed through the Softmax function, it results in vanishing gradients, preventing the network from learning effectively.
* **Solution:** The manual attention computation was replaced entirely with PyTorch's optimized `F.scaled_dot_product_attention`, which automatically applies the correct mathematical scaling factor and leverages FlashAttention for superior memory efficiency.

### 2. Incorrect Layer Normalization Placement (Post-LN vs. Pre-LN)
* **Location:** [Block.forward](src/gpt_sano.py#L99), within the residual connections: `x = self.ln_1(x + self.attn(x))`.
* **Description:** The original code applied LayerNorm after the residual addition (Post-LN architecture). Modern deep Transformer models, including GPT, utilize a Pre-LN architecture (`x = x + self.attn(self.ln_1(x))`), where normalization is applied to the input of the sub-layer rather than the output. Post-LN is known to cause gradient explosion in early training phases, making deep networks highly unstable without complex learning rate warmup schedules.
* **Solution:** The residual block structure in [Block.forward](src/gpt_sano.py#L99) was refactored to properly implement the Pre-LN architecture.

### 3. Inefficient Cross-Entropy Loss Computation
* **Location:** [GPT.forward](src/gpt_sano.py#L145), during the `loss` calculation when `targets` are provided.
* **Description:** The original implementation utilized a nested loop in Python (`for b in range(B): for t in range(T):`) to iterate over each item in the batch and sequence, calculating the cross-entropy loss per token and aggregating it. Performing element-wise tensor operations in pure Python bypasses PyTorch's optimized C++/CUDA backend, resulting in a severe bottleneck that slows down training by orders of magnitude, rendering the training loop practically unusable.
* **Solution:** The nested loops were removed. The `logits` and `targets` tensors were flattened and processed via vectorized operations: `loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))`.

---

## Part 2: Quality and Efficiency Errors

These errors do not necessarily prevent the model from compiling or converging, but they represent suboptimal practices, inefficient memory utilization, and deviations from modern GPT architectural standards.

### 1. Improper Optimizer Weight Decay Configuration
* **Location:** [GPT.configure_optimizers](src/gpt_sano.py#L187).
* **Description:** The original script applied weight decay globally to all parameters in the model. This included bias terms, embedding matrices, and LayerNorm weights. Standard deep learning practice dictates that weight decay should only be applied to multi-dimensional weight matrices (like linear layer weights) to prevent unnecessary regularization of normalization layers and biases, which can degrade model capacity.
* **Solution:** The parameters were segregated into two groups (`decay_params` and `nodecay_params`) based on their tensor dimensions before being passed to the AdamW optimizer.

### 2. Incorrect Hidden Dimension Multiplier in the MLP Block
* **Location:** [MLP.__init__](src/gpt_sano.py#L73), specifically the `fc1` projection layer.
* **Description:** The internal projection layer of the FeedForward block was set to project to `2 * n_embd`. The standard Transformer and GPT architecture requires the hidden dimension of the MLP to be exactly 4 times the embedding dimension (`4 * n_embd`). Using a smaller multiplier unnecessarily constrains the expressivity and capacity of the network.
* **Solution:** The multiplier was corrected to project from `n_embd` to `4 * n_embd`, and subsequently back to `n_embd`.

### 3. Suboptimal Activation Function (ReLU vs. GELU)
* **Location:** [MLP.__init__](src/gpt_sano.py#L73).
* **Description:** The original implementation utilized the standard ReLU activation function. While functional, modern language models (including GPT-2 and GPT-3) utilize the GELU (Gaussian Error Linear Unit) activation function. GELU provides a smoother non-linearity, which typically results in better gradient propagation and overall model performance in Natural Language Processing tasks.
* **Solution:** The `nn.ReLU()` activation was replaced with `nn.GELU()`.

### 4. Absence of Weight Tying
* **Location:** [GPT.__init__](src/gpt_sano.py#L108), specifically between `self.transformer.wte` and `self.lm_head`.
* **Description:** A standard architectural optimization in language modeling is to tie (share) the weights of the initial token embedding layer (`wte`) with the final linear projection layer (`lm_head`) that outputs the vocabulary logits. Failing to tie these weights drastically increases the model's parameter count and memory footprint without yielding proportional performance gains.
* **Solution:** Weight tying was implemented by adding the assignment `self.lm_head.weight = self.transformer.wte.weight` during initialization.

### 5. Unoptimized Manual Attention Implementation
* **Location:** [CausalSelfAttention.forward](src/gpt_sano.py#L53).
* **Description:** The model split its projections into three separate linear layers for Query, Key, and Value, followed by manual matrix multiplications, a lower-triangular mask application, softmax, and dropout. This manual sequence of operations is highly inefficient on modern GPUs.
* **Solution:** The projection layers were consolidated into a single linear layer (`c_attn`) for Q, K, and V. The manual attention calculation was entirely replaced with `F.scaled_dot_product_attention`, enabling PyTorch's FlashAttention backend. Additionally, the residual projection weights were properly scaled by `1 / sqrt(2 * n_layer)` during initialization, a crucial optimization missing from the original code.

---

## Remediation Plan Implementation

Rather than destructively modifying the original flawed scripts, the following approach was taken to resolve the project requirements:

1. **[gpt_sano.py](src/gpt_sano.py)**: A new, stable module was created containing the refactored GPT architecture with all 8 errors resolved.
2. **[prepare.py](data/script/prepare.py)**: A data processing script was written to parse [tinyshakespeare.txt](data/raw/tinyshakespeare.txt), construct a character-level vocabulary, and output the required binary arrays ([train.bin](data/train.bin) and [val.bin](data/val.bin)).
3. **[recuperacion_sana.py](src/recuperacion_sana.py)**: The training loop was rewritten to properly ingest the generated Tiny Shakespeare binaries and import the optimized model from [gpt_sano.py](src/gpt_sano.py), allowing for efficient and stable monitoring of the model's recovery.
