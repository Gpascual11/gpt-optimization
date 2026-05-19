# GPT Optimization and Debugging Project

This project focuses on the structural analysis, debugging, and optimization of a custom Transformer-based language model (GPT). The objective is to identify and resolve both critical architectural flaws and suboptimal implementations within a provided codebase, subsequently training the corrected model on the Tiny Shakespeare dataset to evaluate its performance and stability.

## Project Context and Provided Files

The repository initially provided two core scripts that served as the baseline for this project:

- **`gpt_enfermo.py`**: A foundational but flawed implementation of a GPT model. This file contained 8 distinct bugs: 3 critical errors that caused mathematical instability or completely hindered the training process, and 5 quality/efficiency errors that degraded performance, memory usage, or deviated from standard deep learning practices.
- **`recuperacion.py`**: The original training loop script. It was designed to train the model, but it referenced an incorrect dataset path and imported the flawed GPT model without proper optimization settings.

The goal of the project was to document these flaws, implement robust solutions, and achieve a successful training cycle.

## Project Structure

The current optimized repository is organized as follows:

### `data/`
- **`tinyshakespeare.txt`**: The raw text dataset containing all of Shakespeare's works.
- **`prepare.py`**: A script developed to process the raw text, build a character-level vocabulary, and encode the data into binary memory-mapped files (`train.bin` and `val.bin`) for efficient data loading during training.

### `src/`
- **`gpt_enfermo.py`**: The original flawed GPT model implementation (preserved for reference).
- **`gpt_sano.py`**: The corrected and optimized GPT model. This file implements the architectural fixes, including proper attention scaling, Pre-LayerNorm configuration, vectorized loss computation, FlashAttention integration, and correct weight initialization.
- **`recuperacion.py`**: The original training script (preserved for reference).
- **`recuperacion_sana.py`**: The modernized training script, updated to correctly import the optimized model and properly interface with the generated Tiny Shakespeare binary data.

### Documentation
- **`README.md`**: Project overview and structure.
- **`EXPLANATION.md`**: A detailed technical report outlining the bugs identified in the original codebase, their implications on model training, and the methodological solutions applied.

## Running the Project

The project is configured to run inside a Python virtual environment managed by `uv`.

1. **Set up the Environment**:
   Initialize the virtual environment and install the required dependencies (PyTorch and NumPy).
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install torch numpy
   ```

2. **Prepare the Dataset**:
   Navigate to the project root and execute the data preparation script to generate the training and validation splits.
   ```bash
   python data/prepare.py
   ```

3. **Train the Model**:
   Execute the training script to begin the model's recovery process.
   ```bash
   python src/recuperacion_sana.py
   ```
   Checkpoints and logs will be saved in the generated `out/` directory.
