# Steering Vector Experiments for Coconut

This repository contains code for extracting steering vectors from Coconut (Chain-of-Thought with Continuous Thinking) language model and analyzing their effects on reasoning behavior.

## Overview

The implementation supports two contrastive modes for steering vector extraction:

- **`swap`** (Phase 1): Swaps answer options within the same question
- **`cross_pair`** (Phase 2): Pairs different samples with different answers (stronger signal)

## Prerequisites

```bash
# Activate conda environment
conda activate coconut

# Install dependencies (if not already present)
pip install torch transformers sklearn pandas matplotlib numpy pyyaml
```

## Getting Started with Coconut

The first step is to clone the Coconut repository from https://github.com/facebookresearch/coconut and train the model on the ProsQA dataset using their instructions, configuring YAML files as needed. Use the best model (highest test/validation accuracy) for all steering experiments described above.

Make sure to update the `model_path` in `steering/config/steering_config.yaml` to point to your trained checkpoint path (e.g., `~/coconut-cot/checkpoints/prosqa-coconut/checkpoint_48` or your best performing checkpoint).

## Configuration

Edit `steering/config/steering_config.yaml` to set:

```yaml
model_path: ~/coconut-cot/checkpoints/prosqa-coconut/checkpoint_48  # Model checkpoint
train_path: ~/coconut-cot/coconut/data/prosqa_train.json            # Training data
val_path:   ~/coconut-cot/coconut/data/prosqa_valid.json            # Validation data
output_dir: ~/coconut-cot/steering/outputs                          # Output directory

n_latent: 6                         # Number of latent tokens per sample
latent_positions: [1, 2, 3, 4, 5]   # Positions to extract vectors from (1-indexed)
contrastive_mode: cross_pair        # 'swap' or 'cross_pair'
n_contrast: 500                     # Number of contrastive pairs
alpha_sweep: [-1200, -600, -300, -100, 100, 300, 600, 1200]  # Steering strengths
n_probe: 400                        # Samples for answer-decodability probe
```

## Running the Pipeline

### Step 1: Extract Steering Vectors

```bash
cd ~/coconut-cot

# Extract vectors using cross_pair mode (recommended)
python steering/extract_vectors.py --config steering/config/steering_config.yaml

# Extract vectors using swap mode
python steering/extract_vectors.py --config steering/config/steering_config.yaml --mode swap

# Only run the probe (skip vector extraction)
python steering/extract_vectors.py --config steering/config/steering_config.yaml --probe-only
```

This generates:
- `steering_vectors_{mode}.pt` - Steering vectors for each latent position
- `probe_thoughts_L{pos}.npy` - Latent representations for probe analysis
- `probe_results.csv` - Answer decodability accuracy per position

### Step 2: Run Behavioral Evaluation

```bash
# Evaluate cross_pair mode
python steering/run_evaluation.py --config steering/config/steering_config.yaml --mode cross_pair

# Evaluate swap mode
python steering/run_evaluation.py --config steering/config/steering_config.yaml --mode swap

# Evaluate both modes
python steering/run_evaluation.py --config steering/config/steering_config.yaml --mode both

# Limit samples for quick testing
python steering/run_evaluation.py --config steering/config/steering_config.yaml --max-samples 50
```

This generates:
- `results_{mode}.csv` - Detailed evaluation results with accuracy and logit margins

### Step 3: Visualize Results

```bash
python steering/analyze_results.py --config steering/config/steering_config.yaml
```

This generates the following plots in `steering/outputs/figures/`:

| Plot | Description |
|------|-------------|
| `probe_accuracy.png` | Answer decodability accuracy across latent positions |
| `dose_response_L{pos}.png` | Accuracy delta and logit margin delta vs. alpha |
| `sign_asymmetry.png` | Comparison of positive vs. negative steering effects |
| `mode_comparison_pca_L{pos}.png` | PCA visualization of latent representations |

## Expected Output Structure

```
steering/outputs/
├── figures/
│   ├── probe_accuracy.png
│   ├── dose_response_L*.png
│   ├── ...
│   ├── sign_asymmetry.png
│   └── mode_comparison_pca_L*.png
├── probe_results.csv
├── probe_thoughts_L*.npy
├── ...
├── results_cross_pair.csv
├── results_swap.csv
├── steering_vectors_cross_pair.pt
└── steering_vectors_swap.pt
```

## Full Pipeline (One-liner)

```bash
cd ~/coconut-cot && \
python steering/extract_vectors.py --config steering/config/steering_config.yaml --mode cross_pair && \
python steering/run_evaluation.py --config steering/config/steering_config.yaml --mode cross_pair && \
python steering/analyze_results.py --config steering/config/steering_config.yaml
```

## File Descriptions

| File | Purpose |
|------|---------|
| `steering/extract_vectors.py` | Main script for extracting steering vectors and running probe |
| `steering/run_evaluation.py` | Evaluates steering effects on model behavior |
| `steering/analyze_results.py` | Generates all visualization plots |
| `steering/core/model_utils.py` | Model loading, tokenization, thought vector extraction |
| `steering/core/contrastive_pairs.py` | Builds contrastive pairs for vector extraction |
| `steering/core/steering_injection.py` | Steering vector injection into Coconut's forward pass |

## Troubleshooting

### CUDA out of memory
Reduce `n_contrast` or `n_probe` in config, or run on CPU by modifying `device` logic.

### Vectors not found
Run `extract_vectors.py` before `run_evaluation.py` for the same mode.

### Missing `coconut` module
Ensure the coconut directory is in your Python path:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "coconut"))
```
