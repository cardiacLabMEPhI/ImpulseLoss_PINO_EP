# CardiacEP-PINOS

CardiacEP-PINOS is a Physics-Informed Neural Operator pipeline for learning cardiac action-potential propagation from OpenCARP simulations. It uses the Aliev–Panfilov model and a Fourier Neural Operator implemented with the [NeuralOperator](https://github.com/neuraloperator/neuraloperator) library.

The current training pipeline has two stages:

1. **Init** — data-driven pretraining.
2. **Main** — physics-informed training with data and Aliev–Panfilov physics losses.

Rollout is available as a post-training evaluation mode, but it is not a training stage.

## Repository layout

```text
.
├── AP_neuralop_utils/       # Data loaders, losses, model utilities, and Trainer
├── Data/                    # OpenCARP extraction and tensor construction
├── PINO_Train.py            # Init/main training entry point
├── PINO_Evaluation.py       # Evaluation wrapper
├── Evaluation_P2P.py       # Point-to-point evaluation
├── Evaluation_Rollout.py   # Autoregressive evaluation
├── train_config.yaml        # Training configuration
├── evaluation_config.yaml   # Evaluation configuration
└── requirements.txt
```

Generated datasets, checkpoints, logs, and evaluation results are local artifacts. The repository `.gitignore` excludes the main generated-output directories.

## Installation

Use Python 3.11 or a compatible version:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements install NeuralOperator from a pinned Git commit. A CUDA-enabled PyTorch installation is recommended for training. Evaluation automatically uses CUDA when it is available and otherwise falls back to CPU.

Run commands from the repository root:

```bash
cd /path/to/CardiacEP-PINOS
```

## Data preparation

The data pipeline has two steps: extract tensors from an OpenCARP simulation, then split the trajectory into train, validation/test, and evaluation windows.

### OpenCARP input

Set `sim_folder` in `Data/config.yaml` to the directory containing the OpenCARP output files. The standard extractor expects voltage and recovery-variable data, typically files such as `v.igb` and `vm.igb`.

The repository includes example mesh, cell-model, and state files in `Data/OpenCARP_simulation_files/`. Running OpenCARP is optional when precomputed simulation outputs are already available.

### Configure `Data/config.yaml`

Set the simulation path and split parameters. For example:

```yaml
sim_folder: /path/to/opencarp_simulation
sim_type: planar
runtime: 1000
is_full_eval: false
train: 0.8
test: 0.1       # Validation/test fraction
eval: 0.1
timestep: 5.0
input_window: 5
window_mode: overlapping  # overlapping or block
```

The fractions `train + test + eval` must not exceed `1.0`. With `is_full_eval: false`, the constructor creates sequential train, validation/test, and evaluation subsets. With `is_full_eval: true`, it creates a full trajectory for evaluation only.

Other important options include `space_factors`, `time_factors`, `extraction_start_index`, `n_repeat`, `norm`, `channels`, and `random_split`. Their descriptions are included in `Data/config.yaml`.

### Generate tensors

Run:

```bash
python Data/dataset_preparation.py
```

The generated data is written below:

```text
datasets_2/openCARP_Data/<simulation_type>/
```

Generated folders contain tensor files and a `dataset_info_*.txt` metadata file. Check this metadata before training. It records tensor shapes, sample counts, spatial resolution, and time resolution.

For a two-channel dataset, a tensor normally has the shape:

```text
(number_of_samples, channels, frames, height, width)
```

For example, `(152, 2, 5, 101, 101)` means 152 samples, two state channels, five frames, and a `101 × 101` spatial grid.

## Training configuration

Edit `train_config.yaml` before starting a run. At minimum, set the dataset path and available resolutions:

```yaml
data_path: /path/to/tensorized_dataset
frames: 5
train_res: 101
test_res:
  - 101
eval_res: 101
eval_con: 1.0
```

`data_path` must point to the generated tensor directory. `train_res`, `test_res`, and `eval_res` must match the resolutions available in the dataset.

### Training stages

The default configuration runs both stages sequentially:

```yaml
train_types: [init, main]
train_types_epochs: [1000, 1000]
```

- `init` optimizes the selected data loss.
- `main` continues from the best init checkpoint and adds the physics-informed loss when `phys_loss: true`.

The physics configuration contains the Aliev–Panfilov parameters, derivative method, and physics-loss weights. Adaptation modes are `1` for fixed weights, `2` for SoftAdapt, and `3` for RelAdapt.

Supported primary data-loss names are:

```text
mse
rmse
weighted_mse
weighted_rmse
impulse_aware_mse
impulse_aware_rmse
relative_l2
soft_relative_l2_impulse_loss
hard_relative_l2_impulse_loss
```

Example:

```yaml
init_data_loss_type: relative_l2
main_data_loss_type: soft_relative_l2_impulse_loss
```

Impulse-loss parameters are controlled by `false_negative_weight`, `false_positive_weight`, `impulse_threshold`, `impulse_temperature`, and `impulse_loss_eps`.

### Start training

After configuring `train_config.yaml`, run:

```bash
python PINO_Train.py
```

The script creates a timestamped directory under `runs/`. A typical result tree is:

```text
runs/<run_name>/
├── train/
│   ├── init/
│   │   ├── best_model_state_dict_init.pt
│   │   ├── training_log_init.txt
│   │   ├── training_log.jsonl
│   │   ├── validation_metrics.json
│   │   └── training_plots/
│   └── main_<timestamp>/
│       ├── best_model_state_dict_main.pt
│       ├── training_log_main.txt
│       └── training_plots/
└── eval/
```

When both stages are enabled, the main stage automatically receives the best init checkpoint. Set `model_path_train` only when starting from an existing checkpoint. Training and validation metrics are logged during the run, and plots are updated after validation events.

## Evaluation

Configure `evaluation_config.yaml` before evaluation:

```yaml
model_path_eval: /path/to/checkpoint.pt
data_path: /path/to/training_output
train_result_path: /path/to/training_output
data_path_full_eval: /path/to/full_evaluation_dataset
data_path_sample_eval: /path/to/sample_evaluation_dataset
eval_stage: init        # init or main
eval_types:
  - full
  - sample
```

Evaluation tensors must be compatible with the checkpoint: channel count, input-window length, spatial resolution, normalization, and time-step convention must match training. For rollout evaluation, `horizon` must not exceed the available trajectory length and `moving_window` must match the model input length.

Run all configured evaluations:

```bash
python PINO_Evaluation.py
```

Or run one method directly:

```bash
python Evaluation_P2P.py
python Evaluation_Rollout.py
```

To use another configuration file:

```bash
PINO_EVAL_CONFIG=/path/to/custom_evaluation_config.yaml python Evaluation_P2P.py
```

P2P evaluation uses ground-truth input windows for each prediction. Rollout evaluation is autoregressive and feeds model outputs back as inputs for subsequent predictions. Results are saved below the configured run directory, normally under `eval/p2p/<stage>/` and `eval/rollout/<stage>/`.

## Reproducibility

Keep the following fixed when comparing experiments:

- generated tensors and metadata;
- spatial and temporal resolution;
- model architecture (`hidden_channels`, `modes`, `ch`, and encoding flags);
- loss and physics parameters;
- train/validation split and `random_split` setting;
- optimizer and scheduler parameters;
- the software environment from `requirements.txt`.

GPU kernels and library versions can still produce small numerical differences.

## Common problems

### Empty validation or evaluation loader

Check the metadata file and make sure the selected tensor contains samples. A shape such as `torch.Size([0, 2, 5, 101, 101])` means that the split produced no samples. Adjust the source trajectory length, split fractions, input window, or sequential buffer.

### Checkpoint shape mismatch

The checkpoint and evaluation configuration must use the same architecture and data layout. Check `ch`, `hidden_channels`, `modes`, spatial resolution, input-window length, and input encoding.

### Rollout prediction has an invalid length

Set `horizon` to a value no larger than the number of target frames in the evaluation tensor. Verify that `moving_window` equals the model input frame count.

### Physics loss behaves unexpectedly

Verify the Aliev–Panfilov parameters, `t_scale`, `timestep`, `mesh_size`, `phys_method`, and tensor normalization. The physical scaling in the training configuration must match preprocessing.

## Citation

If you use this repository, please cite:

> *Impulse Loss Improves Physics-Informed Neural Operators in Cardiac Electrophysiology*
> [](https://...)

This project also uses NeuralOperator:

> Kossaifi, J., Kovachki, N., Li, Z., Pitt, D., Liu-Schiaffini, M., Duruisseaux, V., George, R., Bonev, B., Azizzadenesheli, V., Berner, J., and Anandkumar, A.
> *A Library for Learning Neural Operators.* arXiv, 2025.
> [doi:10.48550/arXiv.2412.10354](https://doi.org/10.48550/arXiv.2412.10354)
