"""
Training an FNO/PINO model on 2D simulations of Aliev-Panfilov Cell Model 
=============================
Based on the plot_FNO_darcy.py example from the neuralop library. 

Optional Arguments: 
    - Dataset to train on (specify via filepath)
    - Training and testing set size - informed based on the size of the dataset
    - Batch sizes for testing and training 
    - Modes to keep during fourier transform step 
    - Number of hidden channels to use
    - Flag to incorporate physics loss into the model 

"""

#Import required modules for constructing and training the model.
import torch
import yaml
import shutil
# NeuralOps and data loading 
from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND
from AP_neuralop_utils import Trainer
from neuralop.training import AdamW
from AP_neuralop_utils import load_2D_AP
from neuralop.utils import count_model_params
from neuralop.losses import SoftAdapt, Relobralo
from AP_neuralop_utils import MSELoss, RMSELoss, WeightedMSELoss, WeightedRMSELoss, ImpulseAwareMSELoss, ImpulseAwareRMSELoss, SoftRelativeL2ImpulseLoss, HardRelativeL2ImpulseLoss, ImpulseRMSELoss, APLoss, WeightedSumLoss, LpLoss, BoundaryLoss, ICLoss, BCNeumann, APFFTLoss, AdaptiveTrainingLoss

# Device and system imports 
import sys


import os
import subprocess
from pathlib import Path
import re
from torch.utils.data import Dataset, DataLoader

#imports required for data logging
import json
from datetime import datetime 


# Define the optional arguments to use when calling the operator
# if __name__ == "__main__":
class Args:
    def __init__(self, config_path):
        self.config_path = config_path

        with open(self.config_path, 'r') as f:
            self.config_yaml = yaml.safe_load(f)
        
        for key, value in self.config_yaml.items(): 
            setattr(self, key, value)

args = Args('train_config.yaml')

# A sequential run is orchestrated by launching this script once per stage.
# This keeps each stage isolated while passing the previous best checkpoint
# to the next stage and preserving the existing stage-specific code paths.
stage_child = os.environ.get("PINOS_STAGE_CHILD")
configured_train_types = getattr(args, "train_types", None)
if configured_train_types is not None and not stage_child:
    if isinstance(configured_train_types, str):
        configured_train_types = [item.strip() for item in configured_train_types.split(",") if item.strip()]
    allowed_train_types = {"init", "main"}
    if not isinstance(configured_train_types, list) or not configured_train_types:
        raise ValueError("train_types must be a non-empty list containing init and/or main")
    configured_train_types = [str(item).lower() for item in configured_train_types]
    invalid_train_types = set(configured_train_types) - allowed_train_types
    if invalid_train_types:
        raise ValueError(f"Unknown train_types: {sorted(invalid_train_types)}")
    if len(set(configured_train_types)) != len(configured_train_types):
        raise ValueError("train_types must not contain duplicate stages")
    configured_train_epochs = getattr(args, "train_types_epochs", None)
    if not isinstance(configured_train_epochs, list) or len(configured_train_epochs) != len(configured_train_types):
        raise ValueError(
            "train_types_epochs must be a list with exactly one epoch value "
            "for each item in train_types"
        )
    configured_train_epochs = [int(value) for value in configured_train_epochs]
    if any(value < 1 for value in configured_train_epochs):
        raise ValueError("All train_types_epochs values must be positive")

    sequence_timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    dataset_label = "dataset"
    dataset_path_for_label = str(getattr(args, "data_path", "")).lower()
    for candidate in ("stable", "chaotic", "planar", "centrifugal"):
        if candidate in dataset_path_for_label:
            dataset_label = candidate
            break
    sequential_run_dir = Path("runs") / f"{dataset_label}_sequential_{args.frames}_frames_{sequence_timestamp}"
    previous_model = str(getattr(args, "model_path_train", "") or "")
    for stage, stage_epochs in zip(configured_train_types, configured_train_epochs):
        child_env = os.environ.copy()
        child_env["PINOS_STAGE_CHILD"] = stage
        child_env["PINOS_STAGE_EPOCHS"] = str(stage_epochs)
        child_env["PINOS_RUN_DIR"] = str(sequential_run_dir)
        child_env["PINOS_MODEL_PATH"] = previous_model
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=child_env)
        if result.returncode != 0:
            sys.exit(result.returncode)

        checkpoint_candidates = sorted(
            sequential_run_dir.glob(f"train/**/best_model_state_dict_{stage}.pt"),
            key=lambda path: path.stat().st_mtime,
        )
        if not checkpoint_candidates:
            raise FileNotFoundError(
                f"Stage '{stage}' completed but no best_model_state_dict_{stage}.pt "
                f"was found below {sequential_run_dir / 'train'}"
            )
        previous_model = str(checkpoint_candidates[-1].resolve())
        print(f"Completed stage '{stage}'. Next stage checkpoint: {previous_model}")
    sys.exit(0)

if stage_child:
    args.train_type = stage_child.lower()
    args.train_types = None
    args.model_path_train = os.environ.get("PINOS_MODEL_PATH") or None
    stage_epochs = int(os.environ["PINOS_STAGE_EPOCHS"])
    # Keep the legacy internal field populated for run names and diagnostics.
    args.epochs = stage_epochs
    if args.train_type == "init":
        args.init_epochs = stage_epochs
    else:
        args.epochs = stage_epochs
        args.main_epochs = stage_epochs



#8.0_K_0.099_a_0.15_b_0.002_epsilon_0.2_mu1_0.3_mu2_1.0_conmul_2500
## ----------------------------------------------------------------------- ##
# FILE SET FOR SAVING RESULTS
## ----------------------------------------------------------------------- ##

# Define a class for logging the outputs of the training process
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() 
    def flush(self):
        for f in self.files:
            f.flush()


def log_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def log_subsection(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

# Generate the results folder to store the training logs and results
timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M")
main_stage_id = f"main_{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}"
dataset_path_lower = args.data_path.lower()
train_type = str(getattr(args, "train_type", "full")).lower()
model_path_train = getattr(args, "model_path_train", None)
model_path_train = str(model_path_train).strip() if model_path_train else None

if train_type not in {"init", "main", "full"}:
    raise ValueError("train_type must be one of: init, main, full")


# Determine if dataset is planar, centrifugal, spiral or spiral-breakup
if "stable" in dataset_path_lower:
    dataset_type = "stable"
elif "chaotic" in dataset_path_lower:
    dataset_type = "chaotic"
elif "planar" in dataset_path_lower:
    dataset_type = "planar"
elif "centrifugal" in dataset_path_lower:
    dataset_type = "centrifugal"
else:
    dataset_type = "Unknown"

# Build run folder name based on physics loss and dataset type
if args.phys_loss: 
    run_name = (f"{dataset_type}_PINO_"
                f"{args.adapt}_{args.epochs}_{args.eval_metric}_{args.frames}_frames{timestamp}")
else: 
    run_name = f"{dataset_type}_Data_Only_{args.eval_metric}_{args.frames}_frames_{timestamp}"


def infer_run_dir_from_model_path(model_path):
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    parts = path.parts
    if "train" in parts:
        train_idx = parts.index("train")
        if train_idx > 0:
            return Path(*parts[:train_idx])
    return None


runs_root = Path("runs")
loaded_run_dir = infer_run_dir_from_model_path(model_path_train) if model_path_train else None
if os.environ.get("PINOS_RUN_DIR"):
    run_dir = Path(os.environ["PINOS_RUN_DIR"])
elif train_type == "main" and loaded_run_dir is not None:
    run_dir = loaded_run_dir
else:
    run_dir = runs_root / run_name

train_root = run_dir / "train"
eval_root = run_dir / "eval"
init_train_dir = train_root / "init"
main_train_dir = train_root / main_stage_id

# Create the directory if it doesn't exist
os.makedirs(run_dir, exist_ok=True)
os.makedirs(train_root, exist_ok=True)
os.makedirs(eval_root / "p2p", exist_ok=True)
os.makedirs(eval_root / "rollout", exist_ok=True)
print(f"Saving run to: {run_dir}")


def stage_path(stage):
    if stage == "init":
        return init_train_dir
    if stage == "main":
        return main_train_dir
    raise ValueError(f"Unknown stage {stage}")


def stage_file(stage_dir, filename, stage):
    stem, ext = os.path.splitext(filename)
    return os.path.join(stage_dir, f"{stem}_{stage}{ext}")


def write_stage_config(stage_dir, stage):
    os.makedirs(stage_dir, exist_ok=True)
    # Copy the original file verbatim so comments and formatting are preserved.
    source_config = Path(args.config_path)
    if not source_config.is_absolute():
        source_config = Path.cwd() / source_config
    destination_config = Path(stage_file(stage_dir, "train_config.yaml", stage))
    shutil.copyfile(source_config, destination_config)


def load_model_weights(model, model_path):
    if not model_path:
        return
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    print(f"Loading training model weights from: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)


def finalize_training_artifacts(stage_dir, stage, log_jsonl_name):
    log_jsonl = os.path.join(stage_dir, log_jsonl_name)
    log_json = stage_file(stage_dir, "training_log.json", stage)
    if os.path.exists(log_jsonl):
        with open(log_jsonl, "r") as file:
            data = [json.loads(line) for line in file if line.strip()]
        with open(log_json, "w") as file:
            json.dump(data, file, indent=2)
        suffixed_jsonl = stage_file(stage_dir, "training_log.jsonl", stage)
        if log_jsonl != suffixed_jsonl:
            os.replace(log_jsonl, suffixed_jsonl)

    rename_map = {
        "model_state_dict.pt": f"model_state_dict_{stage}.pt",
        "best_model_state_dict.pt": f"best_model_state_dict_{stage}.pt",
        "optimizer_state_dict.pt": f"optimizer_state_dict_{stage}.pt",
        "scheduler_state_dict.pt": f"scheduler_state_dict_{stage}.pt",
        "regularizer_state_dict.pt": f"regularizer_state_dict_{stage}.pt",
        "best_model_metric.json": f"best_model_metric_{stage}.json",
        "manifest.pt": f"manifest_{stage}.pt",
    }
    for src_name, dst_name in rename_map.items():
        src = os.path.join(stage_dir, src_name)
        dst = os.path.join(stage_dir, dst_name)
        if os.path.exists(src):
            os.replace(src, dst)


def model_for_eval(stage_dir, stage):
    best_path = stage_file(stage_dir, "best_model_state_dict.pt", stage)
    final_path = stage_file(stage_dir, "model_state_dict.pt", stage)
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(final_path):
        return final_path
    raise FileNotFoundError(f"No saved model found for stage {stage} in {stage_dir}")


#Set the device for training:
device = 'cuda'


## ----------------------------------------------------------------------- ##
#  LOADING AND PREPARING THE DATASETS (WITH INSPECTIONS)
## ----------------------------------------------------------------------- ##


# Define the path to find the dataset to load from 
def get_data_root(custom_path: str):
    return (Path.cwd() / custom_path).resolve()
print('\n### --------- ###\n')
custom_path = args.data_path
example_data_root = get_data_root(custom_path)
print(f"Loading datasets from {example_data_root}")

# Define the size of the training and testing set sizes using the dataset_info file in the folder (if available):

#Default values for training (if no dataset info file is available)
n_train = getattr(args, "n_train_samples", getattr(args, "n_train", None))
train_res = args.train_res

#Use the dataset info to extract the relevant parameters for embedding and training set up 
target_name = f"dataset_info_{args.train_res}_{args.conmul}.txt"
dataset_info = os.path.join(args.data_path, target_name)
print(dataset_info)
with open(dataset_info, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith('Grid_resolution'):
            numbers = re.findall(r'\d+\.?\d*', line)
            if numbers:
                dx = float(numbers[0])
                print(f"dx: {dx}")
                dy = float(numbers[0])
                print(f"dy: {dy}")
        if line.startswith('Timestep resolution'):
            numbers = re.findall(r'\d+', line)
            if numbers:
                delta_t = int(numbers[0])
                print(f"delta_t: {delta_t} ms")
        if line.startswith('Training data shapes:'):
            numbers = re.findall(r'\d+', line)
            if numbers:
                n_train = int(numbers[0])
                print(f"n_train: {n_train}")
                train_res = int(numbers[3])
                print(f"training_resolution: {train_res}")
        if line.startswith("Input-Output pairs"):
            numbers = re.findall(r'\d+', line)
            if numbers:
                time_frames = numbers[0]
                print(f"Time boundary: {time_frames} frames")
        if line.startswith("Resting potential (E_rest) = "):
            # Match numbers (including negative and decimal)
            numbers = re.findall(r'-?\d+\.\d+', line)
            if numbers and len(numbers) >= 2:
                V_rest = float(numbers[0]) 
                V_amp = float(numbers[1])
                print(f"V_rest = {V_rest}, V_amp = {V_amp}")
print('\n### --------- ###\n')
time_boundary = (float(delta_t) * (float(time_frames) -1) )

# Default arguments for testing data:
n_test = getattr(args, "n_test_samples", getattr(args, "n_test", None))
test_res = args.test_res

# Extract testing data information: 
testing_dataset_info_files = [
    os.path.join(args.data_path, f)
    for f in os.listdir(args.data_path)
    if f.startswith("dataset_info_") and f.endswith(".txt")
]

#print(f"LOADING TEST DATASETS: {testing_dataset_info_files}")


test_resolutions = []   # store all test resolutions here
cm_tests = [] # store all conductivity multiplier here
n_tests = [] # store number of testing samples here
batch_tests = [] #store the size of the testng batches here

for dataset_info in testing_dataset_info_files:
    basename = os.path.basename(dataset_info)
    match = re.match(r"dataset_info_(\d+)_([\d\.]+)\.txt", basename)
    if not match:
        print(f"Skipping file with unexpected name format: {basename}")
        continue
    res_str, conmul_str = match.groups()
    res = int(res_str)
    conmul = float(conmul_str) if '.' in conmul_str else int(conmul_str)
    with open(dataset_info, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('Testing data shapes:'):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    n_test = int(numbers[0])
                    #print(f"n_test: {n_test}")
                    n_tests.append(n_test)
                    res = int(numbers[3]) 
                    test_resolutions.append(res)
                    batch_tests.append(int(args.batch_size_test))
    cm_tests.append(conmul)

print('\n### --------- ###\n')
print(f" Number of testing samples: {n_tests}")
print(f" Testing resolutions: {test_resolutions}") 
print(f" Testing Conductivities: {cm_tests}")
print(f" Testing batch sizes: {batch_tests}")
print('\n### --------- ###\n')

# Load the datasets (using the new loading function defined for the dataset): 
train_loader, test_loaders, data_processor = load_2D_AP(
        n_train=n_train, 
        batch_size=int(args.batch_size),
        train_resolution= train_res,
        test_resolutions= test_resolutions, 
        n_tests=n_tests,
        test_batch_sizes= batch_tests, 
        data_root=example_data_root, 
        dataset_name = '2D_AP',
        cm_train = args.conmul, 
        cm_tests = cm_tests,
        encode_input = args.encode_input, 
        encode_output = args.encode_output,
        start_index = getattr(args, "start_index", 0)
)
data_processor = data_processor.to(device)


# Establish the embedding and boundaries for the inclusion of the time channel (Adapting the spatial embeddings for the grid resolutions):
print('\n### --------- ###\n')
print(f"Mesh size: {args.mesh_size} cm x {args.mesh_size} cm")
time_boundary = (float(delta_t) * (float(time_frames) -1) )
print(f"Sample Time Boundary: {time_frames} frames with spacing {delta_t} ms =  {time_boundary} ms ")
embedding = GridEmbeddingND(in_channels=args.ch, dim=3, grid_boundaries=[[0,float(time_boundary)], [0, args.mesh_size], [0, args.mesh_size]])
print("Embedding Grid Boundaries =", embedding.grid_boundaries)


# Load one training sample for inspection
train_dataset = train_loader.dataset
index = 0
data = train_dataset[index]
data = data_processor.preprocess(data, batched=True)  # shape: [channels, time, H, W]
if args.encode_input:
    batch_size, channels, time_steps, H, W = data['x'].shape
    print(f"Sample Data shape: {data['x'].shape}: batch_size={batch_size}, channels={channels}, time={time_steps}, H={H}, W={W}\n")
else:
    channels, time_steps, H, W = data['x'].shape
    print(f"Sample Data shape: {data['x'].shape}: channels={channels}, time={time_steps}, H={H}, W={W}\n")
print('\n### --------- ###\n')

## ----------------------------------------------------------------------- ##
# CREATING THE MODEL
## ----------------------------------------------------------------------- ##


# Building the FNO model structure for training 
model = FNO(n_modes=(8, 16, 16),
             in_channels=args.ch, 
             out_channels=args.ch,
             hidden_channels=32, 
             projection_channel_ratio=2,
             positional_embedding = embedding)
model = model.to(device)
n_params = count_model_params(model)
print(f'\nModel has {n_params} parameters.')
#sys.stdout.Tee()

## TRAINING SETUP ##

# Define optimisation 
optimizer = AdamW(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
#scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(...)

## ----------------------------------------------------------------------- ##
# DEFINING THE LOSSES   
## ----------------------------------------------------------------------- ##



# data losses 
l2loss = LpLoss(d=2, p=2)
mse = MSELoss()
rmse = RMSELoss()
boundary = BoundaryLoss()
ic = ICLoss()
bcn = BCNeumann()


def make_data_loss(loss_type):
    loss_type = str(loss_type).lower()
    if loss_type in ["mse", "mean_squared_error"]:
        return MSELoss(reduction="mean")
    if loss_type in ["rmse", "root_mean_squared_error"]:
        return RMSELoss(reduction="mean")
    if loss_type in ["weighted_mse", "impulse_mse"]:
        return WeightedMSELoss(
            reduction="mean",
            impulse_threshold=getattr(args, "impulse_threshold", 0.05),
            impulse_weight=getattr(args, "impulse_weight", 10.0),
            transition_threshold=getattr(args, "transition_threshold", None),
            transition_weight=getattr(args, "transition_weight", 0.0),
        )
    if loss_type in ["weighted_rmse", "impulse_rmse"]:
        return WeightedRMSELoss(
            reduction="mean",
            impulse_threshold=getattr(args, "impulse_threshold", 0.05),
            impulse_weight=getattr(args, "impulse_weight", 10.0),
            transition_threshold=getattr(args, "transition_threshold", None),
            transition_weight=getattr(args, "transition_weight", 0.0),
        )
    if loss_type in ["impulse_aware_mse", "asymmetric_impulse_mse"]:
        return ImpulseAwareMSELoss(
            reduction="mean",
            impulse_threshold=getattr(args, "impulse_threshold", 0.05),
            false_negative_weight=getattr(args, "false_negative_weight", 20.0),
            false_positive_weight=getattr(args, "false_positive_weight", 5.0),
            peak_weight=getattr(args, "peak_weight", 5.0),
        )
    if loss_type in ["impulse_aware_rmse", "asymmetric_impulse_rmse"]:
        return ImpulseAwareRMSELoss(
            reduction="mean",
            impulse_threshold=getattr(args, "impulse_threshold", 0.05),
            false_negative_weight=getattr(args, "false_negative_weight", 20.0),
            false_positive_weight=getattr(args, "false_positive_weight", 5.0),
            peak_weight=getattr(args, "peak_weight", 5.0),
        )
    if loss_type == "soft_relative_l2_impulse_loss":
        return SoftRelativeL2ImpulseLoss(
            false_negative_weight=getattr(args, "false_negative_weight", 0.1),
            false_positive_weight=getattr(args, "false_positive_weight", 0.1),
            impulse_threshold=getattr(args, "impulse_threshold", 0.05),
            impulse_temperature=getattr(args, "impulse_temperature", 0.05),
            eps=getattr(args, "impulse_loss_eps", 1e-8),
        )
    if loss_type == "hard_relative_l2_impulse_loss":
        return HardRelativeL2ImpulseLoss(
            false_negative_weight=getattr(args, "false_negative_weight", 0.1),
            false_positive_weight=getattr(args, "false_positive_weight", 0.1),
            impulse_threshold=getattr(args, "impulse_threshold", 0.05),
            impulse_temperature=getattr(args, "impulse_temperature", 0.05),
            eps=getattr(args, "impulse_loss_eps", 1e-8),
        )
    if loss_type in ["l2", "lp", "relative_l2", "relative_lp"]:
        return LpLoss(d=2, p=2)
    raise ValueError(
        f"Unknown data_loss_type={loss_type}. "
        "Supported values: mse, rmse, weighted_mse, weighted_rmse, "
        "impulse_aware_mse, impulse_aware_rmse, relative_l2, "
        "soft_relative_l2_impulse_loss, hard_relative_l2_impulse_loss."
    )


data_loss_type = getattr(args, "data_loss_type", "mse")
init_data_loss_type = getattr(args, "init_data_loss_type", data_loss_type)
main_data_loss_type = getattr(args, "main_data_loss_type", data_loss_type)
init_data_loss = make_data_loss(init_data_loss_type)
main_data_loss = make_data_loss(main_data_loss_type)
print(f"Using init data loss type: {init_data_loss_type} ({init_data_loss})")
print(f"Using main data loss type: {main_data_loss_type} ({main_data_loss})")
soft_relative_l2_impulse_metric = SoftRelativeL2ImpulseLoss(
    false_negative_weight=getattr(args, "false_negative_weight", 0.1),
    false_positive_weight=getattr(args, "false_positive_weight", 0.1),
    impulse_threshold=getattr(args, "impulse_threshold", 0.05),
    impulse_temperature=getattr(args, "impulse_temperature", 0.05),
    eps=getattr(args, "impulse_loss_eps", 1e-8),
)
hard_relative_l2_impulse_metric = HardRelativeL2ImpulseLoss(
    false_negative_weight=getattr(args, "false_negative_weight", 0.1),
    false_positive_weight=getattr(args, "false_positive_weight", 0.1),
    impulse_threshold=getattr(args, "impulse_threshold", 0.05),
    impulse_temperature=getattr(args, "impulse_temperature", 0.05),
    eps=getattr(args, "impulse_loss_eps", 1e-8),
)
impulse_rmse_metric = ImpulseRMSELoss(
    impulse_threshold=getattr(args, "impulse_threshold", 0.05),
    eps=getattr(args, "impulse_loss_eps", 1e-8),
)

#conductivity scaling: 
D = args.D * float(args.conmul)

# FFT FDM loss 
apfft = APFFTLoss(k=args.K, a=args.a, epsilon=args.epsilon, mu1=args.mu1, mu2=args.mu2, b=args.b, D=args.D, Lt=float(time_boundary), t_scale=args.t_scale)

# physics losses
apfdm = APLoss( k=args.K, 
                 a=args.a, 
                 epsilon=args.epsilon, 
                 mu1=args.mu1, 
                 mu2=args.mu2, 
                 b=args.b,
                 D = D,
                 method = args.phys_method, 
                 delta_t = delta_t, 
                 Lt = time_boundary,
                 dx = dx, 
                 dy = dy, 
                 v_loss_weighting = args.v_loss, 
                 w_loss_weighting = args.w_loss,
                 ic_weighting=args.ic_loss,
                 bc_weighting=args.bc_loss,
                 res_weighting=args.res_loss,
                 t_scale = args.t_scale, 
                 V_rest = V_rest,
                 V_amp = V_amp
)

# Determine which losses to use for training based on physics loss weight
# resloss is the physics-based loss (either FDM or FFT)

if args.phys_method == "finite_difference":
    resloss = apfdm
    print("Calculating Residual Loss via Finite Difference")
elif args.phys_method == "finite_difference_fft":
    resloss = apfft
    print("Calculating Residual Loss via FFT Finite Difference")
else:
     print("Unknown method for residual loss calculation!")

# --- Define a physics-only loss for evaluation ---
phys_eval_loss = WeightedSumLoss(
    losses=[resloss, ic, bcn],
    weights=[args.res_loss, args.ic_loss, args.bc_loss]
)

# Establish Global Evaluation losses:
eval_losses = {
        'l2': l2loss,
        'mse': mse,
        'rmse': rmse,
        'ap_phys': resloss,
        'boundary': boundary,
        'ic': ic,
        'bcn': bcn,
        'phys_loss': phys_eval_loss,
        'soft_relative_l2_impulse_loss': soft_relative_l2_impulse_metric,
        'hard_relative_l2_impulse_loss': hard_relative_l2_impulse_metric,
        'impulse_rmse': impulse_rmse_metric,
    }

# Keep impulse_rmse in both training and validation metric collections.  The
# Trainer uses this same object to produce train_impulse_rmse and the
# per-validation-loader impulse_rmse series used by training_plots/.
assert 'impulse_rmse' in eval_losses

def stage_eval_metric(metric_name):
    return f"({args.eval_res}, {args.eval_con})_{metric_name}"


init_eval_metric_name = getattr(args, "init_eval_metric", getattr(args, "eval_metric", "rmse"))
main_eval_metric_name = getattr(args, "main_eval_metric", getattr(args, "eval_metric", "rmse"))
init_eval_metric = stage_eval_metric(init_eval_metric_name)
main_eval_metric = stage_eval_metric(main_eval_metric_name)
early_stopping_enabled = bool(getattr(args, "early_stopping", False))
early_stopping_patience = getattr(args, "early_stopping_patience", None)
early_stopping_min_delta = getattr(args, "early_stopping_min_delta", 0.0)
eval_step = int(getattr(args, "eval_step", 100))


def run_evaluation(test_config_path, stage_train_path, stage_model_path, eval_type, eval_data_type, stage, stage_dir_name):
    # Update evaluation_config.yaml for this stage-specific evaluation.
    with open(test_config_path, 'r') as f:
        test_config = yaml.safe_load(f)

    eval_output_dir = eval_root / eval_type / stage_dir_name / eval_data_type
    os.makedirs(eval_output_dir, exist_ok=True)

    test_config['model_path_eval'] = stage_model_path
    test_config['data_path'] = stage_train_path
    test_config['train_result_path'] = stage_train_path
    test_config['eval_output_dir'] = str(eval_output_dir)
    test_config['eval_data_type'] = eval_data_type
    test_config['eval_stage'] = stage
    test_config['eval_stage_dir_name'] = stage_dir_name
    test_config['run_dir'] = str(run_dir)

    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)

    print(f"✓ Обновлен {test_config_path}")

    script_dir = Path(__file__).parent
    if eval_type == 'rollout':
        eval_script = script_dir / "Evaluation_Rollout.py"
    elif eval_type == 'p2p':
        eval_script = script_dir / "Evaluation_P2P.py"
    else:
        raise ValueError(f"{eval_type} does not exist")

    if not eval_script.exists():
        raise FileNotFoundError(f"Не найден {eval_script}")

    print(f"\n▶ Запуск {eval_script}...")
    print("="*50)

    result = subprocess.run(
        [sys.executable, str(eval_script)],
        cwd=script_dir,
        stdout=None,
        stderr=None
    )

    if result.returncode != 0:
        print(f"\n✗ Ошибка: скрипт завершился с кодом {result.returncode}")
        sys.exit(result.returncode)

    print("="*50)
    print("✓ Evaluation завершен")


def run_stage_evaluations(stage, stage_dir_name, stage_train_path):
    if not args.eval_types:
        return
    stage_model_path = model_for_eval(stage_train_path, stage)
    test_config_path = "evaluation_config.yaml"
    with open(test_config_path, "r") as f:
        test_config = yaml.safe_load(f) or {}
    configured_data_types = test_config.get("eval_types", ["full"])
    for eval_type in args.eval_types:
        for eval_data_type in configured_data_types:
            if eval_data_type not in {"full", "sample"}:
                raise ValueError("test_config.eval_types values must be 'full' or 'sample'")
            run_evaluation(
                test_config_path=test_config_path,
                stage_train_path=stage_train_path,
                stage_model_path=stage_model_path,
                eval_type=eval_type,
                eval_data_type=eval_data_type,
                stage=stage,
                stage_dir_name=stage_dir_name,
            )

## ----------------------------------------------------------------------- ##
# INITIAL TRAINING ROUND: DATA ONLY LOSS
## ----------------------------------------------------------------------- ##
init_completed = False
if train_type in {"init", "full"}:
    init_stage_dir = str(stage_path("init"))
    os.makedirs(init_stage_dir, exist_ok=True)
    write_stage_config(init_stage_dir, "init")
    init_log_file = open(stage_file(init_stage_dir, args.output + ".txt", "init"), "w")
    sys.stdout = Tee(sys.__stdout__, init_log_file)

    if model_path_train:
        load_model_weights(model, model_path_train)

    train_loss = init_data_loss

    # Training the model
    # ---------------------
    # Print out the training info
    log_section("INITIAL TRAINING ROUND")
    log_subsection("MODEL")
    print(model)
    log_subsection("OPTIMIZER")
    print(optimizer)
    log_subsection("SCHEDULER")
    print(scheduler)
    log_subsection("LOSSES")
    print(f'\n * Train: {train_loss}')
    print(f'\n * Train data loss type: {init_data_loss_type}')
    print(f'\n * Test: {eval_losses}')
    print(f'\n * Initial Training Round for {args.init_epochs} epochs')
    print(f'\n * Saving best init model according to: {init_eval_metric_name}')
    #sys.stdout.flush()

    # Create the trainer:
    trainer = Trainer(model=model, n_epochs=args.init_epochs,
                      device=device,
                      data_processor=data_processor,
                      wandb_log=True,
                      eval_interval=eval_step,
                      use_distributed=False,
                      verbose=True
                      )
    # Then train the model on the loaded dataset for an inital training run, checkpoint for each epoch
    json_log_path=os.path.join(init_stage_dir, "training_log.jsonl")
    trainer.train(train_loader=train_loader,
                  test_loaders=test_loaders,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=False,
                  training_loss=train_loss,
                  train_metrics=eval_losses,
                  eval_losses=eval_losses,
                  save_best = init_eval_metric,
                  save_dir = init_stage_dir,
                  json_log_path = json_log_path,
                  training_plots_dir = os.path.join(init_stage_dir, "training_plots"),
                  stage_name = "Initial data training",
                  early_stopping = early_stopping_enabled,
                  early_stopping_metric = init_eval_metric,
                  early_stopping_patience = early_stopping_patience,
                  early_stopping_min_delta = early_stopping_min_delta
                  )
    finalize_training_artifacts(init_stage_dir, "init", "training_log.jsonl")
    init_completed = True
    sys.stdout = sys.__stdout__
    init_log_file.close()
    run_stage_evaluations("init", "init", init_stage_dir)
    if train_type == "full":
        load_model_weights(model, model_for_eval(init_stage_dir, "init"))
elif train_type == "main" and model_path_train:
    load_model_weights(model, model_path_train)

## ----------------------------------------------------------------------- ##
# MAIN TRAINING ROUND: PHYSICS LOSS INCLUDED IN THE TRAINING
## ----------------------------------------------------------------------- ##

# Pick the weighting combitnations for each of the losses


if args.phys_loss:

    if args.adapt == 2.0:

        # Use SoftAdapt weighting during the main training phase

        num_losses = 4  # data, phys, ic, bcn
        initial_weights = {
            'data': args.data_loss,
            'phys': args.res_loss,  
            'ic': args.ic_loss,     
            'bc': args.bc_loss      
        }
    
        aggregator = SoftAdapt(
            params=model.parameters(),
            num_losses=num_losses,
            weights=initial_weights,
            eps=1e-8
        )

        # Adaptive training loss using SoftAdapt
        training_loss = AdaptiveTrainingLoss(
            aggregator=aggregator,
            data_loss=main_data_loss,
            phys_loss=resloss,      # apfdm or apfft depending on args
            ic_loss=ic,       
            bc_loss=bcn            
        )
        train_loss = training_loss

    elif args.adapt == 3.0:
            
        # Use Relative weighting during the main training phase

        num_losses = 4  # data, phys, ic, bcn
        initial_weights = {
            'data': args.data_loss,
            'phys': args.res_loss,  
            'ic': args.ic_loss,     
            'bc': args.bc_loss      
        }
    

        aggregator = Relobralo(
        params=model.parameters(),
        num_losses=num_losses,
        alpha=getattr(args, "alpha", 0.95),
        beta=getattr(args, "beta", 0.99),
        tau=getattr(args, "tau", 1.0),
        eps=1e-8,
        weights=initial_weights
    )

        # Adaptive training loss using SoftAdapt
        training_loss = AdaptiveTrainingLoss(
            aggregator=aggregator,
            data_loss=main_data_loss,
            phys_loss=resloss,      # apfdm or apfft depending on args
            ic_loss=ic,       
            bc_loss=bcn            
        )         
        train_loss = training_loss

    else:
        # Use physics loss weighted with data loss using fixed weights
        combinedloss = WeightedSumLoss(
            losses=[main_data_loss, resloss, ic, bcn],
            weights=[args.data_loss, args.res_loss, args.ic_loss, args.bc_loss]
        )
        train_loss = combinedloss
else:
    train_loss = main_data_loss



main_completed = False
main_epochs = int(getattr(args, "main_epochs", getattr(args, "epochs", 1000)))
if train_type in {"main", "full"}:
    main_stage_dir = str(stage_path("main"))
    os.makedirs(main_stage_dir, exist_ok=True)
    write_stage_config(main_stage_dir, "main")
    main_log_file = open(stage_file(main_stage_dir, args.output + ".txt", "main"), "w")
    sys.stdout = Tee(sys.__stdout__, main_log_file)

    # Training the model
    # ---------------------

    # Print out the training info
    log_section("MAIN TRAINING ROUND")
    log_subsection("MODEL")
    print(model)
    log_subsection("OPTIMIZER")
    print(optimizer)
    log_subsection("SCHEDULER")
    print(scheduler)
    log_subsection("AGGREGATOR")
    if args.adapt in [2.0, 3.0]:
        print(aggregator)
    else:
        print('None (not used for this configuration)')
    log_subsection("LOSSES")
    print(f'\n * Train: {train_loss}')
    print(f'\n * Train data loss type: {main_data_loss_type}')
    print(f'\n * Test: {eval_losses}')
    print(f'\n * Main Training Round for {main_epochs} epochs')
    print(f'\n * Saving best main model according to: {main_eval_metric_name}')
    #sys.stdout.flush()

    # Create the trainer:
    trainer = Trainer(model=model, n_epochs=main_epochs,
                      device=device,
                      data_processor=data_processor,
                      wandb_log=True,
                      eval_interval=eval_step,
                      use_distributed=False,
                      verbose=True
                      )

    # Then train the model on the loaded dataset - save the best performing model according to given metric (default is the global mse score)
    print(f"Best model metric: {main_eval_metric}")
    json_log_path=os.path.join(main_stage_dir, "training_log.jsonl")
    trainer.train(train_loader=train_loader,
                  test_loaders=test_loaders,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=False,
                  training_loss=train_loss,
                  train_metrics=eval_losses,
                  eval_losses=eval_losses,
                  save_best = main_eval_metric,
                  save_dir = main_stage_dir,
                  json_log_path = json_log_path,
                  training_plots_dir = os.path.join(main_stage_dir, "training_plots"),
                  stage_name = "Main training",
                  early_stopping = early_stopping_enabled,
                  early_stopping_metric = main_eval_metric,
                  early_stopping_patience = early_stopping_patience,
                  early_stopping_min_delta = early_stopping_min_delta
                  )
    finalize_training_artifacts(main_stage_dir, "main", "training_log.jsonl")
    main_completed = True
    sys.stdout = sys.__stdout__
    main_log_file.close()
    run_stage_evaluations("main", main_stage_id, main_stage_dir)
