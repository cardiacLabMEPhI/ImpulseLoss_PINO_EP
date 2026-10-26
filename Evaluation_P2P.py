"""
Evaluating the results of training an FNO/PINO model on 2D AP simulations using the best performing model. 
Runs a single evaluation pass on the evaluation dataset. 
=============================
Based on the plot_FNO_darcy.py example from the neuralop library. 

Optional Arguments: 
    - File path to load the datasets and trained model from. 
    - Training and testing set size - informed based on the size of the dataset
    - Batch sizes for testing and training 
    - Modes to keep during fourier transform step 
    - Number of hidden channels to use
"""

#Import required modules for constructing and training the model.
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from neuralop.models import FNO
#from neuralop.models import FNO, FNO1d, FNO2d, FNO3d 
from neuralop.layers.embeddings import GridEmbeddingND
from AP_neuralop_utils import load_2D_AP_eval
from torch.nn import MSELoss
from AP_neuralop_utils import (
    RMSELoss, ImpulseRMSELoss, APLoss, LpLoss, BoundaryLoss, ICLoss,
    BCNeumann, APFFTLoss,
)
import os
import shutil
from pathlib import Path
import re
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter 
#imports required for data logging
import json
import matplotlib.ticker as ticker
import time


# Define the optional arguments to use when calling the operator
class Args:
    def __init__(self, config_path):
        self.config_path = config_path

        with open(self.config_path, 'r') as f:
            self.config_yaml = yaml.safe_load(f)
        
        for key, value in self.config_yaml.items(): 
            setattr(self, key, value)

args = Args(os.environ.get("PINO_EVAL_CONFIG", "evaluation_config.yaml"))
eval_data_type = getattr(args, "eval_data_type", "full")
if eval_data_type not in ("full", "sample"):
    raise ValueError("eval_data_type must be 'full' or 'sample'")
if eval_data_type == "full":
    args.data_path_eval = args.data_path_full_eval
else:
    args.data_path_eval = args.data_path_sample_eval
eval_stage = getattr(args, "eval_stage", None)
if eval_stage not in ("init", "rollout", "main"):
    eval_stage = None


def stage_filename(filename):
    if not eval_stage:
        return filename
    stem, ext = os.path.splitext(filename)
    if stem.endswith(f"_{eval_stage}"):
        return filename
    return f"{stem}_{eval_stage}{ext}"


def stage_path(filename):
    return os.path.join(args.eval_output_dir, stage_filename(filename))

def innit_eval_dir():
    candidates = [
        getattr(args, "train_result_path", None),
        getattr(args, "data_path", None),
    ]
    train_dir = None
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = (Path.cwd() / candidate_path).resolve()
        if (
            (candidate_path / "training_log.json").exists()
            or (candidate_path / "best_model_state_dict.pt").exists()
            or (eval_stage and (candidate_path / f"training_log_{eval_stage}.json").exists())
            or (eval_stage and (candidate_path / f"best_model_state_dict_{eval_stage}.pt").exists())
            or (eval_stage and (candidate_path / f"model_state_dict_{eval_stage}.pt").exists())
        ):
            train_dir = candidate_path
            break
    if train_dir is None:
        train_dir = Path(getattr(args, "data_path", candidates[0])).expanduser()
        if not train_dir.is_absolute():
            train_dir = (Path.cwd() / train_dir).resolve()

    configured_output_dir = getattr(args, "eval_output_dir", None)
    if configured_output_dir:
        output_dir = Path(configured_output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
    else:
        output_dir = Path.cwd() / "eval_results" / "p2p" / train_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    args.train_result_path = str(train_dir)
    args.data_path = str(train_dir)
    args.eval_output_dir = str(output_dir)

    training_log_path = train_dir / stage_filename("training_log.json")
    if not training_log_path.exists():
        training_log_path = train_dir / "training_log.json"
    if training_log_path.exists():
        shutil.copy(training_log_path, output_dir / stage_filename("training_log.json"))
    model_path = Path(args.model_path_eval).expanduser()
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    if model_path.exists():
        shutil.copy(model_path, output_dir / stage_filename(model_path.name))

    print(f"Training results path: {args.train_result_path}")
    print(f"Evaluation output path: {args.eval_output_dir}")

innit_eval_dir()

## ======================================================================= ##
# == DEFINING KEY FUNCTIONS FOR THE SIMULATOR  == ##
## ======================================================================= ##

# Use CUDA when available, otherwise allow evaluation on CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the path to find the dataset to load from 
def get_data_root(custom_path: str):
    return (Path.cwd() / custom_path).resolve()


# Define function to plot the errors during training
def plot_training_log(training_log_json, data_path, phase):
    with open(training_log_json, 'r') as file:
        data = json.load(file)
    '''
          # Use this section if the run was interupted 
    with open(training_log_json, 'r') as file:
            data = [json.loads(line) for line in file if line.strip()]

    with open(training_log_json, 'w') as file:
        json.dump(data, file, indent=2)
    
    '''
    # Extract training log data for visualisation
    epochs = [entry["epoch"] for entry in data]
    train_err = [entry["train_err"] for entry in data]
    avg_loss = [entry["avg_loss"] for entry in data]
    train_time =  [entry["epoch_train_time"] for entry in data]

    def metric_series(suffix):
        series_epochs = []
        series_values = []
        for entry in data:
            value = next((v for k, v in entry.items() if re.search(rf"_{suffix}$", k)), None)
            if value is not None:
                series_epochs.append(entry["epoch"])
                series_values.append(value)
        return series_epochs, series_values

    # Load the testing errors throughout the training process
    h1 = [next((v for k, v in entry.items() if re.search(r"_h1$", k)), None) for entry in data]
    l2 = [next((v for k, v in entry.items() if re.search(r"_l2$", k)), None) for entry in data]
    mse = [next((v for k, v in entry.items() if re.search(r"_mse$", k)), None) for entry in data]
    rmse = [next((v for k, v in entry.items() if re.search(r"_rmse$", k)), None) for entry in data]
    phys_ap = [next((v for k, v in entry.items() if re.search(r"_ap_phys$", k)), None) for entry in data]
    combined = [next((v for k, v in entry.items() if re.search(r"_combined$", k)), None) for entry in data]
    oploss = [next((v for k, v in entry.items() if re.search(r"_op$", k)), None) for entry in data]
    ftloss = [next((v for k, v in entry.items() if re.search(r"_ft$", k)), None) for entry in data]

    # Plot the results:  
    # 1: Training and evaluation (testing) losses for the epochs 
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_loss, label="Training Loss ")
    
    #if any(h1): plt.plot(epochs, h1, label="Testing (Evaluation) Loss - h1 ", color = 'green', linestyle = 'dashdot')
    #if any(l2): plt.plot(epochs, l2, label="Testing (Evaluation) Loss - l2 ", color = 'green')
    mse_epochs, mse_values = metric_series("mse")
    if mse_values:
        plt.plot(mse_epochs, mse_values, label="Testing Loss - mse ", marker="o", linestyle="--")
    else:
        print(f"No testing MSE values found in {training_log_json}; skipping testing loss curve.")
    #if any(phys_ap): plt.plot(epochs, phys_ap, label="Testing (Evaluation) Loss - ap_phys ", color = 'red')
    #if any(combined): plt.plot(epochs, combined, label="Testing (Evaluation) Loss - combined ", color = 'red', linestyle = 'dashdot')
    #if any(oploss): plt.plot(epochs, combined, label="Testing (Evaluation) Loss - operator ", color = 'purple')
    #if any(ftloss): plt.plot(epochs, ftloss, label="Testing (Evaluation) Loss - finetune ", color = 'purple', linestyle = 'dashdot')

    plt.xlabel("Epoch")
    plt.ylabel("Model Losses")
    plt.title("Training/Testing Losses over Epochs")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, stage_filename("Training_Testing_Losses_" + phase + ".png")))
   # plt.show()


    '''
    # 2: Training Time  
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_time, color = 'black')

    plt.xlabel("Epoch")
    plt.ylabel("Training Time")
    plt.title("Model Training Time")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, stage_filename("Training_Time_" + phase + ".png")))
   # plt.show()
   '''
    
    # 2: Compute and return NET training time and overall/average rmse value on the training data:
    net_training_time = sum(train_time)
    # Filter out None values for mse/rmse
    valid_mse = [(i, v) for i, v in enumerate(mse) if v is not None]
    valid_rmse = [(i, v) for i, v in enumerate(rmse) if v is not None]

    min_mse, min_mse_epoch = (None, None)
    min_rmse, min_rmse_epoch = (None, None)

    if valid_mse:
        min_mse_idx, min_mse = min(valid_mse, key=lambda x: x[1])
        min_mse_epoch = epochs[min_mse_idx]
    if valid_rmse:
        min_rmse_idx, min_rmse = min(valid_rmse, key=lambda x: x[1])
        min_rmse_epoch = epochs[min_rmse_idx]

    # === Print summary ===
    print(f"\nNet training time: {net_training_time:.2f} seconds ({net_training_time/60:.2f} minutes)")
    if min_mse is not None:
        print(f" Minimum MSE: {min_mse:.6f} at epoch {min_mse_epoch}")
    if min_rmse is not None:
        print(f" Minimum RMSE: {min_rmse:.6f} at epoch {min_rmse_epoch}\n")

    # === Save summary to JSON ===
    summary = {
        "phase": phase,
        "net_training_time_sec": net_training_time,
        "net_training_time_min": net_training_time / 60,
        "min_mse": min_mse,
        "min_mse_epoch": min_mse_epoch,
        "min_rmse": min_rmse,
        "min_rmse_epoch": min_rmse_epoch
    }

    summary_path = os.path.join(data_path, f"Training_Summary_{phase}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved to: {summary_path}")



# Define function to extract relevant information from the dataset
def info_extraction(dataset_info):
    print('\n### --------- ###\n')
    #Use the dataset info to extract the relevant parameters for embedding and training set up 
    target_name = f"dataset_info_{args.eval_res}_{args.conmul}.txt"
    dataset_info = os.path.join(args.data_path_eval, target_name)
    print(dataset_info)
    num_timesteps = None
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
            if line.startswith('Evaluation data shapes:'):
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        num_timesteps = int(numbers[0])
                        print(f'Number of evaluation time frames:{num_timesteps}')
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
        return  dx, dy, delta_t, time_frames, num_timesteps
    

def reconstruct_rollout(y_pred_full, output_steps, stride=1):
    """
    Reconstructs a continuous temporal rollout from overlapping predicted windows.
    Averages predictions in overlapping regions.
    """
    B, C, T, X, Y = y_pred_full.shape
    T_total = (B - 1) * stride + T

    rollout = torch.zeros((C, T_total, X, Y), device=y_pred_full.device)
    count = torch.zeros_like(rollout)

    for i in range(B):
        start = i * stride
        end = start + T
        rollout[:, start:end, :, :] += y_pred_full[i]
        count[:, start:end, :, :] += 1

    rollout /= torch.clamp(count, min=1.0)
    return rollout

def infer_window_mode_from_path(path):
    path = str(path).lower()
    if "outputsteps_block" in path or "_block_" in path or path.endswith("_block"):
        return "block"
    return "overlapping"
        
'''
RESULTS:
Plot the training loss and ALL evaluation losses over the epochs

Re-load the best performing model and use this model to run a single forward pass on the evaluation set. 

Following the single forward pass, compare the prediction to the ground truth:
    - Return information about the best/worst performing cells and timesteps.
    - Plot the voltage map visualisations - 3 selected samples chosen to compare GT with prediction across all models 
    - Plot the single cell APDs - 3 selected samples chosen to compare GT with prediction across all models AND the best/worst performing cells. 
'''

## ======================================================================= ##
# == PLOTTING THE LOSSES THROUGHOUT TRAINING & TESTING PROCESS == ##
## ======================================================================= ##
if args.fine_tune: 
    
    # Plotting the fine tuning phase of training 
    training_log_json_2 = os.path.join(args.data_path, "training_log_phase_2.json")
    phase = "fine_tuning_phase"

    # Convert the json training log file into a valid file to read in the results analysis:
    
    with open(training_log_json_2, 'r') as file:
        data = [json.loads(line) for line in file if line.strip()]

    with open(training_log_json_2, 'w') as file:
        json.dump(data, file, indent=2)


    plot_training_log(training_log_json_2, args.eval_output_dir, phase)

else:
    # Plotting the training
    training_log_json_1 = os.path.join(args.data_path, stage_filename("training_log.json"))
    if not os.path.exists(training_log_json_1):
        training_log_json_1 = os.path.join(args.data_path, "training_log.json")
    phase = ""
    plot_training_log(training_log_json_1, args.eval_output_dir, phase)


## ===================================================================================== ##
# == RUNNING THE BEST PERFORMING MODEL == ##
## ===================================================================================== ##

## ----------------------------------------------------------------------- ##
#  LOADING THE EVALUATION DATASET
## ----------------------------------------------------------------------- ##

# Load the evaluation datasets to test the model on:
custom_path = args.data_path_eval
example_data_root = get_data_root(custom_path)

#Default values for testing and training (if no dataset info file is available)
n_eval = (
    int(getattr(args, "n_sample_eval_samples", 0))
    if eval_data_type == "sample"
    else 10**9
)
#Use the dataset info to extract the relevant parameters for embedding and training set up 
target_name = f"dataset_info_{args.eval_res}_{args.conmul}.txt"
dataset_info = os.path.join(args.data_path_eval, target_name)

# Extract information from the dataset 
dx, dy, delta_t, time_frames, num_timesteps = info_extraction(dataset_info)

# Setting the number of timesteps seen during the training phase
print(f'Total number of frames in evaluation set: {num_timesteps}')
#============ fixed======================
#n_eval = num_timesteps
#=================================

print(f'eval_res: {args.eval_res}')
# Load the dataset (using the new loading function defined for the dataset): 
# No batching of the evaluation dataset (i.e use a single batch of the full dataset)
eval_loader, data_processor = load_2D_AP_eval(
        n_eval=n_eval, batch_size= 5, data_root=example_data_root, eval_resolution = args.eval_res, dataset_name = '2D_AP', cm_eval = args.conmul,
         encode_input = False, encode_output = False,
         start_index = getattr(args, "start_index", 0),
)

data_processor = data_processor.to(device)
full_dataset = eval_loader.dataset
print('full dataset')
print(full_dataset)


# Establish the embedding and boundaries for the inclusion of the time channel (Adapting the spatial embeddings for the grid resolutions):
print('\n### --------- ###\n')
print(f"Mesh size: {args.mesh_size} cm x {args.mesh_size} cm")
time_boundary = float(delta_t) * (float(time_frames) -1 )
print(f"Sample Time Boundary: {time_frames} frames with spacing {delta_t} ms =  {time_boundary} ms ")
embedding = GridEmbeddingND(in_channels=args.ch, dim=3, grid_boundaries=[[0,float(time_boundary)], [0, args.mesh_size], [0, args.mesh_size]])
print("Embedding Grid Boundaries =", embedding.grid_boundaries)


## ----------------------------------------------------------------------- ##
#  LOADING THE PRE-TRAINED MODEL
## ----------------------------------------------------------------------- ##
print('--------------------------------------------------')
# best_model_path = os.path.join(args.eval_output_dir, f'best_model_state_dict.pt')
print(f"LOADING PERFORMING MODEL FROM: {args.model_path_eval}")
print('--------------------------------------------------')
json_log_path_best=stage_path("Best_Model_Results.json")
sys.stdout.flush()


#embedding = GridEmbeddingND(in_channels=args.ch, dim=3, grid_boundaries=[[0,float(args.moving_window)], [0, H], [0, W]])
#print("grid_boundaries =", embedding.grid_boundaries)
model = FNO(n_modes=(8, 16, 16),
             in_channels=args.ch, 
             out_channels=args.ch,
             hidden_channels=32, 
             projection_channel_ratio=2,
             positional_embedding = embedding)
model = model.to(device)
model.load_state_dict(torch.load(args.model_path_eval, map_location=device, weights_only=False))
model.eval()


## ================================================================================= ##

## ----------------------------------------------------------------------- ##
#  RUNNING THE SINGLE FORWARD PASS
## ----------------------------------------------------------------------- ##

## ----------------------------------------------------------------------- ##
# DEFINING THE LOSSES 
## ----------------------------------------------------------------------- ##

# data losses 
l2loss = LpLoss(d=2, p=2)
mse = MSELoss()
rmse = RMSELoss()
impulse_rmse = ImpulseRMSELoss(
    impulse_threshold=getattr(args, "impulse_threshold", 0.05),
    eps=getattr(args, "impulse_loss_eps", 1e-8),
)
boundary = BoundaryLoss()
ic = ICLoss()
bcn = BCNeumann()

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


# Establish Global Evaluation losses:
eval_losses = {
        'l2': l2loss,
        'mse': mse,
        'rmse': rmse,
        'impulse_rmse': impulse_rmse,
        'ap_phys': resloss,
        'boundary': boundary,
        'ic': ic,
        'bcn': bcn
    }

# Initialise The logs for losses and outputs: 

eval_logs = {
    'l2': [], 'mse': [], 'rmse': [], 'impulse_rmse': [], 'ap_phys': [],
    'boundary': [], 'ic': [], 'bcn': [],
}
#cell results 
all_x_true = []
all_y_true = []
all_y_pred = []

## ----------------------------------------------------------------------- ##
# RUN THE MODEL ON THE EVALUATION SET
## ----------------------------------------------------------------------- ##

print('\n### RUNNING EVALUATION ###\n')
print('\n### MODEL ###\n', model)

start_time = time.time()  # start timer

# calculate losses on the dataset 
with torch.no_grad():
    for batch in eval_loader:
        batch = data_processor.preprocess(batch, batched=True)
        x = batch['x']
        y_true = batch['y'] 
    
        # Use the best performing model to predict the outcome:
        y_pred = model(x)

        # Compute and store each loss
        # Compute and store each loss with keyword arguments
        eval_logs['l2'].append(l2loss(y_pred=y_pred, y=y_true).item())
        eval_logs['mse'].append(mse(y_pred, y_true).item())
        eval_logs['rmse'].append(rmse(y_pred=y_pred, y=y_true).item())
        eval_logs['impulse_rmse'].append(
            impulse_rmse(y_pred=y_pred, y=y_true).item()
        )
        eval_logs['ap_phys'].append(resloss(x=x, y=y_true, y_pred=y_pred).item())
        eval_logs['boundary'].append(boundary(x=x,y=y_true, y_pred=y_pred).item())
        eval_logs['ic'].append(ic(x=x,y=y_true, y_pred=y_pred).item())
        eval_logs['bcn'].append(bcn(x=x,y=y_true, y_pred=y_pred).item())
        # Collect all outputs for spatial/temporal analysis and plotting later on 
        all_x_true.append(x)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

print("Completed Evaluation run!")

end_time = time.time()  # stop timer
elapsed = end_time - start_time
print(f"\nCompleted Evaluation run in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")



# Collate the evaluation losses from the forward pass of the model
# avg_results = {key: sum(vals)/len(vals) for key, vals in eval_logs.items()}
avg_results = {key: sum(vals)/len(vals) if len(vals) > 0 else 0 for key, vals in eval_logs.items()}


# Concatenate outputs from all of the batches
y_true_full = torch.cat(all_y_true, dim=0)  # Shape: [B, C, T, X, Y]
x_true_full = torch.cat(all_x_true, dim=0)  # Shape: [B, C, T, X, Y]
y_pred_full = torch.cat(all_y_pred, dim=0)  # Same shape

print(f"y_pred_full.shape before reconstruction: {y_pred_full.shape}")


# Reconstruct the full rollout (averaging overlapping windows)
output_steps = y_pred_full.shape[2]
window_mode = infer_window_mode_from_path(args.data_path_eval)
reconstruction_stride = output_steps if window_mode == "block" else 1
print(f"P2P reconstruction window_mode={window_mode}, stride={reconstruction_stride}")
y_true_full = reconstruct_rollout(y_true_full, output_steps, stride=reconstruction_stride)
x_true_full = reconstruct_rollout(x_true_full, output_steps, stride=reconstruction_stride)
y_pred_full = reconstruct_rollout(y_pred_full, output_steps, stride=reconstruction_stride)


print(f"Reconstructed rollout_true shape: {y_true_full.shape}")
print(f"Reconstructed rollout_input shape: {x_true_full.shape}")
print(f"Reconstructed rollout_pred shape: {y_pred_full.shape}")


## ================================================================================= ##

## ----------------------------------------------------------------------- ##
#  COMPARING THE SIMULATOR OUTPUT AND GROUND TRUTH DATA
## ----------------------------------------------------------------------- ##

signed_error = y_pred_full - y_true_full
abs_error = torch.abs(signed_error)
impulse_mask = y_true_full[0] > getattr(args, "impulse_threshold", 0.05)
voltage_error = signed_error[0]
impulse_voltage_error = voltage_error[impulse_mask]
overall_impulse_rmse = (
    torch.sqrt((impulse_voltage_error ** 2).mean()).item()
    if impulse_voltage_error.numel() else 0.0
)
avg_results["impulse_rmse_reconstructed"] = overall_impulse_rmse
print("Impulse RMSE:", overall_impulse_rmse)
mse_per_timestep_list = []
rmse_per_timestep_list = []
# Initialize analysis results

analysis = {}

# Error calculation per channel
num_channels = y_true_full.shape[0]
for c in range(num_channels):
    channel_name = f"Channel {c} (0 = Voltage, 1 = Recovery Current)"

    # Errors for this channel
    channel_signed_error = signed_error[c, :, :, :]  # [T, X, Y]
    channel_abs_error = abs_error[c, :, :, :]        # [T, X, Y]

    # Metric 1: Mean Absolute Error
    avg_mae = channel_abs_error.mean().item()

    # Metric 2: Mean Absolute Signed Error (bias)
    mean_abs_signed_error = channel_signed_error.abs().mean().item()

    # Metric 3: Max Absolute Signed Error
    max_abs_signed_error = channel_signed_error.abs().max().item()

    # Metric 4: Per-cell average error
    error_per_cell = channel_abs_error.mean(dim=(0, 1))  # [X, Y]
    avg_cell = error_per_cell.mean().item()

    # Metric 5: Top 10 worst cells
    top10 = torch.topk(error_per_cell.flatten(), k=10)
    avg_top10_cell = top10.values.mean().item()

   # Metric 6: Percentage of individual errors above average MAE
    total_elements = channel_abs_error.numel()
    threshold_exceed_count = (channel_abs_error > avg_mae).sum().item()
    threshold_exceed_pct = (threshold_exceed_count / total_elements) * 100

    # Metric 7: Percentage of cells with average error above average cell error
    total_cells = error_per_cell.numel()
    num_cells_above_threshold = (error_per_cell > avg_cell).sum().item()
    num_cells_above_threshold_pct = (num_cells_above_threshold / total_cells) * 100

    # Metric 8: Worst-case (max abs error)
    max_error = channel_abs_error.max().item()
    max_idx = torch.nonzero(channel_abs_error == max_error, as_tuple=False)[0]
    worst_case = {
        #"Batch": int(max_idx[0]),
        "Timestep": int(max_idx[0]),
        "Cell [x, y]": [int(max_idx[1]), int(max_idx[2])],
        "Absolute Error": max_error
    }

    # Metric 9: Best-case (min abs error)
    min_error = channel_abs_error.min().item()
    min_idx = torch.nonzero(channel_abs_error == min_error, as_tuple=False)[0]
    best_case = {
        #"Batch": int(min_idx[0]),
        "Timestep": int(min_idx[0]),
        "Cell [x, y]": [int(min_idx[1]), int(min_idx[2])],
        "Absolute Error": min_error
    }

    # Metric 10: Best/Worst cell overall across time
    error_map = channel_abs_error.mean(dim=0)  # [X, Y]
    best_overall_error = error_map.min().item()
    worst_overall_error = error_map.max().item()

    best_flat = torch.argmin(error_map)
    worst_flat = torch.argmax(error_map)

    best_x, best_y = divmod(best_flat.item(), error_map.shape[1])
    worst_x, worst_y = divmod(worst_flat.item(), error_map.shape[1])

    best_worst_over_time = {
        "Best Cell [x, y]": [best_x, best_y],
        "Best Cell Avg Error": best_overall_error,
        "Worst Cell [x, y]": [worst_x, worst_y],
        "Worst Cell Avg Error": worst_overall_error
    }

    # Metric 11: MSE across batch and cells for each timestep
    mse_per_timestep = (channel_signed_error ** 2).mean(dim= (1, 2))  # [T]
    mse_per_timestep_list.append(mse_per_timestep.cpu().numpy().tolist())

    # Metric 12: RMSE across batch and cells for each timestep AND As an overall metric
    # Compute RMSE per timestep
    rmse_per_timestep = torch.sqrt((channel_signed_error ** 2).mean(dim=(1, 2)))  # [T]
    rmse_per_timestep_list.append(rmse_per_timestep.cpu().numpy().tolist())

    # Compute overall RMSE over the whole evaluation set
    overall_rmse = torch.sqrt((channel_signed_error ** 2).mean())
    print("Overall RMSE:", overall_rmse.item())
        
    
    # Store everything
    analysis[channel_name] = {
        "Avg Mean Absolute Error": avg_mae,
        "Average Cell Error": avg_cell,
        "Average Error of Top 10 Worst Cells": avg_top10_cell,
        "Mean Signed Error (Bias)": mean_abs_signed_error,
        "Max Signed Error": max_abs_signed_error,
        "Pct Errors > Avg MAE": threshold_exceed_pct,
        "Pct Cells with Avg Error > Avg Cell Error": num_cells_above_threshold_pct,
        "Worst Case Results": worst_case,
        "Best Case Results": best_case,
        "Best/Worst Cells Over Time": best_worst_over_time,
        "Overall RMSE:" : overall_rmse.item(),
        **({"Impulse RMSE": overall_impulse_rmse} if c == 0 else {}),
        #"Per-Timestep MSE": mse_per_timestep_list
    }


# Collate errors from the model:
avg_results = {}
avg_results.update(analysis)
avg_results["impulse_rmse"] = overall_impulse_rmse

with open(json_log_path_best, "w") as f:
    json.dump(avg_results, f, indent=2)

print('--------------------------------------------------')
print("MODEL EVALUATION RESULTS:")
for ch_key, ch_result in analysis.items():
    print(f"\n{ch_key}:")
    for k, v in ch_result.items():
        print(f"  {k}: {v}")
print('--------------------------------------------------')


## ======================================================================= ##
# == VISUALISATION ANALYSIS == ##
## ======================================================================= ##

# Change device to cpu for plotting
device = 'cpu'
if args.ch == 2:
    channels = [0,1]
    channel_names = ["Voltage (V)", "Recovery (W)"]
elif args.ch == 1:
    channels = [0]
    channel_names = ["Voltage (V)"]


## ----------------------------------------------------------------------- ##
# TIME SERIES PLOTTING OF V AND W FOR BEST-WORST PERFORMING CELLS
## ----------------------------------------------------------------------- ##

# extract the cells from the earlier error calculations: 
cells = [
    (best_x, best_y),
    (worst_x, worst_y)
]

# Remove batch dimension to ease visualisation
input = x_true_full
gt = y_true_full 
pred = y_pred_full

cell_data = {
    "model_name": args.data_path,
    "channels": {
        "0": "Voltage",
        "1": "Recovery"
    },
    "cells": {}  # Will store results keyed by cell tuple
}


pred_len = pred.shape[1]
x_pred = np.arange(pred_len)

# Ground truth full x-axis
x_gt_full = np.arange(gt.shape[1])



# Save the time series for the cells (later used for model comparison plots)
for ch in [0, 1]:
    for cell_x, cell_y in cells:
        
        cell_key = f"{cell_x}_{cell_y}"
        
        gt_series = gt[ch, :, cell_x, cell_y].detach().cpu().numpy().tolist()
        pred_series = pred[ch, :, cell_x, cell_y].detach().cpu().numpy().tolist()
    
        # Initialize if cell not already present
        if cell_key not in cell_data["cells"]:
            cell_data["cells"][cell_key] = {}

        cell_data["cells"][cell_key][f"{ch}"] = {
            "ground_truth": gt_series,
            "prediction": pred_series
        }

# Save to JSON
save_path = stage_path("best_worst_cell_time_series.json")
with open(save_path, "w") as f:
    json.dump(cell_data, f, indent=2)


# Plot the full time series (Voltage only)
titles = ["Best Cell", "Worst Cell"]

# Plot the full time series (Voltage only)
titles = ["Best Cell", "Worst Cell"]

if args.seen != 0.0:
    seen_cutoff = 152   # frames 0–152 are training (seen)
    total_frames = 200  # total frames
    line_label = "Time Horizon (training)"

# Create a figure with 2 vertical subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for i, (cell_x, cell_y) in enumerate(cells):
    ax = axes[i]

    # Full ground truth (plot the entire dataset)
    gt_full_series = gt[0, :, cell_x, cell_y].detach().cpu().numpy()
    # Prediction rollout (starting from the random initial window)
    pred_series = pred[0, :, cell_x, cell_y].detach().cpu().numpy()

    # Plot full GT and prediction rollout
    gt_line, = ax.plot(x_gt_full, gt_full_series, label='Ground Truth', color='blue', linewidth=2)
    pred_line, = ax.plot(x_gt_full, pred_series, label='Prediction', color='red', linewidth=2, linestyle='--')

    ax.set_title(f"{titles[i]} - Cell ({cell_x}, {cell_y})")
    ax.set_ylabel("Voltage (AU)")
    ax.grid(False)

    # Optional: mark unseen horizon
    if args.seen != 0.0:
        ax.axvline(x=seen_cutoff, color='black', linestyle=':', linewidth=2)

# Add common x-label
axes[-1].set_xlabel("Time step (ms)")
for ax in axes:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x * int(delta_t))))

# --- Single Legend and Layout Adjustment ---
# Create a single shared legend on the right of the top title
lines = [gt_line, pred_line]
labels = [l.get_label() for l in lines]

# Adjust layout before adding legend so it sits flush with the title
plt.tight_layout(rect=[0, 0, 1, 0.95])  

# Place legend manually aligned with the top title
# We use figure coordinates (0 to 1 range)
fig.legend(
    lines, labels,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.955),  
    ncol=2,                        
    frameon=False,
    handlelength=3,
    columnspacing=1.5,
    fontsize=10
)

# Save and close
save_path = stage_path('Best_Worst_Voltage_Cell_Plots.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)


## ----------------------------------------------------------------------- ##
# TIME SERIES PLOTTING OF V AND W FOR SELECTED CELLS FOR COMPARISON 
## ----------------------------------------------------------------------- ##

# Pick specific grid cells to monitor in the mesh - the centre, the bottom left and the top right cell. 

centre = int(int(args.eval_res) / 2)
quarter = int(int(args.eval_res) / 4)
three_quarter = int(int(args.eval_res) * (3/4))

cells = [[quarter, quarter], [centre, centre], [three_quarter, three_quarter]]

print(f"SELECTED CELLS: {cells}")


cell_data = {
    "model_name": args.data_path,
    "channels": {
        "0": "Voltage",
        "1": "Recovery"
    },
    "cells": {}  # Will store results keyed by cell tuple
}

# Save the time series for the cells (later used for model comparison)
for ch in [0, 1]:
    for cell_x, cell_y in cells:
        cell_key = f"{cell_x}_{cell_y}"

        gt_series = gt[ch, :, cell_x, cell_y].tolist()
        pred_series = pred[ch, :, cell_x, cell_y].tolist()

        # Initialize if cell not already present
        if cell_key not in cell_data["cells"]:
            cell_data["cells"][cell_key] = {}

        cell_data["cells"][cell_key][f"{ch}"] = {
            "ground_truth": gt_series,
            "prediction": pred_series
        }

# Save to JSON
save_path = stage_path("cell_time_series.json")
with open(save_path, "w") as f:
    json.dump(cell_data, f, indent=2)


# Plot the full time series (Voltage only)
if args.seen != 0.0:
    # Mark where the initial input frames end in the sequential roll-out simulation 
    x_line = args.seen  # the x-coordinate where you want the line
    line_label = "Unseen Time Horizon"  # text label for the line


# Create a figure with 3 vertical subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, (cell_x, cell_y) in enumerate(cells):
    
    ax = axes[i]

    # Full ground truth (plot the entire dataset)
    gt_full_series = gt[0, :, cell_x, cell_y].detach().cpu().numpy()
    # Prediction rollout (starting from the random initial window)
    pred_series = pred[0, :, cell_x, cell_y].detach().cpu().numpy()

    # Plot full GT and prediction rollout
    ax.plot(x_gt_full, gt_full_series, label='Ground Truth (full)', color='blue', linewidth=2)
    ax.plot(x_gt_full, pred_series, label='Prediction', color='red', linewidth=2, linestyle='--')
    
    ax.set_title(f"Voltage @ - Cell ({cell_x}, {cell_y}) - Channel 0 (Voltage)")
    ax.set_ylabel("Voltage (AU)")
    ax.grid(False)

   
    # Optional: mark unseen horizon
    if args.seen != 0.0:
        ax.axvline(x=seen_cutoff, color='black', linestyle=':', linewidth=2)
        ax.text(args.seen + 2, ax.get_ylim()[1]*0.95, '', color='green', rotation=90, verticalalignment='top')
    ax.legend(loc="upper right")


# Common x-label
#axes[-1].set_xlabel("Time step (AU)")
# Convert tick labels from frame index → milliseconds
axes[-1].set_xlabel("Time step (ms)")
for ax in axes:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x * int(delta_t))))
plt.tight_layout()
save_path = stage_path('Selected_Voltage_Cell_Plots.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)




# ----------------------------------------------------------------------- ##
# SAMPLE PLOTS OF Input - GT, PREDICTION, AND ERROR (each with colorbars)
# ----------------------------------------------------------------------- ##
def plot_selected_frames_with_input(x, gt, pred, save_dir, selected_frames=[0]):
    """
    Plots selected timesteps of INPUT, GT, PRED, and ERROR side-by-side.
    Each subplot (row,col) has its own colorbar next to it.
    """
    import os, numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Convert tensors to numpy
    voltage_in = x[0].detach().cpu().numpy()       # [T_in, H, W]
    voltage_gt = gt[0].detach().cpu().numpy()      # [T, H, W]
    voltage_pred = pred[0].detach().cpu().numpy()  # [T, H, W]
    voltage_err = voltage_gt - voltage_pred        # [T, H, W]

    # Color scale limits
    vmin = min(voltage_in.min(), voltage_gt.min(), voltage_pred.min())
    vmax = max(voltage_in.max(), voltage_gt.max(), voltage_pred.max())
    err_abs = np.abs(voltage_err).max()

    num_frames = len(selected_frames)

    # 4 rows: Input, GT, Pred, Error
    fig, axs = plt.subplots(4, num_frames, figsize=(5.5 * num_frames, 14))
    if num_frames == 1:
        axs = np.expand_dims(axs, axis=1)

    row_labels = ["Input", "GT", "Pred", "Error"]

    # Iterate over selected frames (columns)
    for col, frame in enumerate(selected_frames):
        in_frame = voltage_in[frame - int(args.moving_window)]
        gt_frame = voltage_gt[frame]
        pred_frame = voltage_pred[frame]
        err_frame = voltage_err[frame]

        images = [
            axs[0, col].imshow(in_frame, cmap='viridis', vmin=vmin, vmax=vmax),
            axs[1, col].imshow(gt_frame, cmap='viridis', vmin=vmin, vmax=vmax),
            axs[2, col].imshow(pred_frame, cmap='viridis', vmin=vmin, vmax=vmax),
            axs[3, col].imshow(err_frame, cmap='bwr', vmin=-err_abs, vmax=err_abs)
        ]
        titles = [
            f"Input - Timestep {frame - int(args.moving_window)}",
            f"GT - Timestep {frame}",
            f"Pred - Timestep {frame}",
            f"Error - Timestep {frame}"
        ]

        for row in range(4):
            axs[row, col].set_title(titles[row], fontsize=10)
            axs[row, col].axis('off')

            # Add colorbar next to each subplot
            cbar = fig.colorbar(images[row], ax=axs[row, col],
                                fraction=0.046, pad=0.02, orientation='vertical')
            cbar.ax.tick_params(labelsize=7)

    # Add row labels on the left
    for row in range(4):
        axs[row, 0].text(
            -0.4, 0.5, row_labels[row],
            fontsize=12, fontweight='bold',
            rotation='vertical', ha='center', va='center',
            transform=axs[row, 0].transAxes
        )

    plt.subplots_adjust(wspace=0.5, hspace=0.4, top=0.9, bottom=0.05)

    save_path = os.path.join(save_dir, stage_filename("Voltage_Samples_With_Input.png"))
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f" Saved input+GT+prediction visualization to {save_path}")



def plot_selected_frames(gt, pred, save_dir, selected_frames=[0, 50, 100, 150, 199]):
    """
    Plots selected timesteps of GT, prediction, and error side-by-side
    with row labels and adjustable spacing between rows.
    """

    # Convert tensors to numpy arrays
    voltage_gt = gt[0].detach().cpu().numpy()      # [T, H, W]
    voltage_pred = pred[0].detach().cpu().numpy()  # [T, H, W]
    voltage_err = voltage_gt - voltage_pred        # [T, H, W]

    # Consistent color scales
    #vmin = min(voltage_gt.min(), voltage_pred.min())
    #vmax = max(voltage_gt.max(), voltage_pred.max())
    err_abs = np.abs(voltage_err).max()

    # ====== NEW: Fixing colour bars: Comment out later! 
    vmin = 0.0 
    vmax = 1.0 
    # ====== NEW: Fixing colour bars: Comment out later! 
    num_frames = len(selected_frames)

    # Create figure grid: 3 rows (GT, Pred, Err) × N columns (frames)
    # Disable constrained_layout since we will manage spacing manually
    fig, axs = plt.subplots(3, num_frames, figsize=(4*num_frames, 10))

    if num_frames == 1:
        axs = np.expand_dims(axs, axis=1)  # ensure consistent indexing

    #row_titles = ["Ground Truth", "Prediction", "Error (GT - Pred)"]
    short_labels = ["GT", "Pred", "Error"]

    for col, frame in enumerate(selected_frames):
        gt_frame = voltage_gt[frame]
        pred_frame = voltage_pred[frame]
        err_frame = voltage_err[frame]

        # Plot data
        im0 = axs[0, col].imshow(gt_frame, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0, col].set_title(f"Timestep {frame}", fontsize=10)
        im1 = axs[1, col].imshow(pred_frame, cmap='viridis', vmin=vmin, vmax=vmax)
        im2 = axs[2, col].imshow(err_frame, cmap='bwr', vmin=-err_abs, vmax=err_abs)

        for row in range(3):
            axs[row, col].axis('off')

    # Label rows (leftmost column)
    for row in range(3):
        axs[row, 0].text(
            -0.3, 0.5,               # position relative to image
            #f"{short_labels[row]}\n{row_titles[row]}",
            f"{short_labels[row]}",
            fontsize=12,
            fontweight="bold",
            rotation='vertical',
            ha='center',
            va='center',
            transform=axs[row, 0].transAxes
        )

    # Adjust spacing manually
    plt.subplots_adjust(wspace=0.05, hspace=0.35, top=0.9, bottom=0.05)

    # Add colorbars (shared per row)
    for row, im in enumerate([im0, im1, im2]):
        cbar = fig.colorbar(
            im, ax=axs[row, :].ravel().tolist(),
            fraction=0.02, pad=0.04, orientation='vertical'
        )
        cbar.ax.tick_params(labelsize=8)

    # Title
    #fig.suptitle("Selected Voltage Frames: GT, Prediction, and Error", fontsize=16)

    # Save figure
    save_path = os.path.join(save_dir, stage_filename("Voltage_Samples.png"))
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved selected frame visualization to {save_path}")

# Select the samples to plot:
# Select ONLY cells that are beyond the training horizon (post 150)! 
#==================== fixed =================================
# if args.horizon == 500:
#     centre = 350
#     quarter = 275
#     three_quarter = 425
# else: 
#     centre = 180
#     quarter = 160
#     three_quarter = 190
#=================================================================

# Вместо хардкода — вычислять из данных
T = gt.shape[1]  # реальное число кадров
quarter = int(T * 0.25)
centre = int(T * 0.5)
three_quarter = int(T * 0.75)



#centre = int(int(args.horizon) / 2)
#quarter = int(int(args.horizon) / 4)
#three_quarter = int(int(args.horizon) * (3/4))

plot_selected_frames(
    gt=gt,
    pred=pred,
    save_dir=args.eval_output_dir,
    selected_frames = [quarter, centre, three_quarter]

)

plot_selected_frames_with_input(
    x=x_true_full,
    gt=y_true_full,
    pred=y_pred_full,
    save_dir=args.eval_output_dir,
    selected_frames=[quarter, centre, three_quarter]
)



## ----------------------------------------------------------------------- ##
# ANIMATION OF GT, PREDICTION, AND ERROR (OVER ALL AVAILABLE TIMESTEPS) (with colorbars)
## ----------------------------------------------------------------------- ##
# Get averaged GT and prediction (assuming [B, T, H, W] and using batch 0)

# Use the aligned versions for error calculations
if args.random:
    gt = y_true_full
    pred = y_pred_full
    voltage_gt = gt[0, :, :, :].detach().cpu().numpy()      # [T, H, W]
    voltage_pred = pred[0, :, :, :].detach().cpu().numpy()  # [T, H, W]

    voltage_err = voltage_gt - voltage_pred                # [T, H, W]
else: 

    voltage_gt = gt[0, :, :, :].detach().cpu().numpy()      # [T, H, W]
    voltage_pred = pred[0, :, :, :].detach().cpu().numpy()  # [T, H, W]
    voltage_err = voltage_gt - voltage_pred                # [T, H, W]

# Consistent color scales
vmin = min(voltage_gt.min(), voltage_pred.min())
vmax = max(voltage_gt.max(), voltage_pred.max())

# Error should be symmetric around 0
err_abs = np.abs(voltage_err).max()
#errmin = max(voltage_err.min(), voltage_err.min())
#errmax = max(voltage_err.max(), voltage_err.max())


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
gt_ax, pred_ax, err_ax = axs

# Initial images
gt_im = gt_ax.imshow(voltage_gt[0], cmap='viridis', vmin=vmin, vmax=vmax)
pred_im = pred_ax.imshow(voltage_pred[0], cmap='viridis', vmin=vmin, vmax=vmax)
err_im = err_ax.imshow(voltage_err[0], cmap='bwr', vmin=-err_abs, vmax=err_abs)

# Titles
gt_ax.set_title("Ground Truth - Voltage")
pred_ax.set_title("Prediction - Voltage")
err_ax.set_title("Error (GT - Pred)")

# Add colorbars
fig.colorbar(gt_im, ax=gt_ax, fraction=0.046, pad=0.04)
fig.colorbar(pred_im, ax=pred_ax, fraction=0.046, pad=0.04)
fig.colorbar(err_im, ax=err_ax, fraction=0.046, pad=0.04)

for ax in axs:
    ax.axis('off')

fig.suptitle("Voltage Channel Evolution Over Time", fontsize=14)

# Update function
def update(frame):
    gt_im.set_array(voltage_gt[frame])
    pred_im.set_array(voltage_pred[frame])
    err_im.set_array(voltage_err[frame])
    fig.suptitle(f"Voltage Channel Evolution Over Time\nTimestep {frame}", fontsize=14)
    return gt_im, pred_im, err_im

# Animation (removed blit=True so all frames refresh)
ani = animation.FuncAnimation(
    fig, update, frames=range(voltage_gt.shape[0]), interval=100, blit=False
)

# Save as GIF
gif_path = stage_path("Voltage_Evolution_GT_Pred_Error.gif")
ani.save(gif_path, writer=PillowWriter(fps=10))
plt.close()
'''

## ----------------------------------------------------------------------- ##
# MSE PLOTTING OF V AND W OVER ALL TIMESTEPS 
## ----------------------------------------------------------------------- ##

# Save the mse results for model comparison
mse_data = {
    "model_name": args.data_path,  
    "timesteps": list(range(len(mse_per_timestep_list))),
    "mse": {
        "Voltage": mse_per_timestep_list[0],
        "Recovery": mse_per_timestep_list[1]
    }
}
save_path = stage_path("mse_per_timestep.json")
with open(save_path, "w") as f:
    json.dump(mse_data, f, indent=2)


# Save the rmse results for model comparison
rmse_data = {
    "model_name": args.data_path,  
    "timesteps": list(range(len(rmse_per_timestep_list))),
    "rmse": {
        "Voltage": rmse_per_timestep_list[0],
        "Recovery": rmse_per_timestep_list[1]
    }
}
save_path = stage_path("rmse_per_timestep.json")
with open(save_path, "w") as f:
    json.dump(rmse_data, f, indent=2)

#print(mse_per_timestep_list)

#mse_per_timestep_array = np.array(mse_per_timestep_list)
mse_per_timestep_array = np.stack(mse_per_timestep_list)    
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
#print(mse_per_timestep_array.shape)
num_timesteps = np.arange(mse_per_timestep_array.shape[1])
for ch in [0, 1]:
    ax = axes[ch]
    ax.plot(num_timesteps, mse_per_timestep_array[ch], color='black', linewidth=2)
    ax.set_title(f'{channel_names[ch]}: Mean Square Error Over Time', fontsize=12)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Squared Error')
    ax.grid(False)
    #ax.set_ylim(0, 0.25)

plt.tight_layout()
save_path = stage_path('MSE_Evolution.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)

rmse_per_timestep_array = np.stack(rmse_per_timestep_list)    
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
#print(mse_per_timestep_array.shape)
num_timesteps = np.arange(rmse_per_timestep_array.shape[1])
for ch in [0, 1]:
    ax = axes[ch]
    ax.plot(num_timesteps, mse_per_timestep_array[ch], color='black', linewidth=2)
    ax.set_title(f'{channel_names[ch]}: Root Mean Square Error Over Time', fontsize=12)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Root Mean Squared Error')
    ax.grid(False)
    #ax.set_ylim(0, 0.25)

plt.tight_layout()
save_path = stage_path('RMSE_Evolution.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)

'''
