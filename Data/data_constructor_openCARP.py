'''
Transform the EP PINNs data from its matlab form into pytorch tensors that can be used in operator training:
This version loads up a full sample of many simulation runs to form the dataset:
    - Load the mat data, for each of the N runs:
    - extract the voltage and recovery current maps, the time dimension, and flatten to pytorch tensors to form a dataset of shape: (1, ch, T, X, Y)
    - Combine all of the tensors to form the dataset (N, ch, T, X, Y)
    - Split into a testing and training set and then split the results into input-output pairs according to a moving time window.
    (option: given the first T/2 data - you need to predict the next T/2 timesteps) 
    - Save the metadata of the training-testing tensors to a log file
note: This version of the code presumes that both channels are present in the data (V, W) and saves them both in the dataset. 
'''

import scipy.io
from scipy.io import loadmat
import torch
import argparse
import shutil, os
import tempfile
import sys
import sklearn 
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import traceback
try:
    import h5py
except ImportError:
    h5py = None
import re
#import concurrent.futures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder-name', dest='folder_name', required = True, type = str, help='Data folder containing the mat files to transform to training tensors')
    parser.add_argument('-o', '--output', dest='output', type =str, default = 'dataset_info', help = 'Output file to log the construction information of the input data. Default = %(default)s')
    parser.add_argument('-r', '--resolution', dest = 'resolution', default = 100, type = int, help='Grid resolution of the data (res x res). Default is %(default)s')
    parser.add_argument('-tr', '--train', dest='train', default = 0.8, type = float, help='Fraction of data to use in training. Default is %(default)s')
    parser.add_argument('-tl', '--time_limit', dest='time_limit', type = int,  default = 500, help='Number of frames to keep in construction of training data. Default is the number available in the struct.')
    parser.add_argument('-ts', '--timestep', dest='timestep', default = 1.0, type = int, help = "Timestep resolution for simulation data. Default is %(default)s")
    parser.add_argument('-iw', '--input-window', dest='input_window', default = 5, type = int, help='Window for input timesteps. Default is %(default)s')
    parser.add_argument('-wm', '--window-mode', dest='window_mode', default='overlapping',
                        choices=['overlapping', 'block'],
                        help='Window generation mode: overlapping uses stride 1; block uses stride=input_window.')
    parser.add_argument('-rnd', '--random-split', dest='random_split', action='store_true', help='Use random train/test split')
    parser.add_argument('-n', '--norm', dest='norm', action='store_true', help='Normalize V and W to [0,1]')
    parser.add_argument('-ch', '--channels', dest='ch', type=int, default = 2, help='Number of channels in the dataset. Default is %(default)s - Voltage and Recovery. Pass 2 for Voltage only')
    parser.add_argument('-co', '--coords', dest ='coords', type = int, default = 0, help = 'Flag to save the coordinate structure of the data as a tensor. Default is %(default)s, no saving. Pass 1 to save')
    parser.add_argument('-m', '--mat', dest = 'mat', action = 'store_true', help = 'Flag to construct testing/training data from matlab inputs. Default to construct from full tensors.')
    args = parser.parse_args()


## ======================================================================= ##
## DEFINING THE FUNCTIONS FOR BUILDING THE TENSOR DATASETS ## 
## ======================================================================= ##

# Define a class for logging the outputs of the data generation process
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

def parse_filename_mat(filename):
    """
    Parse filename of the form:
    2D_AP_{stable/chaotic}_{res}um_{timeframe}ms_{conmul}_{scaleres}.mat
    Returns (sim_type, conmul).
    """
    basename = os.path.basename(filename)
    match = re.match(r"2D_AP_(\w+)_\d+um_\d+ms_([\d\.]+)_([\d\.]+)\.mat", basename)
    if not match:
        raise ValueError(f"Filename {basename} does not match expected pattern.")
    sim_type, conmul, scaleres = match.groups()
    return sim_type, float(conmul), float(scaleres)

def parse_filename_pt(filename):
    """
    Parse filename of the form:
    2D_AP_{stable/chaotic}_{res}um_{timeframe}ms_{conmul}_x{scaleres}_t{tres}.pt
    Returns (sim_type, conmul, scaleres).
    """
    basename = os.path.basename(filename)
    match = re.match(r"2D_AP_(\w+)_\d+um_\d+ms_([\d\.]+)_x([\d\.]+)_t([\d\.]+)\.pt", basename)
    if not match:
        raise ValueError(f"Filename {basename} does not match expected pattern.")
    sim_type, conmul, scaleres, scalet = match.groups()
    print(f"Simulation Info: simtype = {sim_type}, Conductivity Multipler = {conmul}, resolution abstraction factor = x{scaleres}, time resolution scaling = t{scalet}")
    return sim_type, float(conmul), float(scaleres), float(scalet)

# Function for running in parallel:
def safe_load(path):
    if args.mat:
        try:
            return load_single_sample_mat(path)
        except Exception as e:
            print(f"Skipping {os.path.basename(path)} due to error: {e}")
            return None
    else:
        try:
            return load_single_sample_pt(path)
        except Exception as e:
            print(f"Skipping {os.path.basename(path)} due to error: {e}")
            return None

    
# Loading a single matlab sample from the parent data folder and extracting the data channels 
def load_single_sample_mat(filepath, time_limit=None):
    if h5py is None:
        raise ImportError("h5py is required to load .mat datasets. Install h5py or use .pt input data.")

    with h5py.File(filepath, 'r') as f:
        vmw = f['VmW']  # MATLAB struct stored as an HDF5 group

        voltage = np.array(vmw['VSav'])       # shape: (x, y, t)
        recovery = np.array(vmw['WSav'])

        print(voltage.shape)
        print(recovery.shape)
        t_vals = np.array(vmw['t']).squeeze()
        print(t_vals.shape)
    
        x_vals = np.array(vmw['x']).squeeze()
        #y_vals = np.array(vmw['y']).squeeze()

    # MATLAB HDF5 stores data in column-major order (Fortran-style), so we may need to transpose
    voltage = np.transpose(voltage)  
    recovery = np.transpose(recovery)
    timesteps = np.transpose(t_vals)

    # Ensure that the timesteps are equal for the voltage and recovery channels 
    t_v = voltage.shape[2]
    t_w = recovery.shape[2]
    if t_v != t_w:
        raise ValueError("Timestep mismatch for V and W")
    
    # If T is given, slice the first T timesteps
    if time_limit is not None:
        T = time_limit
        if T > t_v:
            raise ValueError(f"Requested T={T}, but only {t_v} timesteps available.")
        voltage = voltage[:, :, :T]
        recovery = recovery[:, :, :T]
        timesteps = timesteps[:T]
        t_v = T  # update count

    x_v = voltage.shape[0]
    print(f'Grid_size: {x_v}')

    x_res = np.mean(np.diff(x_vals))
    print(f'Grid_resolution: {x_res} cm')

    # Determine the timestep resolution in ms:
    t_res = np.mean(np.diff(t_vals))
    print(f"Timestep resolution: {t_res} ms")

    # Determine the full time rollout in ms:
    full_rollout = (t_v -1) * t_res 
    print(f"Full simulation time period: {full_rollout} ms")

    data = np.stack([voltage, recovery], axis=0)  # (2, x, y, t)
    data = np.transpose(data, (0, 3, 1, 2))       # (2, t, x, y)
    tensor = torch.tensor(data, dtype=torch.float32)
    return tensor, t_v, t_res, x_v, x_res 

def load_single_sample_pt(filepath, coords_path=None, time_resolution_ms=None, time_limit=None):
    """
    Load a preconstructed PyTorch tensor (2-channel V/W).
    Optionally load coordinate info and limit time window.

    Args:
        filepath (str): Path to .pt file containing the tensor (2, X, Y, T) or (2, T, X, Y).
        coords_path (str, optional): Path to .npy file of normalized coordinates (for x/y scaling info).
        time_resolution_ms (float, optional): Time step between frames, for logging.
        time_limit (int, optional): Number of timesteps to load (truncate in time).
    
    Returns:
        tensor (torch.Tensor): Loaded tensor of shape (2, T, X, Y).
        metadata (dict): Dictionary with simulation metadata.
    """

    # Load tensor
    tensor = torch.load(filepath, map_location='cpu')
    print(f"Loaded tensor from {filepath}, shape = {tuple(tensor.shape)}")

    # Handle both channel-first or time-first versions
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor of shape (2, X, Y, T) or (2, T, X, Y), got {tensor.shape}")

    # Ensure format is (2, T, X, Y)
    if tensor.shape[1] <= 4 and tensor.shape[0] > 4:
        # tensor is probably (2, X, Y, T) → rearrange
        tensor = tensor.permute(0, 3, 1, 2)  # (2, T, X, Y)
        print("Permuted tensor to shape (2, T, X, Y)")

    # Check channel count
    if tensor.shape[0] != 2:
        raise ValueError(f"Expected 2 channels (V, W), found {tensor.shape[0]}")

    # Truncate time if requested
    total_timesteps = tensor.shape[1]
    if time_limit is not None:
        if time_limit <= total_timesteps:
            tensor = tensor[:, :time_limit, :, :]
            print(f"Truncated to first {time_limit} timesteps.")
        else:
            print(f"Requested T={time_limit}, but only {total_timesteps} timesteps available. Saving availabel timesteps")

    # Extract spatial dimensions
    _, T, X, Y = tensor.shape

    # Optional coordinate information
    x_res = y_res = None
    if coords_path is not None and os.path.exists(coords_path):
        coords = np.load(coords_path)
        x_vals = np.unique(coords[:, 0])
        y_vals = np.unique(coords[:, 1])
        x_res = float(np.mean(np.diff(x_vals))) if len(x_vals) > 1 else None
        y_res = float(np.mean(np.diff(y_vals))) if len(y_vals) > 1 else None
        print(f"Loaded normalized coordinates: x_res={x_res:.4f}, y_res={y_res:.4f}")

    # Temporal information
    t_res = time_resolution_ms if time_resolution_ms is not None else None
    full_duration = (T - 1) * t_res if t_res is not None else None
    
    # Collect metadata
    metadata = {
        "timesteps": T,
        "x_dim": X,
        "y_dim": Y,
        "x_res": x_res,
        "y_res": y_res,
        "t_res_ms": t_res,
        "duration_ms": full_duration,
        "path": filepath
    }

    if x_res == None:
        x_res = 10/X 

    print(f'Grid_size: {X}')
    print(f'Grid_resolution: {x_res} cm')
    print(f"Timestep resolution: {t_res} ms")
    print(f"Full simulation time period: {full_duration} ms")

    print("Metadata:",metadata)

    return tensor.float(), T, t_res, X, x_res 


def create_io_pairs(dataset, input_steps, window_mode='overlapping'):
    print(dataset.shape)
    C, T, X, Y = dataset.shape
    output_steps = input_steps
    if window_mode == 'overlapping':
        window_stride = 1
    elif window_mode == 'block':
        window_stride = input_steps
    else:
        raise ValueError("window_mode must be either 'overlapping' or 'block'")

    inputs_all, outputs_all, times_all = [], [], []

    # The farthest point we can start is when we still have enough steps for input + output
    sample = dataset
    for t in range(0, T - (input_steps + output_steps) + 1, window_stride):
        x = sample[:, t:t + input_steps, :, :]
        y = sample[:, t + input_steps:t + input_steps + output_steps, :, :]
        inputs_all.append(x)
        outputs_all.append(y)
        # store the range of time indices used for this window
        times_all.append(list(range(t, t + input_steps + output_steps)))

    return torch.stack(inputs_all), torch.stack(outputs_all), times_all

def create_io_pairs_nonoverlapping(dataset, input_steps, output_steps):
    """
    Create non-overlapping input-output pairs from a dataset tensor.
    dataset: Tensor of shape (C, T, X, Y)
    input_steps: number of time steps in each input
    output_steps: number of time steps in each output
    """
    print(dataset.shape)
    C, T, X, Y = dataset.shape
    inputs_all, outputs_all, times_all = [], [], []


    # Move forward by (input_steps + output_steps) each time
    step_size = input_steps + output_steps
    for t in range(0, T - step_size + 1, step_size):
        x = dataset[:, t : t + input_steps, :, :]
        y = dataset[:, t + input_steps : t + input_steps + output_steps, :, :]
        inputs_all.append(x)
        outputs_all.append(y)
        # store the range of time indices used for this window
        times_all.append(list(range(t, t + input_steps + output_steps)))

    return torch.stack(inputs_all), torch.stack(outputs_all), times_all

def block_split(inputs, outputs, times_all, test_fraction=0.2, n_blocks=None, seed=42):
    """
    Temporally contiguous, leakage-free split for sequential datasets.
    Works directly with your (inputs, outputs, times_all) structure.
    """
    np.random.seed(seed)
    num_windows = len(inputs)

    # Auto-select block count if not specified (≈10 for 200-frame trajectories)
    if n_blocks is None:
        n_blocks = max(5, min(10, num_windows // 20))

    # Divide the full trajectory into contiguous blocks of window indices
    blocks = np.array_split(np.arange(num_windows), n_blocks)

    # Randomly choose some blocks for testing
    n_test_blocks = max(1, int(round(n_blocks * test_fraction)))
    test_block_ids = np.random.choice(n_blocks, n_test_blocks, replace=False)

    # Gather indices
    train_idx, test_idx = [], []
    for i, b in enumerate(blocks):
        if i in test_block_ids:
            test_idx.extend(b)
        else:
            train_idx.extend(b)

    # --- Ensure indices are plain Python lists ---
    train_idx = list(map(int, train_idx))
    test_idx  = list(map(int, test_idx))

    # --- Split the data ---
    input_train  = inputs[train_idx]
    output_train = outputs[train_idx]
    input_test   = inputs[test_idx]
    output_test  = outputs[test_idx]

    # `times_all` is usually a list of lists → use list comprehension instead of tensor indexing
    times_train = [times_all[i] for i in train_idx]
    times_test  = [times_all[i] for i in test_idx]

    # --- Leakage check ---
    train_times = set(t for w in times_train for t in w)
    test_times  = set(t for w in times_test for t in w)
    overlap = train_times.intersection(test_times)

    print("\n Temporal block split summary:")
    print(f" - Total windows: {num_windows}")
    print(f" - Blocks: {n_blocks}")
    print(f" - Test blocks: {sorted(test_block_ids.tolist())}")
    print(f" - Train windows: {len(train_idx)}, Test windows: {len(test_idx)}")
    print(f" - Overlap in time indices: {len(overlap)} (should be 0)\n")

    return input_train, output_train, times_train, input_test, output_test, times_test


def sequential_three_way_split(inputs, outputs, times_all, train_fraction, test_fraction,
                               eval_fraction, buffer_size):
    """Split window samples into sequential train, test and eval partitions."""
    total = inputs.shape[0]
    train_requested = int(train_fraction * total)
    test_requested = int(test_fraction * total)
    eval_requested = int(eval_fraction * total)

    train_end = min(train_requested, total)
    test_start = min(train_end + buffer_size, total)
    test_end = min(test_start + test_requested, total)
    eval_start = min(test_end + buffer_size, total)
    eval_end = min(eval_start + eval_requested, total)

    ranges = {
        "train": range(0, train_end),
        "test": range(test_start, test_end),
        "eval": range(eval_start, eval_end),
    }

    def select(index_range):
        indices = list(index_range)
        return inputs[indices], outputs[indices], [times_all[i] for i in indices]

    train = select(ranges["train"])
    test = select(ranges["test"])
    evaluation = select(ranges["eval"])
    used = train_end + (test_end - test_start) + (eval_end - eval_start)
    print("Sequential train/test/eval split:")
    print(f" - Requested fractions: train={train_fraction}, test={test_fraction}, eval={eval_fraction}")
    print(f" - Total window samples: {total}")
    print(f" - Actual samples: train={len(train[0])}, test={len(test[0])}, eval={len(evaluation[0])}")
    print(f" - Buffer size between partitions: {buffer_size}")
    print(f" - Unassigned samples (including buffers and tail): {total - used}")
    return train, test, evaluation


## SAVING THE DATASETS FOR OPERATOR TRAINING AND EVALUATION ##

def save_dataset_pair(folder, filename, x_tensor, y_tensor):
    save_path = os.path.join(folder, filename)
    torch.save({'x': x_tensor, 'y': y_tensor}, save_path)
    if 'train' in filename: 
        print(f"Training data shapes:", {x_tensor.shape}, {y_tensor.shape})
    elif 'test' in filename:
        print(f"Testing data shapes:", {x_tensor.shape}, {y_tensor.shape})
    elif 'eval' in filename:
        print(f"Evaluation data shapes:", {x_tensor.shape}, {y_tensor.shape})


def save_full_dataset(folder, filename, full_tensor):
    save_path = os.path.join(folder, filename)
    torch.save(full_tensor, save_path)
    print(f"Evaluation data shape:", {full_tensor.shape})


_saved_signal_previews = set()
_saved_gif_previews = set()


def save_voltage_signal_preview(folder, tensor, t_res, resolution, conmul, set_name=None):
    """Save a single-cell voltage preview next to constructed training/eval tensors."""
    preview_key = (os.path.abspath(folder), set_name or "full")
    if preview_key in _saved_signal_previews:
        return
    _saved_signal_previews.add(preview_key)

    data = tensor[0].detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor[0])
    T, X, Y = data.shape
    cell_x = X // 2
    cell_y = Y // 2
    time = np.arange(T) * t_res

    plt.figure(figsize=(12, 4))
    plt.plot(time, data[:, cell_x, cell_y], color="tab:blue", linewidth=2)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage")
    plt.title(f"Voltage signal @ cell ({cell_x}, {cell_y}), resolution={resolution}, conmul={conmul}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    suffix = f"_{set_name}" if set_name else ""
    save_path = os.path.join(folder, f"Voltage_signal{suffix}_{resolution}_{conmul}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved voltage signal preview: {save_path}")


def save_voltage_gif(folder, tensor, t_res, set_name, max_frames=200, max_size=200):
    """Save one voltage-field GIF for a dataset subset.

    The constructor loops over spatial resolutions.  The output folder is
    shared by those resolutions, so ``set_name`` is used as a guard and only
    one GIF is written for each subset (train/test/eval), rather than one GIF
    per 101/401 tensor.
    """
    if not set_name or not torch.is_tensor(tensor) or tensor.ndim != 4:
        return

    folder = os.path.abspath(folder)
    gif_key = (folder, set_name)
    if gif_key in _saved_gif_previews:
        return

    voltage = tensor[0].detach().cpu()
    if voltage.shape[0] == 0:
        return

    # Keep generated visualisations practical for long simulations and large
    # grids while retaining the full tensor unchanged on disk.
    frame_indices = np.linspace(
        0, voltage.shape[0] - 1,
        min(max_frames, voltage.shape[0]),
        dtype=int,
    )
    voltage = voltage[frame_indices]
    spatial_step = max(1, int(np.ceil(max(voltage.shape[1:]) / max_size)))
    voltage = voltage[:, ::spatial_step, ::spatial_step].numpy()

    v_min = float(np.nanmin(voltage))
    v_max = float(np.nanmax(voltage))
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        return
    if v_min == v_max:
        v_max = v_min + 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(
        voltage[0], cmap="viridis", vmin=v_min, vmax=v_max,
        origin="lower", animated=True,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Voltage")
    title = ax.set_title(f"{set_name}: t = {frame_indices[0] * t_res:.1f} ms")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame_number):
        image.set_data(voltage[frame_number])
        title.set_text(
            f"{set_name}: t = {frame_indices[frame_number] * t_res:.1f} ms"
        )
        return image, title

    animation = FuncAnimation(
        fig, update, frames=len(frame_indices), interval=100,
        blit=True, repeat=False,
    )
    save_path = os.path.join(folder, f"Voltage_Evolution_{set_name}.gif")
    animation.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    _saved_gif_previews.add(gif_key)
    print(f"Saved voltage GIF: {save_path}")

# Function for APD extraction from simulation:
def compute_apd_map_multibeat_coarse(
    voltage_array, t_res=10.0, rest_potential=-80.0, repolarization_level=0.9,
    amp_threshold=5.0, upstroke_frac=0.5, min_cycle_ms=100.0):
    """
    Robust APD (e.g. APD90) computation for coarsely sampled (e.g. 10ms) data.
    Detects beats using upward threshold crossings and computes per-pixel APDs.
    """

    t_len, nx, ny = voltage_array.shape
    apd_map_mean = np.full((nx, ny), np.nan)
    apd_map_var  = np.full((nx, ny), np.nan)
    min_cycle_pts = int(min_cycle_ms / t_res)

    for i in range(nx):
        for j in range(ny):
            v = voltage_array[:, i, j]
            if np.isnan(v).any() or np.allclose(v, v[0]):
                continue

            v_rest = rest_potential if rest_potential is not None else np.percentile(v, 5)
            v_peak = np.max(v)
            amp = v_peak - v_rest
            if amp < amp_threshold:
                continue

            v_up = v_rest + upstroke_frac * amp     # upward crossing
            v_thr = v_rest + (1 - repolarization_level) * amp  # e.g., 0.1*A above rest for APD90

            # Detect upstroke indices (low→high crossing of v_up)
            crossings = np.where((v[:-1] < v_up) & (v[1:] >= v_up))[0]
            if len(crossings) == 0:
                continue

            # Remove closely spaced ones (noise)
            beat_starts = [crossings[0]]
            for idx in crossings[1:]:
                if idx - beat_starts[-1] >= min_cycle_pts:
                    beat_starts.append(idx)
            beat_starts = np.array(beat_starts)

            apds = []
            for up_idx in beat_starts:
                # Find repolarization point after upstroke
                after = np.where(v[up_idx:] <= v_thr)[0]
                if len(after) == 0:
                    continue
                repol_idx = up_idx + after[0]
                apd_ms = (repol_idx - up_idx) * t_res
                apds.append(apd_ms)

            if len(apds) > 0:
                apd_map_mean[i, j] = np.mean(apds)
                apd_map_var[i, j]  = np.var(apds)

    global_mean = np.nanmean(apd_map_mean)
    global_var  = np.nanvar(apd_map_mean)
    print(f"Global mean APD90: {global_mean:.1f} ms, spatial variance: {global_var:.2f}")
    return apd_map_mean, apd_map_var, global_mean, global_var
    

def amplitudes(channel_data):
    ch_data = channel_data # shape: [N_samples, T, X, Y]
    ch_min = ch_data.min()
    ch_max = ch_data.max()
    if args.norm:
        if (ch_max - ch_min) == 0:
            print(f"Warning: Channel {ch} has constant values. Skipping normalization.")
        else:
            dataset[:, ch] = (ch_data - ch_min) / (ch_max - ch_min)
    return ch_max, ch_min

def plot_cell_trajectories(tensor, t_res, cell_coords, channel=0):
    """
    Plot the voltage (or recovery variable) trajectories of selected cells.

    Args:
        tensor: torch.Tensor of shape (2, T, X, Y)
        t_res: timestep resolution in ms
        cell_coords: list of (x, y) tuples to plot (e.g., [(10, 10), (20, 20)])
        channel: 0 for voltage, 1 for recovery
    """
    data = tensor[channel].cpu().numpy()  # shape: (T, X, Y)
    T, X, Y = data.shape
    time = np.arange(T) * t_res

    plt.figure(figsize=(10, 6))
    for (x, y) in cell_coords:
        if x >= X or y >= Y:
            print(f"Skipping ({x}, {y}) – out of bounds (max X={X-1}, Y={Y-1})")
            continue

        trace = data[:, x, y]
        plt.plot(time, trace, label=f'Cell ({x}, {y})')

        # --- Compute APD & amplitude for each cell ---
        # Find min/max voltage (amplitude)
        vmax = np.max(trace)
        vmin = np.min(trace)
        amplitude = vmax - vmin

        # Estimate APD90 (duration until voltage repolarizes to 90% of amplitude)
        threshold = vmin + 0.1 * amplitude
        above = trace > threshold
        if np.any(above):
            t_up = np.argmax(above) * t_res
            t_down = (len(above) - np.argmax(above[::-1]) - 1) * t_res
            apd90 = (t_down - t_up)
        else:
            apd90 = np.nan

        #print(f"Cell ({x}, {y}) → amplitude={amplitude:.3f}, APD90={apd90:.2f} ms")

    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)" if channel == 0 else "Recovery variable")
    plt.title("Action Potentials at Selected Cells")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(args.folder_name, 'APD_traces.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

## ======================================================================= ##
# == PROCESSING EACH DATA STRUCTURE IN THE PARENT FOLDER  == ##
## ======================================================================= ##
def data_constructor_openCARP(args):
    # LOADING EACH OF THE STRUCTURES: 
    folder = args.folder_name

    if args.mat:
        #load the matlab files 
        all_files = [f for f in sorted(os.listdir(folder)) if f.endswith(".mat") and "init" not in f.lower()]
    else:
        #load the tensors
        all_files = [f for f in sorted(os.listdir(folder)) if f.endswith(".pt") and "init" not in f.lower()]

    for fname in all_files:
        filepath = os.path.join(args.folder_name, fname)
        
        # Create log file for saving the metadata
        temp_log = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.log')
        log_path_temp = temp_log.name
        # Redirect stdout to the log
        sys.stdout = temp_log

        try:
            if args.mat:
                sim_type, conmul, scaleres = parse_filename_mat(fname)
                dataset, t_v, t_res, x_v, x_res  = load_single_sample_mat(filepath, time_limit=args.time_limit)
            else:
                sim_type, conmul, scaleres, scalet = parse_filename_pt(fname)
                coords_path = os.path.join(args.folder_name, 'coords_cm.npy')
                print(f"Coordinates loaded from mesh: {coords_path}")
                dataset, t_v, t_res, x_v, x_res  = load_single_sample_pt(filepath, coords_path=coords_path, time_resolution_ms=args.timestep * scalet, time_limit= args.time_limit)
                '''
                #Extract voltage for APD calculation 
                    voltage_np = dataset[0].numpy()  # (t, x, y)
                    apd_mean_map, apd_var_map, mean_apd, var_apd = compute_apd_map_multibeat_coarse(
                        voltage_np,
                        t_res= t_res,
                        rest_potential=0.0,      # or None if baseline varies spatially
                        repolarization_level=0.9,  # APD90
                        upstroke_frac=0.5,
                        min_cycle_ms=180.0
                    )
                '''
        
        

            # Optional Normalization step - using Aliev_panfilov scaling model
            C = dataset.shape[0]

            for ch in range(C):
                ch_data = dataset[ch]  # shape: (T, X, Y)
                E_rest = ch_data.min()
                E_peak = ch_data.max()
                A = E_peak - E_rest

                print(f"Channel {ch} pre-normalisation range: {E_rest:.4f}, {E_peak:.4f}")
                if ch == 0:
                    print(f"Resting potential (E_rest) = {E_rest:.2f} mV, amplitude (A) = {A:.2f} mV")
                if args.norm:
                    if A == 0:
                        print(f"Channel {ch} has constant values, skipping normalization.")
                    else:
                        dataset[ch] = (ch_data - E_rest) / A
                        print(f"Channel {ch} post-normalisation range: {dataset[ch].min():.4f}, {dataset[ch].max():.4f}")
                else:
                    print(f"Channel {ch} not normalised, raw values retained")
            

            #save the full dataset for simulator comparison later:
            full_dataset = dataset 
            print(f"Loaded dataset shape: {dataset.shape}") 
            
            C, T, X, Y = dataset.shape
            centre = int(int(X) / 2)
            quarter = int(int(X) / 4)
            three_quarter = int(int(X) * (3/4))

            cells = [[quarter, quarter], [centre, centre], [three_quarter, three_quarter]]

            #plot_cell_trajectories(full_dataset, t_res, cell_coords=cells)

            # Samples formed of input-output pairs split by moving time windows.
            input_steps = args.input_window
            output_steps = input_steps
            window_mode = str(getattr(args, 'window_mode', 'overlapping')).lower()
            print(
                "Input-Output pairs formed using moving window of "
                f"[ Input: {input_steps}, Output: {output_steps}, Mode: {window_mode} ]"
            )
            inputs, outputs, times_all = create_io_pairs(dataset, input_steps, window_mode)
            #inputs, outputs, times_all = create_io_pairs_nonoverlapping(dataset, input_steps, output_steps)
            print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}")
            print(f"Dataset source frames: {dataset.shape[1]}")
            print(f"Dataset total samples (input-output windows): {inputs.shape[0]}")

            ## FORMING THE TEST-TRAIN SETS #
            train_percent = args.train
            test_percent = 1 - train_percent

            # Perform the split
            # These are also used to build subset visualisations.  Initialise
            # them before the split because the simple train/test branch does
            # not otherwise create time-window lists.
            times_train = []
            times_test = []
            times_eval = []
            if test_percent != 0:
                if args.random_split:
                    print("Train-Test split performed randomly")

                    if args.channels == 2:

                        # Perform random split on inputs, outputs, AND times
                        #input_data_train, input_data_test, output_data_train, output_data_test, times_train, times_test = train_test_split(
                        #    inputs, outputs, times_all, test_size=test_percent, random_state=42, shuffle=True
                        #)

                        # Optional: flatten the time indices to see all times used in each set
                        #train_times_flat = sorted({t for w in times_train for t in w})
                        #test_times_flat = sorted({t for w in times_test for t in w})

                        #print("Training time steps:", train_times_flat)
                        #print("Testing time steps:", test_times_flat)

                        input_data_train, output_data_train, times_train, input_data_test, output_data_test, times_test = block_split(
                            inputs, outputs, times_all,
                            test_fraction=1 - args.train,
                            n_blocks=None,   # auto chooses ~10 blocks for 200-frame trajectory
                            seed=42
                        )

                        # Optional diagnostics
                        train_times_flat = sorted({t for w in times_train for t in w})
                        test_times_flat  = sorted({t for w in times_test for t in w})

                        print("Training time steps:", train_times_flat)
                        print("Testing time steps:", test_times_flat)

                        
                        #input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(
                        # inputs, outputs, test_size=test_percent, random_state=42, shuffle=True)

                    elif args.channels == 1:
                        input_v = inputs[:, 0].unsqueeze(1)
                        output_v = outputs[:, 0].unsqueeze(1)
                        input_w = inputs[:, 1].unsqueeze(1)
                        output_w = outputs[:, 1].unsqueeze(1)

                        input_data_train_v, input_data_test_v, output_data_train_v, output_data_test_v = train_test_split(
                            input_v, output_v, test_size=test_percent, random_state=42, shuffle=True)

                        input_data_train_w, input_data_test_w, output_data_train_w, output_data_test_w = train_test_split(
                            input_w, output_w, test_size=test_percent, random_state=42, shuffle=True)

                else:
                    buffer = 1 if window_mode == 'block' else input_steps + output_steps - 1
                    print(f"Train-Test split performed sequentially with buffer={buffer}")

                    num_samples = inputs.shape[0]
                    split_index = int(train_percent * num_samples)

                    # Adjust split_index to leave a buffer
                    adjusted_split_index = max(0, min(split_index, num_samples - buffer))

                    if args.channels == 2:
                        input_data_train = inputs[:adjusted_split_index]
                        output_data_train = outputs[:adjusted_split_index]

                        input_data_test = inputs[adjusted_split_index + buffer:]
                        output_data_test = outputs[adjusted_split_index + buffer:]
                        times_train = times_all[:adjusted_split_index]
                        times_test = times_all[adjusted_split_index + buffer:]

                    elif args.channels == 1:
                        input_v = inputs[:, 0].unsqueeze(1)
                        output_v = outputs[:, 0].unsqueeze(1)
                        input_w = inputs[:, 1].unsqueeze(1)
                        output_w = outputs[:, 1].unsqueeze(1)

                        input_data_train_v = input_v[:adjusted_split_index]
                        output_data_train_v = output_v[:adjusted_split_index]
                        input_data_test_v = input_v[adjusted_split_index + buffer:]
                        output_data_test_v = output_v[adjusted_split_index + buffer:]

                        input_data_train_w = input_w[:adjusted_split_index]
                        output_data_train_w = output_w[:adjusted_split_index]
                        input_data_test_w = input_w[adjusted_split_index + buffer:]
                        output_data_test_w = output_w[adjusted_split_index + buffer:]
                    
                    
                    
            eval_fraction = float(getattr(args, "eval", 0.0) or 0.0)
            train_fraction = float(getattr(args, "train", 0.0) or 0.0)
            test_fraction = float(getattr(args, "test", 1.0 - train_fraction) or 0.0)
            fraction_total = train_fraction + test_fraction + eval_fraction
            if fraction_total < 0 or fraction_total > 1.0 + 1e-8:
                raise ValueError(
                    f"train + test + eval fractions must be between 0 and 1, got {fraction_total}"
                )

            has_split_eval = False
            if not args.is_full_eval and eval_fraction > 0:
                buffer = 1 if window_mode == 'block' else input_steps + output_steps - 1
                (input_data_train, output_data_train, times_train), \
                    (input_data_test, output_data_test, times_test), \
                    (input_data_eval, output_data_eval, times_eval) = sequential_three_way_split(
                        inputs, outputs, times_all,
                        train_fraction=train_fraction,
                        test_fraction=test_fraction,
                        eval_fraction=eval_fraction,
                        buffer_size=buffer,
                    )
                has_split_eval = True

            if args.is_full_eval:
                print(f"Dataset evaluation samples: {inputs.shape[0]}")
            else:
                print(f"Dataset train samples: {input_data_train.shape[0]}")
                print(f"Dataset test samples: {input_data_test.shape[0]}")
                if has_split_eval:
                    print(f"Dataset evaluation samples: {input_data_eval.shape[0]}")
                print(f"Dataset buffer/unassigned samples: {max(0, inputs.shape[0] - input_data_train.shape[0] - input_data_test.shape[0] - (input_data_eval.shape[0] if has_split_eval else 0))}")

            # Set folder base path
            # Change resolution to the the extracted value from the dataset: 
            resolution = x_v 
            base_folder = os.path.join("datasets_2", "openCARP_Data", sim_type)
            #base_folder = os.path.join("datasets", "openCARP_Data", sim_type)
            norm_suffix = "_norm" if args.norm else ""

            if args.is_full_eval:
                # No test train split - saving for evaluation
                split_label = f"Full_Set_{t_v}_frames_{input_steps}_inputsteps_{output_steps}_outputsteps_{window_mode}"
                folder = os.path.join(base_folder, split_label + norm_suffix)
                os.makedirs(folder, exist_ok=True)

                if args.channels == 2:
                    # Save in training format
                    label = f"2D_AP_eval_{resolution}_{conmul}.pt"
                    save_dataset_pair(folder, label, inputs, outputs)

                    #Save full set
                    label = f"2D_AP_eval_full_{resolution}_{conmul}.pt"
                    save_full_dataset(folder, label, full_dataset)
                    save_voltage_signal_preview(folder, full_dataset, t_res, resolution, conmul, "full")
                    save_voltage_gif(folder, full_dataset, t_res, "eval")

                elif args.channels == 1:
                    label_v = f"2D_AP_eval_v_{resolution}_{conmul}.pt"
                    label_w = f"2D_AP_eval_w_{resolution}_{conmul}.pt"

                    input_data_all_v = inputs[:, 0].unsqueeze(1)
                    output_data_all_v = outputs[:, 0].unsqueeze(1)
                    input_data_all_w = inputs[:, 1].unsqueeze(1)
                    output_data_all_w = outputs[:, 1].unsqueeze(1)

                    save_dataset_pair(folder, label_v, input_data_all_v, output_data_all_v)
                    save_dataset_pair(folder, label_w, input_data_all_w, output_data_all_w)
                    save_voltage_signal_preview(folder, full_dataset, t_res, resolution, conmul, "full")
            else:
                # Train/Test split
                split_type = "RDN" if args.random_split else "SEQ"
                split_suffix = "-eval" if has_split_eval else ""
                split_label = f"{args.train}_trn-tst{split_suffix}_{t_v}_frames_{input_steps}_inputsteps_{output_steps}_outputsteps_{window_mode}_{split_type}"
                folder = os.path.join(base_folder, split_label + norm_suffix)
                os.makedirs(folder, exist_ok=True)

                if args.channels == 2:
                    label_train = f"2D_AP_train_{resolution}_{conmul}.pt"
                    label_test = f"2D_AP_test_{resolution}_{conmul}.pt"
                    save_dataset_pair(folder, label_train, input_data_train, output_data_train)
                    save_dataset_pair(folder, label_test, input_data_test, output_data_test)
                    if has_split_eval:
                        label_eval = f"2D_AP_eval_{resolution}_{conmul}.pt"
                        save_dataset_pair(folder, label_eval, input_data_eval, output_data_eval)
                    save_voltage_signal_preview(folder, full_dataset, t_res, resolution, conmul, "full")

                    def save_split_preview(split_name, split_times):
                        if not split_times:
                            return
                        start = min(window[0] for window in split_times)
                        stop = max(window[-1] for window in split_times) + 1
                        save_voltage_signal_preview(
                            folder,
                            full_dataset[:, start:stop],
                            t_res,
                            resolution,
                            conmul,
                            split_name,
                        )

                    save_split_preview("train", times_train)
                    save_split_preview("test", times_test)
                    if has_split_eval:
                        save_split_preview("eval", times_eval)

                    def save_split_gif(split_name, split_times):
                        if not split_times:
                            return
                        start = min(window[0] for window in split_times)
                        stop = max(window[-1] for window in split_times) + 1
                        save_voltage_gif(
                            folder,
                            full_dataset[:, start:stop],
                            t_res,
                            split_name,
                        )

                    save_split_gif("train", times_train)
                    save_split_gif("test", times_test)
                    if has_split_eval:
                        save_split_gif("eval", times_eval)

                elif args.channels == 1:
                    label_train_v = f"2D_AP_V_train_{resolution}_{conmul}.pt"
                    label_test_v = f"2D_AP_V_test_{resolution}_{conmul}.pt"

                    label_train_w = f"2D_AP_W_train_{resolution}_{conmul}.pt"
                    label_test_w = f"2D_AP_W_test_{resolution}_{conmul}.pt"

                    save_dataset_pair(folder, label_train_v, input_data_train_v, output_data_train_v)
                    save_dataset_pair(folder, label_test_v, input_data_test_v, output_data_test_v)
                    save_dataset_pair(folder, label_train_w, input_data_train_w, output_data_train_w)
                    save_dataset_pair(folder, label_test_w, input_data_test_w, output_data_test_w)
                    if has_split_eval:
                        save_dataset_pair(folder, f"2D_AP_V_eval_{resolution}_{conmul}.pt", input_data_eval[:, 0].unsqueeze(1), output_data_eval[:, 0].unsqueeze(1))
                        save_dataset_pair(folder, f"2D_AP_W_eval_{resolution}_{conmul}.pt", input_data_eval[:, 1].unsqueeze(1), output_data_eval[:, 1].unsqueeze(1))
                    save_voltage_signal_preview(folder, full_dataset, t_res, resolution, conmul, "full")

            log_file_path = os.path.join(folder, f'{args.output}_{resolution}_{conmul}.txt')
            temp_log.close() 
            shutil.move(log_path_temp, log_file_path)

        except Exception:
                print(f"Skipping {fname} due to error:")
                traceback.print_exc()
                continue
