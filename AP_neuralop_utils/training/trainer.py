from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings
import inspect

import torch

from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False
import json 
import os
import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
from .training_state import load_training_state, save_training_state

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None





class Trainer:
    """
    A general Trainer class to train neural-operators on given datasets. 

    .. note ::
        Our Trainer expects datasets to provide batches as key-value dictionaries, ex.: 
        ``{'x': x, 'y': y}``, that are keyed to the arguments expected by models and losses. 
        For specifics and an example, check ``neuralop.data.datasets.DarcyDataset``. 

    Parameters
    ----------
    model : nn.Module
    n_epochs : int
    wandb_log : bool, default is False
        whether to log results to wandb
    device : torch.device, or str 'cpu' or 'cuda'
    mixed_precision : bool, default is False
        whether to use torch.autocast to compute mixed precision
    data_processor : DataProcessor class to transform data, default is None
        if not None, data from the loaders is transform first with data_processor.preprocess,
        then after getting an output from the model, that is transformed with data_processor.postprocess.
    eval_interval : int, default is 1
        how frequently to evaluate model and log training stats
    log_output : bool, default is False
        if True, and if wandb_log is also True, log output images to wandb
    use_distributed : bool, default is False
        whether to use DDP
    verbose : bool, default is False

    ==== AP PINOs adaptions ===
    We added json logging of the relevant information for later analysis. 
    """
    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool=True,
        device: str='cpu',
        mixed_precision: bool=False,
        data_processor: nn.Module=None,
        eval_interval: int=1,
        log_output: bool=False,
        use_distributed: bool=False,
        verbose: bool=False,

    ):
        """
        """

        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
        self.wandb_log = True
        if wandb_available:
            self.wandb_log = (wandb_log and wandb.run is not None)
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        # handle autocast device
        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            if "cuda" in self.device:
                self.autocast_device_type = "cuda"
            else:
                self.autocast_device_type = "cpu"
        self.mixed_precision = mixed_precision
        self.data_processor = data_processor
    
        # Track starting epoch for checkpointing/resuming
        self.start_epoch = 0

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        train_metrics=None,
        eval_losses=None,
        validation_rollout_steps: int=1,
        save_every: int=None,
        save_best: int=None,
        save_dir: Union[str, Path]="./ckpt",
        resume_from_dir: Union[str, Path]=None,
        json_log_path: Union[str, Path]=None,
        training_plots_dir: Union[str, Path]=None,
        stage_name: str=None,
        early_stopping: bool=False,
        early_stopping_metric: str=None,
        early_stopping_patience: int=None,
        early_stopping_min_delta: float=0.0,
    ):
        """Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here. 

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        scheduler: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        train_metrics: dict, optional
            additional metrics calculated on training predictions and logged
            with the ``train_`` prefix. These metrics are not optimized.
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_best: str, optional, default is None
            if provided, key of metric f"{loader_name}_{loss_name}"
            to monitor and save model with best eval result
            Overrides save_every and saves on eval_interval
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states if
            save_every and/or save_best is provided
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model, 
            optimizer, regularizer, scheduler) from state saved in
            `resume_from_dir`
        json_log_path = path to save the epoch metrics as a json file. Default is none. 
        early_stopping: bool, default False
            if True, stop training when the monitored validation metric does not improve
            for early_stopping_patience validation checks.
        early_stopping_metric: str, optional
            metric key to monitor. Defaults to save_best when not provided.
        early_stopping_patience: int, optional
            number of validation checks without improvement before stopping.
        early_stopping_min_delta: float, default 0.0
            minimum decrease required to count as an improvement.
        
        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders
            
        """
        self.json_log_path = json_log_path
        self.validation_rollout_steps = max(1, int(validation_rollout_steps))
        self.training_plots_dir = Path(training_plots_dir) if training_plots_dir else None
        if self.training_plots_dir is not None:
            self.training_plots_dir.mkdir(parents=True, exist_ok=True)
        self.training_plot_history = []


        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)
        
        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, 'reduction'):
            if training_loss.reduction == "mean":
                warnings.warn(f"{training_loss.reduction=}. This means that the loss is "
                              "initialized to average across the batch dim. The Trainer "
                              "expects losses to sum across the batch dim.")

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)
        
        # accumulated wandb metrics
        self.wandb_epoch_metrics = None
        #initialise the epoch metrics before training:
        epoch_metrics = None  

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        if self.use_distributed and dist.is_initialized():
            device_id = dist.get_rank()
            self.model = DDP(self.model, device_ids=[device_id], output_device=device_id)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
        
        # ensure save_best is a metric we collect
        available_metrics = []
        for name1, name2 in test_loaders.keys():
            for metric in eval_losses.keys():
                available_metrics.append(f"({name1}, {name2})_{metric}")

        if self.save_best is not None:
            assert self.save_best in available_metrics,\
                f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float('inf')
            best_metric_path = None
            initialize_best_from_current = False
            if save_dir is not None:
                best_metric_path = Path(save_dir) / "best_model_metric.json"
                if best_metric_path.exists():
                    with open(best_metric_path, "r") as f:
                        best_metric_state = json.load(f)
                    if best_metric_state.get("metric") == self.save_best:
                        best_metric_value = best_metric_state.get("value", best_metric_value)
                        if self.verbose:
                            print(f"Loaded best {self.save_best}={best_metric_value} from {best_metric_path}")
                elif resume_from_dir is not None:
                    initialize_best_from_current = True
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        early_stopping_metric = early_stopping_metric or self.save_best
        if early_stopping:
            if early_stopping_metric is None:
                raise ValueError("early_stopping=True requires early_stopping_metric or save_best.")
            if early_stopping_metric not in available_metrics:
                raise ValueError(
                    "early_stopping_metric must be one of "
                    f"{available_metrics}, got {early_stopping_metric}"
                )
            if early_stopping_patience is None or early_stopping_patience < 1:
                raise ValueError("early_stopping_patience must be a positive integer.")
            early_stopping_min_delta = float(early_stopping_min_delta)
            early_stopping_best = float("inf")
            early_stopping_bad_evals = 0
        else:
            early_stopping_best = None
            early_stopping_bad_evals = 0

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"TRAINING STAGE: {stage_name or 'Training'}")
            print("-" * 80)
            print(f'Training samples: {len(train_loader.dataset)}')
            print(f'Testing samples: {[len(loader.dataset) for loader in test_loaders.values()]}')
            print(f'Testing resolutions: {[name for name in test_loaders]}')
            print("=" * 80)
            sys.stdout.flush()

        if self.save_best is not None and initialize_best_from_current:
            self.epoch = self.start_epoch
            baseline_metrics = self.evaluate_all(epoch=self.start_epoch,
                                                 eval_losses=eval_losses,
                                                 test_loaders=test_loaders)
            best_metric_value = baseline_metrics[self.save_best]
            if isinstance(best_metric_value, torch.Tensor):
                best_metric_value = best_metric_value.item()
            self.checkpoint(save_dir)
            with open(best_metric_path, "w") as f:
                json.dump({
                    "metric": self.save_best,
                    "value": best_metric_value,
                    "epoch": self.start_epoch
                }, f, indent=2)
        
        epoch_iterator = range(self.start_epoch, self.n_epochs)
        progress_bar = None
        if tqdm is not None and self.verbose:
            progress_bar = tqdm(
                epoch_iterator,
                desc=stage_name or "Training",
                total=self.n_epochs - self.start_epoch,
                dynamic_ncols=True,
                leave=True,
            )
            epoch_iterator = progress_bar

        for epoch in epoch_iterator:
            stop_training = False
            train_err, avg_loss, avg_lasso_loss, epoch_train_time, train_loss_components =\
                  self.train_one_epoch(
                      epoch, train_loader, training_loss,
                      train_metrics=train_metrics,
                  )
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time,
            )
            epoch_metrics.update(train_loss_components)
            
            is_eval_epoch = epoch % self.eval_interval == 0 or epoch == self.n_epochs - 1
            if is_eval_epoch:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(epoch=epoch,
                                                eval_losses=eval_losses,
                                                test_loaders=test_loaders)

                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    current_metric_value = eval_metrics[save_best]
                    if isinstance(current_metric_value, torch.Tensor):
                        current_metric_value = current_metric_value.item()
                    if current_metric_value < best_metric_value:
                        best_metric_value = current_metric_value
                        self.checkpoint(save_dir)
                        if best_metric_path is not None:
                            with open(best_metric_path, "w") as f:
                                json.dump({
                                    "metric": save_best,
                                    "value": best_metric_value,
                                    "epoch": epoch + 1
                                }, f, indent=2)

                if early_stopping:
                    monitored_value = eval_metrics[early_stopping_metric]
                    if isinstance(monitored_value, torch.Tensor):
                        monitored_value = monitored_value.item()
                    improved = monitored_value < early_stopping_best - early_stopping_min_delta
                    if improved:
                        early_stopping_best = monitored_value
                        early_stopping_bad_evals = 0
                    else:
                        early_stopping_bad_evals += 1

                    epoch_metrics["early_stopping_metric"] = early_stopping_metric
                    epoch_metrics["early_stopping_best"] = early_stopping_best
                    epoch_metrics["early_stopping_bad_evals"] = early_stopping_bad_evals

                    if early_stopping_bad_evals >= early_stopping_patience:
                        stop_training = True
                        epoch_metrics["early_stopped"] = True
                        if self.verbose:
                            print(
                                f"Early stopping: {early_stopping_metric} did not improve "
                                f"for {early_stopping_bad_evals} validation checks "
                                f"(patience={early_stopping_patience})."
                            )

            if progress_bar is not None:
                postfix = {
                    "epoch": epoch + 1,
                    "train": f"{train_err:.4g}",
                    "avg": f"{avg_loss:.4g}",
                }
                if save_best is not None and is_eval_epoch and save_best in epoch_metrics:
                    metric_value = epoch_metrics[save_best]
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.item()
                    postfix[self.save_best] = f"{metric_value:.4g}"
                progress_bar.set_postfix(postfix)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

            #save the training results in a json file - after transforming the tensor data:
            # Add epoch number to metrics
            epoch_metrics['epoch'] = epoch + 1

            def tensor_to_python(obj):
                if isinstance(obj, torch.Tensor):
                    if obj.numel() == 1:  # single value tensor
                        return obj.item()
                    else:  # tensor with multiple elements
                        return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: tensor_to_python(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [tensor_to_python(i) for i in obj]
                else:
                    return obj
                
            epoch_metrics_clean = tensor_to_python(epoch_metrics)

            # Update live plots only after validation. Thus every plotted point
            # corresponds to a completed validation event.
            if is_eval_epoch and self.training_plots_dir is not None:
                self.training_plot_history.append(epoch_metrics_clean)
                self.update_training_plots()

            # Append to JSON log file
            if self.json_log_path is not None:
                os.makedirs(os.path.dirname(self.json_log_path), exist_ok=True)
                with open(self.json_log_path, "a") as f:
                    f.write(json.dumps(epoch_metrics_clean) + "\n")

            if stop_training:
                break

        return epoch_metrics

    def update_training_plots(self):
        """Rewrite one PNG plot per train/validation metric.

        ``impulse_rmse`` is intentionally included here as a first-class
        metric.  It must produce ``impulse_rmse.png`` with both the train
        series and every validation-loader series whenever those values are
        present in the epoch log.
        """
        if not self.training_plot_history or self.training_plots_dir is None:
            return

        import re
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        series = {}
        for entry in self.training_plot_history:
            for key, value in entry.items():
                if key in {
                    "epoch", "epoch_train_time", "early_stopping_metric",
                    "early_stopping_best", "early_stopping_bad_evals",
                    "early_stopped",
                }:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue

                if key.startswith("train_"):
                    metric_name = "train_err" if key == "train_err" else key[len("train_"):]
                    label = "train"
                elif key in {"avg_loss", "train_err", "avg_lasso_loss"}:
                    metric_name = key
                    label = "train"
                elif key.startswith("(") and ")_" in key:
                    # Validation keys look like
                    # ``(101, 1.0)_soft_relative_l2_impulse_loss``.
                    # Split at the loader boundary, not at the last '_',
                    # because metric names themselves contain underscores.
                    loader_name, metric_name = key.split(")_", 1)
                    label = f"val {loader_name})"
                else:
                    continue
                series.setdefault(metric_name, {}).setdefault(label, []).append(
                    (entry["epoch"], float(value))
                )

        def save_metric_plot(metric_name, labels):
            fig, ax = plt.subplots(figsize=(9, 5))
            for label, points in labels.items():
                points.sort(key=lambda point: point[0])
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    linewidth=1.8,
                    label=label,
                )
            ax.set_title(metric_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()

            # Keep a stable, predictable filename for the impulse-only RMSE
            # plot instead of relying on any formatting of the log key.
            if metric_name == "impulse_rmse":
                safe_name = "impulse_rmse"
            else:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_name).strip("_")
            output_path = self.training_plots_dir / f"{safe_name}.png"
            # Overwrite the same path in place so viewers that already opened
            # the PNG can detect the updated file after the next validation.
            fig.savefig(output_path, dpi=140)
            plt.close(fig)

        for metric_name, labels in series.items():
            save_metric_plot(metric_name, labels)

        # Save one overview figure containing every metric. It is rewritten
        # after each validation together with the individual metric plots.
        metric_items = list(series.items())
        if metric_items:
            ncols = 2
            nrows = (len(metric_items) + ncols - 1) // ncols
            overview_fig, overview_axes = plt.subplots(
                nrows,
                ncols,
                figsize=(18, max(5, 4.5 * nrows)),
                squeeze=False,
            )
            overview_axes = overview_axes.ravel()
            for axis, (metric_name, labels) in zip(overview_axes, metric_items):
                for label, points in labels.items():
                    points.sort(key=lambda point: point[0])
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        linewidth=1.5,
                        label=label,
                    )
                axis.set_xlabel("Epoch")
                axis.set_ylabel(metric_name)
                axis.grid(True, alpha=0.3)
                axis.legend(fontsize="small")

            for axis in overview_axes[len(metric_items):]:
                axis.axis("off")

            overview_fig.suptitle("Training and validation metrics", fontsize=16)
            overview_fig.tight_layout(rect=(0, 0, 1, 0.97))
            overview_fig.savefig(
                self.training_plots_dir / "all_training_plots.png",
                dpi=140,
            )
            plt.close(overview_fig)

        # Explicitly rewrite the impulse-only RMSE plot on every validation
        # event. This keeps its update behavior identical to rmse.png even if
        # additional metric-specific plotting logic is added later.
        if "impulse_rmse" in series:
            save_metric_plot("impulse_rmse", series["impulse_rmse"])
    

    def train_one_epoch(self, epoch, train_loader, training_loss, train_metrics=None):
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        test_loaders : dict
            dict of test torch.utils.data.DataLoader objects

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        
        # track number of training examples in batch
        self.n_samples = 0
        component_sums = {}
        component_counts = {}
        training_loss_aggregation = getattr(
            training_loss,
            "metric_aggregation",
            getattr(training_loss, "reduction", "sum")
        )

        for idx, sample in enumerate(train_loader):
            batch_size = sample["y"].shape[0] if isinstance(sample.get("y"), torch.Tensor) else 1
            loss = self.train_one_batch(
                idx, sample, training_loss, train_metrics=train_metrics
            )
            for name, value in getattr(self, "last_batch_metrics", {}).items():
                key = f"train_{name}"
                component_sums[key] = component_sums.get(key, 0.0) + value.item()
                component_counts[key] = component_counts.get(key, 0) + 1
            if hasattr(training_loss, "last_losses") and training_loss.last_losses:
                for name, value in training_loss.last_losses.items():
                    key = f"train_{name}_loss"
                    component_sums[key] = component_sums.get(key, 0.0) + value.item()
                    component_counts[key] = component_counts.get(key, 0) + 1
            if hasattr(training_loss, "last_lambdas") and training_loss.last_lambdas is not None:
                loss_names = list(getattr(training_loss, "last_losses", {}).keys())
                for name, value in zip(loss_names, training_loss.last_lambdas):
                    key = f"train_lambda_{name}"
                    component_sums[key] = component_sums.get(key, 0.0) + value.item()
                    component_counts[key] = component_counts.get(key, 0) + 1
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                if training_loss_aggregation == "mean":
                    avg_loss += loss.item() * batch_size
                else:
                    avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader) # This gives the mean error for the epoch in terms of batches (total error/no. batches)
        avg_loss /= self.n_samples # This gives the mean error for the epoch in termss of samples (total error/no. samples)
        
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        train_loss_components = {
            key: component_sums[key] / component_counts[key]
            for key in component_sums
        }
        
        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
               
            )
        return train_err, avg_loss, avg_lasso_loss, epoch_train_time, train_loss_components

    def evaluate_all(self, epoch, eval_losses, test_loaders):
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_metrics = self.evaluate(eval_losses, loader,
                                    log_prefix=loader_name)   
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch,
                      eval_metrics=all_metrics)
        return all_metrics
    
    def evaluate(self, loss_dict, data_loader, log_prefix="", epoch=None):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary
        epoch : int | None
            current epoch. Used when logging both train and eval
            default None
        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        # Ensure model and data processor are loaded to the proper device
        self.model = self.model.to(self.device)
        if self.data_processor is not None and self.data_processor.device != self.device:
            self.data_processor = self.data_processor.to(self.device)
        
        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}
        metric_aggregations = {
            f"{log_prefix}_{loss_name}": getattr(loss, "metric_aggregation", getattr(loss, "reduction", "sum"))
            for loss_name, loss in loss_dict.items()
        }

        # Warn the user if any of the eval losses is reducing across the batch
        for _, eval_loss in loss_dict.items():
            if hasattr(eval_loss, 'reduction'):
                if eval_loss.reduction == "mean":
                    warnings.warn(f"{eval_loss.reduction=}. This means that the loss is "
                                "initialized to average across the batch dim. The Trainer "
                                "expects losses to sum across the batch dim.")

        self.n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                batch_size = sample["y"].size(0)
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                eval_step_losses, outs = self.eval_one_batch(sample, loss_dict, return_output=return_output)

                for loss_name, val_loss in eval_step_losses.items():
                    key = f"{log_prefix}_{loss_name}"
                    aggregation = metric_aggregations[key]
                    if aggregation == "mean":
                        errors[key] += val_loss * batch_size
                    elif aggregation == "rmse":
                        errors[key] += (val_loss ** 2) * batch_size
                    else:
                        errors[key] += val_loss
            
        for key in errors.keys():
            errors[key] /= self.n_samples
            if metric_aggregations[key] == "rmse":
                errors[key] = torch.sqrt(errors[key])

        # on last batch, log model outputs
        if self.log_output:
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)
        
        return errors
    
    def on_epoch_start(self, epoch):
        """on_epoch_start runs at the beginning
        of each training epoch. This method is a stub
        that can be overwritten in more complex cases.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    def train_one_batch(self, idx, sample, training_loss, train_metrics=None):
        """Run one batch of input through model
           and return training loss on outputs

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        self.optimizer.zero_grad(set_to_none=True)
        self.last_batch_metrics = {}
        if self.regularizer:
            self.regularizer.reset()
        rollout_y = sample.get("rollout_y", None)
        if self.data_processor is not None:
            processor_sample = {
                k: v for k, v in sample.items()
                if k != "rollout_y"
            }
            sample = self.data_processor.preprocess(processor_sample)
            if rollout_y is not None:
                sample["rollout_y"] = rollout_y.to(self.device)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        if isinstance(sample["y"], torch.Tensor):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1

        model_sample = {
            k: v for k, v in sample.items()
            if k not in {"y", "rollout_y"}
        }

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**model_sample)
        else:
            out = self.model(**model_sample)
        
        if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
            print(f"Raw outputs of shape {out.shape}")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        if train_metrics:
            with torch.no_grad():
                for name, metric in train_metrics.items():
                    self.last_batch_metrics[name] = metric(
                        y_pred=out,
                        y=sample.get("y", None),
                        x=sample.get("x", None),
                    ).detach()

        loss = 0.0
        #print(f"Using training loss: {training_loss.__class__.__name__}")
        
        # Prepare kwargs for loss
        loss_kwargs = {
            "y_pred": out,
            "y": sample.get("y", None),
            "x": sample.get("x", None),
            "rollout_y": sample.get("rollout_y", None),
        }

        # Compute weighted loss
        try:
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    loss = training_loss(**loss_kwargs)
            else:
                loss = training_loss(**loss_kwargs)
        except TypeError:
            # fallback: pass raw sample in case loss ignores query points
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    loss = training_loss(**sample, y_pred=out)
            else:
                loss = training_loss(**sample, y_pred=out)
   
        if self.regularizer:
            loss += self.regularizer.loss
        
        return loss
    
    def eval_one_batch(self,
                       sample: dict,
                       eval_losses: dict,
                       return_output: bool=False):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        rollout_y = sample.get("rollout_y", None)
        if self.data_processor is not None:
            processor_sample = {
                k: v for k, v in sample.items()
                if k != "rollout_y"
            }
            sample = self.data_processor.preprocess(processor_sample)
            if rollout_y is not None:
                sample["rollout_y"] = rollout_y.to(self.device)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        self.n_samples += sample["y"].size(0)

        model_sample = {
            k: v for k, v in sample.items()
            if k not in {"y", "rollout_y"}
        }
        out = self.model(**model_sample)

        # Rollout validation uses the same autoregressive feedback loop as
        # rollout training. The first prediction starts from the true x
        # window; subsequent predictions receive the previous prediction.
        validation_target = sample.get("y", None)
        if self.validation_rollout_steps > 1 and rollout_y is not None:
            rollout_y = rollout_y.to(out.device)
            window = out.shape[2]
            target_steps = rollout_y.shape[2] // window
            n_steps = min(self.validation_rollout_steps, target_steps)
            predictions = [out]
            state = out
            for _ in range(1, n_steps):
                state = self.model(x=state)
                predictions.append(state)
            out = torch.cat(predictions, dim=2)
            validation_target = rollout_y[:, :, :out.shape[2]]

        if isinstance(validation_target, torch.Tensor):
            validation_target = validation_target.to(out.device)
        
        # Prepare kwargs for evaluation losses
        loss_kwargs = {
            "y_pred": out,
            "y": validation_target,
            "x": sample.get("x", None)
        }

        eval_step_losses = {}
        for loss_name, loss in eval_losses.items():
            try:
                # Filter kwargs per loss signature
                sig = inspect.signature(loss.__call__)
                filtered_kwargs = {k: v for k, v in loss_kwargs.items() if k in sig.parameters}
                eval_step_losses[loss_name] = loss(**filtered_kwargs)
                
                # if 'y_pred' in filtered_kwargs and 'y' in filtered_kwargs:
                #     eval_step_losses[loss_name] = loss(**filtered_kwargs)
                # else:
                #     eval_step_losses[loss_name] = loss(**filtered_kwargs)
            except TypeError:
                # fallback for losses that ignore query points
                filtered_kwargs = {k: v for k, v in sample.items() if k in sig.parameters}
                filtered_kwargs["y_pred"] = out
                eval_step_losses[loss_name] = loss(**filtered_kwargs)
                # if 'y_pred' in filtered_kwargs and 'y' in filtered_kwargs:
                #     eval_step_losses[loss_name] = loss(**filtered_kwargs)
                # else:
                #     eval_step_losses[loss_name] = loss(**filtered_kwargs)

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def log_training(self, 
            epoch:int,
            time: float,
            avg_loss: float,
            train_err: float,
            avg_lasso_loss: float=None,
            lr: float=None,
            train_loss_components: dict=None,
            ):
        """Basic method to log results
        from a single training epoch. 
        

        Parameters
        ----------
        epoch: int
        time: float
            training time of epoch
        avg_loss: float
            average train_err per individual sample
        train_err: float
            train error for entire epoch
        avg_lasso_loss: float
            average lasso loss from regularizer, optional
        lr: float
            learning rate at current epoch
        """
        # accumulate info to log to wandb
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr)
            if train_loss_components:
                values_to_log.update(train_loss_components)

        msg = f"[train epoch {epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"
        if train_loss_components:
            for name, value in train_loss_components.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                msg += f", {name}={float(value):.4f}"

        print(msg)
        sys.stdout.flush()
        
        if self.wandb_log:
            wandb.log(data=values_to_log,
                      step=epoch+1,
                      commit=False)
    
    def log_eval(self,
                 epoch: int,
                 eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout and wandb

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader
       
        """
        values_to_log = {}
        grouped_metrics = {}
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                if "_" in metric:
                    prefix, metric_name = metric.rsplit("_", 1)
                else:
                    prefix, metric_name = "eval", metric
                grouped_metrics.setdefault(prefix, []).append((metric_name, value))
            if self.wandb_log:
                values_to_log[metric] = value       

        print("\n" + "-" * 80)
        print(f"EVAL epoch {epoch + 1}")
        print("-" * 80)
        for prefix, metrics in grouped_metrics.items():
            metric_msg = ", ".join(
                f"{name}={value:.4f}" for name, value in metrics
            )
            print(f"{prefix}: {metric_msg}")
        print("-" * 80)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log,
                      step=epoch+1,
                      commit=True)
            
        

    def resume_state_from_dir(self, save_dir):
        """
        Resume training from save_dir created by `neuralop.training.save_training_state`
        
        Params
        ------
        save_dir: Union[str, Path]
            directory in which training state is saved
            (see neuralop.training.training_state)
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # check for save model exists
        if (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        else:
            raise FileNotFoundError("Error: resume_from_dir expects a model\
                                        state dict named model.pt or best_model.pt.")
        # returns model, loads other modules if provided
        self.model, self.optimizer, self.scheduler, self.regularizer, resume_epoch =\
            load_training_state(save_dir=save_dir, save_name=save_name,
                                                model=self.model,
                                                optimizer=self.optimizer,
                                                regularizer=self.regularizer,
                                                scheduler=self.scheduler)

        if resume_epoch is not None:
            if resume_epoch > self.start_epoch:
                self.start_epoch = resume_epoch
                if self.verbose:
                    print(f"Trainer resuming from epoch {resume_epoch}")


    def checkpoint(self, save_dir):
        """checkpoint saves current training state
        to a directory for resuming later. Only saves 
        training state on the first GPU. 
        See neuralop.training.training_state

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        """
        if comm.get_local_rank() == 0:
            if self.save_best is not None:
                save_name = 'best_model'
            else:
                save_name = "model"
            save_training_state(save_dir=save_dir, 
                                save_name=save_name,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                regularizer=self.regularizer,
                                epoch=self.epoch
                                )
            # Checkpoints are saved silently to keep epoch/eval logs readable.

    def generate_coords(self, x, y, x_range=None, y_range=None):
        """
        Generate spatiotemporal coordinates (t, x, y) for input and output tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 2, H, W] at time t=0.
        y : torch.Tensor
            Ground truth output tensor of shape [B, 2, H, W] at time t= t + delta_t.
        delta_t : float (from self)
            Time step between x and y_pred, set during class initialization.
        x_range : tuple or None
            Range of x-coordinates (left to right). Defaults to (0, W - 1).
        y_range : tuple or None
            Range of y-coordinates (top to bottom). Defaults to (0, H - 1).

        Returns
        -------
        coords : torch.Tensor
            Coordinate tensor of shape [2*B, H, W, 3] with (t, x, y), requires_grad=True.
        """
        B, C, H, W = x.shape
        device = x.device

        # Use index-based range if not provided
        if x_range is None:
            x_range = (0.0, float(W - 1))
        if y_range is None:
            y_range = (0.0, float(H - 1))

        # Spatial coordinates
        x_lin = torch.linspace(x_range[0], x_range[1], W, device=device)
        y_lin = torch.linspace(y_range[0], y_range[1], H, device=device)
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing='ij')  # Shape [H, W]

        # Temporal coordinates: 0 for x, delta_t for y_pred
        t_0 = torch.full_like(xx, fill_value=0.0)
        t_dt = torch.full_like(xx, fill_value=self.delta_t)

        # Stack for each batch sample
        coords = []
        for _ in range(B):
            coords_x = torch.stack([t_0, xx, yy], dim=-1)   # [H, W, 3]
            coords_y = torch.stack([t_dt, xx, yy], dim=-1)  # [H, W, 3]
            coords.append(coords_x)
            coords.append(coords_y)

        # Stack the coordinates 
        coords = torch.stack(coords, dim=0)                
        coords.requires_grad_(True)

        return coords
    
    def sample_query_points(self, B, H, W, n_interior=2048, n_boundary=512, 
                            Lx=10.0, Ly=10.0):
        """
        Sample interior and boundary query points for autograd-based losses.

        Parameters
        ----------
        B : int
            Batch size
        H, W : int
            Grid resolution
        n_interior : int
            Number of interior query points per batch element
        n_boundary : int
            Number of query points per boundary (left/right/top/bottom)
        Lx, Ly : float
            Physical domain dimensions (cm)
        device : str
            Torch device

        Returns
        -------
        query_dict : dict
            {
            'interior': torch.Tensor [B, n_interior, 2] (normalized [-1,1]),
            'boundaries': dict with 'left','right','top','bottom' query points
            }
        """
        # ----- Interior points -----
        x_int = torch.rand(B, n_interior, 1, device=self.device) * Lx
        y_int = torch.rand(B, n_interior, 1, device=self.device) * Ly
        interior_xy = torch.cat([x_int, y_int], dim=-1)

        # normalize to [-1, 1]
        interior_norm = interior_xy.clone()
        interior_norm[..., 0] = 2.0 * interior_xy[..., 0] / Lx - 1.0
        interior_norm[..., 1] = 2.0 * interior_xy[..., 1] / Ly - 1.0

        # ----- Boundary points -----
        y_b = torch.rand(B, n_boundary, 1, device=self.device) * Ly
        x_b = torch.rand(B, n_boundary, 1, device=self.device) * Lx

        boundaries = {
            "left":  torch.cat([torch.zeros_like(y_b), y_b], dim=-1),
            "right": torch.cat([torch.full_like(y_b, Lx), y_b], dim=-1),
            "top":   torch.cat([x_b, torch.zeros_like(x_b)], dim=-1),
            "bottom":torch.cat([x_b, torch.full_like(x_b, Ly)], dim=-1),
        }

        # normalize boundaries
        boundaries_norm = {}
        for key, xy in boundaries.items():
            xy_norm = xy.clone()
            xy_norm[..., 0] = 2.0 * xy[..., 0] / Lx - 1.0
            xy_norm[..., 1] = 2.0 * xy[..., 1] / Ly - 1.0
            boundaries_norm[key] = xy_norm

        return {"interior": interior_norm, "boundaries": boundaries_norm}
