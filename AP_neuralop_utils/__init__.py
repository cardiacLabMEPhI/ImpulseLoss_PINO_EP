__version__ = '1.0.2'

from .data import load_2D_AP, load_2D_AP_eval
from .training.trainer import Trainer
from .losses.equation_losses import APLoss, APFFTLoss
from .losses.meta_losses import OperatorBackboneLoss, WeightedSumLoss, AdaptiveTrainingLoss
from .losses.data_losses import (
    MSELoss, RMSELoss, WeightedMSELoss, WeightedRMSELoss,
    ImpulseAwareMSELoss, ImpulseAwareRMSELoss,
    SoftRelativeL2ImpulseLoss, HardRelativeL2ImpulseLoss,
    ImpulseRMSELoss,
    LpLoss, BoundaryLoss, ICLoss, BCNeumann,
)
