"""Define different loss classes for image, label and regularization."""
# flake8: noqa
from DeepReg.deepreg.loss.deform import BendingEnergy, GradientNorm
from DeepReg.deepreg.loss.image import (
    GlobalMutualInformation,
    GlobalMutualInformationLoss,
    LocalNormalizedCrossCorrelation,
    LocalNormalizedCrossCorrelationLoss,
)
from DeepReg.deepreg.loss.label import (
    CrossEntropy,
    DiceLoss,
    DiceScore,
    JaccardIndex,
    JaccardLoss,
    SumSquaredDifference,
)
