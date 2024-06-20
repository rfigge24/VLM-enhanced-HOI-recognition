"""This module contrains the custom loss used for training of the HICO dataset.

"""

from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional


class MaskedBCELoss(nn.BCELoss):  
    """The custom Binary Cross Entropy Loss function,
    that can mask the outputs of specific classes based on the target label and can convert {-1,1} labels to {0,1}.
    (Note that masking using the target labels is done before target-label conversion!)

    Args:
        ignore_label: The label identifying the classed to ignore with. Defaults to None.
        convert_target_to_01: Will convert {-1,1} target labels to {0,1}. Defaults to False.
        class_weights(Optional): A tensor with classweights, for an input of (nbatch, nclasses) it needs to be of size nclasses. Defaults to None.
        size_average:   (See the docs of torch.nn.BCELoss). Defaults to None.
        reduce:         (See the docs of torch.nn.BCELoss). Defaults to None.
        reduction:      (See the docs of torch.nn.BCELoss). Defaults to 'mean'.

    If ignore_label and convert_target_to_01 are the Default values, the Loss will behave the same as the nn.BCELoss Class.
    The only difference is that there is no option to set a batch element weight!
    It's recommended to then use the nn.BCELoss since this loss contains 2 additional if else statements.
    """  
    def __init__(self, ignore_label=None, convert_target_to_01 = False, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)

        self.ignore_label = ignore_label
        self.convert_target = convert_target_to_01
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """_summary_

        Args:
            input:  (See the docs of torch.nn.BCELoss)
            target: (See the docs of torch.nn.BCELoss)

        Returns:
           A torch.Tensor, for details take a look at the docs of torch.nn.BCELoss
        """
        # Mask the target labels that are equal to ignore_label to 0's if specified:
        # The mask is later used as the weight: 
        if self.ignore_label is not None:
            mask = (target != self.ignore_label).float()
            if self.weight is None:
                weight = mask
            else:
                weight = self.weight * mask
        else:
            weight= self.weight
        
        # Convert the {-1,1} labels to {0,1} if specified:
        if self.convert_target:
            modified_target = (target + 1) / 2
        else: 
            modified_target = target

        loss = F.binary_cross_entropy(input, modified_target, weight, reduction=self.reduction)

        return loss


class MaskedFocalLoss(nn.BCELoss):  
    """The custom Focal Loss function,
    that can mask the outputs of specific classes based on the target label and can convert {-1,1} labels to {0,1}.
    (Note that masking using the target labels is done before target-label conversion!)

    Args:
        ignore_label: The label identifying the classed to ignore with. Defaults to None.
        convert_target_to_01: Will convert {-1,1} target labels to {0,1}. Defaults to False.
        reduction:      (See the docs of torch.nn.BCELoss). Defaults to 'mean'.

    If ignore_label and convert_target_to_01 are the Default values, the Loss will behave the same as the nn.BCELoss Class.
    The only difference is that there is no option to set a batch element weight!
    It's recommended to then use the nn.BCELoss since this loss contains 2 additional if else statements.
    """  
    def __init__(self, ignore_label=None, convert_target_to_01 = False,alpha: float = 0.25, gamma: float = 2, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)

        self.ignore_label = ignore_label
        self.convert_target = convert_target_to_01
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """_summary_

        Args:
            input:  (See the docs of torch.nn.BCELoss)
            target: (See the docs of torch.nn.BCELoss)

        Returns:
           A torch.Tensor, for details take a look at the docs of torch.nn.BCELoss
        """
        # Mask the target labels that are equal to ignore_label to 0's if specified:
        # The mask is later used as the weight: 
        if self.ignore_label is not None:
            mask = (target != self.ignore_label).float()
            if self.weight is None:
                weight = mask
            else:
                weight = self.weight * mask
        else:
            weight= self.weight
        
        # Convert the {-1,1} labels to {0,1} if specified:
        if self.convert_target:
            modified_target = (target + 1) / 2
        else: 
            modified_target = target

        ce_loss = F.binary_cross_entropy(input, modified_target, weight, reduction="none")
        p_t = input * modified_target + (1-input) * (1-modified_target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * modified_target + (1-self.alpha) * (1-modified_target)
            loss = alpha_t * loss
        
        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction =="sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )

        return loss