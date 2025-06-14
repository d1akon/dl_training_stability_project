import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):

        """
        Compute the Focal Loss between `input` and `target`.

        Focal Loss is designed to address class imbalance by down-weighting easy examples
        and focusing training on hard negatives. It applies a modulating factor (1 - p)^gamma
        to the standard cross-entropy loss.

        Args:
            input (Tensor): Predicted unnormalized scores (logits) of shape [B, C].
            target (Tensor): Ground truth class indices of shape [B].

        Returns:
            Tensor: Scalar loss (averaged or summed depending on `reduction`).
        """
        
        logp = F.log_softmax(input, dim=1)
        ce_loss = F.nll_loss(logp, target, weight=self.weight, reduction='none')
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
