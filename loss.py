import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SRLoss(nn.Module):
    def __init__(self, nlcd_class_weights, nlcd_means, nlcd_vars, reduction = True):
        super(SRLoss, self).__init__()
        self.nlcd_class_weights = nlcd_class_weights
        self.nlcd_means = nlcd_means
        self.nlcd_vars = nlcd_vars
        self.reduction = reduction

    def ddist(self, prediction, c_interval_center, c_interval_radius):
        return F.relu(torch.abs(prediction.sub(c_interval_center)).sub(c_interval_radius))

    def forward(self, y_out, y_true, from_logits=True):
        # preprocess
        if from_logits:
            softmax = nn.Softmax(dim=1)
            y_pred = softmax(y_out)
        else:
            y_pred = y_out

        loss = 0
        mask_size = torch.unsqueeze(torch.sum(y_true, dim=(1, 2, 3)) + 10, 1)
        for nlcd_idx in range(self.nlcd_class_weights.shape[0]):
            c_mask = torch.unsqueeze(y_true[:, nlcd_idx, :, :], dim=1) # shape Bx1xHxW
            c_mask_size = torch.sum(c_mask, dim=(2,3)) + 0.000001 # shape Bx1 (smoothing in case of 0)
            c_interval_center = self.nlcd_means[nlcd_idx] # shape hr_classes = C
            c_interval_radius = self.nlcd_vars[nlcd_idx]  # shape hr_classes

            masked_probs = y_pred * c_mask
            # BxCxHxW * BxHxW --> BxCxHxW

            # Mean mean of predicted distribution
            mean = (
                torch.sum(masked_probs, dim=(2, 3)) / c_mask_size
            )  # (B,hr_classes) / (B,1) --> shape Bxhr_classes

            # Mean var of predicted distribution
            var = torch.sum(masked_probs * (1.0 - masked_probs), dim=(2, 3)) / (
                c_mask_size * c_mask_size
            )  # (B,hr_classes) / (B,1) --> shape Bxhr_classes

            # calculate numerator
            c_loss = torch.square(self.ddist(mean, c_interval_center, c_interval_radius))
            # calculate denominator
            c_loss = c_loss / (var + (c_interval_radius * c_interval_radius) + 0.000001)
            # calculate log term
            c_loss = c_loss + torch.log(var + 0.03)
            # weight by the fraction of NLCD pixels and the NCLD class weight
            c_loss = c_loss * (c_mask_size / mask_size) * self.nlcd_class_weights[nlcd_idx]

            loss += c_loss #shape (B,hr_classes)

        if self.reduction:
            return torch.sum(loss, dim=1).mean()
        else:
            return torch.sum(loss, dim=1)

class SoftNaiveLoss(nn.Module):
    def __init__(self, reduction = True):
        super(SoftNaiveLoss, self).__init__()
        self.reduction = reduction

    # y_true is class
    def forward(self, y_pred, y_true, y_nlcd, from_logits = True, smooth = 0.000001):
        if self.reduction == True:
            reduce = "mean"
        else:
            reduce = "none"
        hr_loss = nn.CrossEntropyLoss(reduction=reduce)
        sr_loss = CrossEntropy(self.reduction)
        if self.reduction:
            return self.hr_weight * hr_loss(y_pred, y_true) + self.sr_weight * sr_loss(y_pred, y_nlcd, from_logits)
        else:
            return self.hr_weight * torch.sum(hr_loss(y_pred, y_true), dim=(1,2)) + self.sr_weight * sr_loss(y_pred, y_nlcd, from_logits)
