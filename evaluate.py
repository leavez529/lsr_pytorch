import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from util import load_nlcd_stats, do_nlcd_means_tuning
from dataset import LandCoverDataset
from torch.utils.data import DataLoader
import numpy as np

# overall accuracy and class accuracy
def compute_accuracy(pred, true_masks, class_score=False, n_classes=5):
    pred_hot = F.one_hot(pred, n_classes)
    true_masks_hot = F.one_hot(true_masks, n_classes)
    equal_map = torch.eq(pred_hot, true_masks_hot)
    equal_num = torch.sum(equal_map, dim=(-2,-3))
    total_num = torch.numel(true_masks)
    total_equal = torch.sum(torch.eq(pred, true_masks))
    accu = total_equal / total_num
    shape = equal_map.shape
    map_num = shape[-2] * shape[-3]
    class_accu = equal_num / map_num
    class_accu = class_accu.mean(dim=0)
    if class_score:
        return accu, class_accu
    else:
        return accu

# overall IoU, mean IoU (neglect unseen class) and class IoU
def compute_jaccard(pred, true_masks, class_score = False, n_classes=5):
    pred_hot = F.one_hot(pred, n_classes)
    true_masks_hot = F.one_hot(true_masks, n_classes)
    intersection = true_masks_hot * pred_hot
    union = pred_hot + true_masks_hot - intersection
    intersection = torch.sum(intersection, dim = (-2, -3))
    union = torch.sum(union, dim = (-2, -3))
    class_jac = intersection / union
    jac = torch.sum(intersection) / torch.sum(union)
    mIoU = 0
    for sample in class_jac:
        mIoU += torch.mean(sample[~sample.isnan()])
    mIoU /= len(class_jac)
    if class_score:
        class_jac = np.nanmean(class_jac.cpu().numpy(), axis = 0)
        class_jac = torch.from_numpy(class_jac)
        return torch.FloatTensor([jac, mIoU]), class_jac
    else:
        return torch.FloatTensor([jac, mIoU])

def developed_accuracy(pred, true_masks, nlcd_masks):
    developed_masks = torch.logical_and(nlcd_masks > 2, nlcd_masks < 7)
    equal_map = torch.eq(pred, true_masks)
    equal_map = equal_map * developed_masks
    equal_num = torch.sum(equal_map)
    total_num = torch.sum(developed_masks)
    return equal_num / total_num

def developed_jaccard(pred, true_masks, nlcd_masks, n_classes=5):
    developed_masks = torch.logical_and(nlcd_masks > 2, nlcd_masks < 7)
    pred_hot = F.one_hot(pred * developed_masks, n_classes)
    true_masks_hot = F.one_hot(true_masks * developed_masks, n_classes)
    intersection = true_masks_hot * pred_hot
    union = pred_hot + true_masks_hot - intersection
    intersection = torch.sum(intersection, dim = (-2, -3))
    union = torch.sum(union, dim = (-2, -3))
    if len(intersection.shape) == 2:
        intersection = intersection[:,1:]
        union = union[:,1:]
    else:
        intersection = intersection[1:]
        union = union[1:]
    class_jac = intersection / union
    jac = torch.sum(intersection) / torch.sum(union)
    mIoU = 0
    step = 0
    for sample in class_jac:
        sample_mIoU = torch.mean(sample[~sample.isnan()])
        if not sample_mIoU.isnan():
            mIoU += sample_mIoU
            step += 1
    mIoU /= step
    return torch.FloatTensor([jac, mIoU])

def multiclass_jaccard(input, target, reduce_batch_first=True, class_score=False, epsilon=1e-6, mask=None):
    assert input.size() == target.size()
    jac = 0
    jac_class = []
    for channel in range(input.shape[1]):
        new_mask = mask[:, 0, ...] if mask is not None else None
        j = jaccard(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon, new_mask)
        if class_score:
            jac_class.append(j)
        jac += j
    if class_score:
        return jac / input.shape[1], torch.Tensor(jac_class)
    return jac / input.shape[1]

def jaccard(input, target, reduce_batch_first=True, epsilon=1e-6, mask=None):
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        if mask is not None:
            input = torch.mul(input, mask).float()
            target = torch.mul(target, mask).float()
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        return (inter + epsilon) / (sets_sum - inter + epsilon)
    else:
        # compute and average metric for each batch element
        j = 0
        for i in range(input.shape[0]):
            new_mask = mask[i] if mask is not None else None
            j += jaccard(input[i, ...], target[i, ...], False, epsilon=epsilon, mask=new_mask)
        return j / input.shape[0]

def evaluate(net, dataloader, device, n_classes=4):
    net.eval()
    # num_val_batches = len(dataloader)
    num_val_batches = 200
    iou_score = 0
    iou_class = 0
    step = 0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['label_hr']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # ignore 0 class
        if n_classes >= 4:
            mask_true -= 1
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            pred = F.softmax(mask_pred, dim=1).argmax(dim=1)
            i, i_class = multiclass_jaccard(F.one_hot(pred, n_classes).permute((0, 3, 1, 2)).float(), F.one_hot(mask_true, n_classes).permute((0, 3, 1, 2)).float(), class_score=True)
            iou_score += i.item()
            iou_class += i_class
        step += 1
        if step == num_val_batches:
            break

    net.train()
    return iou_score / num_val_batches, iou_class / num_val_batches