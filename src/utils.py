import cv2
import numpy as np
import torch


def image_to_tensor(img_rgb: np.ndarray, size: int | None = None) -> torch.Tensor:
    """Convert an HxWx3 uint8 RGB array to a (1,3,H,W) float32 tensor in [0,1]."""
    if size is not None:
        img_rgb = cv2.resize(img_rgb, (size, size))
    tensor = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
    return torch.tensor(tensor).unsqueeze(0)


def load_model(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.contiguous().view(pred.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)
    intersect = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return ((2 * intersect + eps) / (denom + eps)).mean()

def iou_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Calculate Intersection over Union (IoU) for binary masks."""
    pred_flat = pred.contiguous().view(pred.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)
    intersect = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersect
    return ((intersect + eps) / (union + eps)).mean()

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy for binary classification with logits."""
    preds_binary = (torch.sigmoid(outputs) > 0.5).float()
    correct = (preds_binary == targets).sum().item()
    return correct / targets.numel()
