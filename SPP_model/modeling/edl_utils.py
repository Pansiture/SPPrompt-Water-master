# Evidential Deep Learning (EDL) utilities for binary segmentation
# Based on Sensoy et al., 2018 (Evidential Deep Learning)
import torch
import torch.nn.functional as F


def evidence_to_prob_uncertainty(evidence, K=2):
    """
    Convert evidence to probability and uncertainty (vacuity).
    Args:
        evidence: (B, K, H, W) non-negative evidence
    Returns:
        prob: (B, K, H, W) class probabilities
        uncertainty: (B, 1, H, W) uncertainty (vacuity = K / S)
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
    prob = alpha / S
    uncertainty = K / S
    return prob, uncertainty


def edl_mse_loss(alpha, target, K=2):
    """
    EDL MSE loss for segmentation.
    Args:
        alpha: (B, K, H, W) Dirichlet parameters
        target: (B, 1, H, W) with values 0 or 1
    Returns:
        loss: scalar
    """
    S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
    prob = alpha / S  # (B, K, H, W)

    # one-hot encoding
    y = torch.zeros_like(alpha)
    y[:, 0:1, :, :] = (target == 0).float()
    y[:, 1:2, :, :] = (target == 1).float()

    # MSE loss per class: (y - p)^2 + p(1-p)/(S+1)
    mse = (y - prob).pow(2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    var = (prob * (1 - prob) / (S + 1)).sum(dim=1, keepdim=True)  # (B, 1, H, W)

    return (mse + var).mean()


def edl_kl_loss(alpha, target, K=2):
    """
    KL divergence: KL(Dir(alpha) || Dir([1,1,...,1])).
    Regularization term to avoid over-confident predictions.
    """
    S = alpha.sum(dim=1)  # (B, H, W)

    # KL for K=2 (binary)
    kl = torch.lgamma(S) \
         - torch.lgamma(alpha[:, 0, :, :]) - torch.lgamma(alpha[:, 1, :, :]) \
         + (alpha[:, 0, :, :] - 1) * (torch.digamma(alpha[:, 0, :, :]) - torch.digamma(S)) \
         + (alpha[:, 1, :, :] - 1) * (torch.digamma(alpha[:, 1, :, :]) - torch.digamma(S))

    return kl.mean()


def edl_loss(evidence, target, epoch_num, num_epochs, K=2, kl_anneal_epochs_ratio=0.5, kl_scale=0.5):
    """
    Combined EDL loss with KL annealing.
    Args:
        evidence: (B, K, H, W) non-negative evidence
        target: (B, 1, H, W)
        epoch_num: current epoch (0-based)
        num_epochs: total epochs
        kl_anneal_epochs_ratio: ratio of total epochs for KL annealing (default 0.5)
        kl_scale: global scaling factor for KL term (default 0.5 to soften regularization)
    Returns:
        total_loss: scalar
        mse_loss: scalar
        kl_loss: scalar
        annealing_coef: scalar
    """
    alpha = evidence + 1.0

    mse = edl_mse_loss(alpha, target, K=K)
    kl = edl_kl_loss(alpha, target, K=K)

    # Linear annealing: first kl_anneal_epochs_ratio of epochs
    annealing_coef = min(1.0, epoch_num / max(1, num_epochs * kl_anneal_epochs_ratio))

    total_loss = mse + kl_scale * annealing_coef * kl
    return total_loss, mse, kl, annealing_coef


def sam_referee_loss(evidence, sam_prob, target_mask=None):
    """
    SAM semantic referee consistency loss.
    Forces the EDL probability to align with the frozen SAM prediction.
    Args:
        evidence: (B, 2, H, W)
        sam_prob: (B, 1, H, W) foreground probability from frozen SAM
        target_mask: optional (B, 1, H, W) to only apply loss on uncertain regions
    Returns:
        loss: scalar
    """
    prob, uncertainty = evidence_to_prob_uncertainty(evidence)
    prob_fg = prob[:, 1:2, :, :]  # (B, 1, H, W)

    # MSE with SAM referee
    diff = (prob_fg - sam_prob).pow(2)

    if target_mask is not None:
        # Apply only on uncertain regions (u > 0.5) or all regions
        diff = diff * target_mask

    return diff.mean()


def uncertainty_guided_fusion(evidence_list, K=2):
    """
    UGPF: Uncertainty-Guided Progressive Fusion across scales.
    Args:
        evidence_list: list of [evidence_0, evidence_1, ...] where evidence_k is (B, K, H, W)
    Returns:
        fused_prob: (B, 1, H, W) fused foreground probability
        fused_uncertainty: (B, 1, H, W) fused uncertainty
    """
    probs = []
    uncertainties = []
    for evidence in evidence_list:
        prob, unc = evidence_to_prob_uncertainty(evidence, K=K)
        probs.append(prob[:, 1:2, :, :])  # foreground prob
        uncertainties.append(unc)

    # Weight by (1 - uncertainty), normalize
    weights = [1.0 - u for u in uncertainties]
    sum_weights = sum(weights)

    fused_prob = sum(w * p for w, p in zip(weights, probs)) / (sum_weights + 1e-6)
    fused_uncertainty = 1.0 - sum_weights / (len(weights) * (sum_weights + 1e-6))
    # Better: average uncertainty weighted by confidence
    fused_uncertainty = sum(w * u for w, u in zip(weights, uncertainties)) / (sum_weights + 1e-6)

    return fused_prob, fused_uncertainty
