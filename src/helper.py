# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
import numpy as np
import torch
from torch import Tensor

import src.models.vision_transformer as vit
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def calc_rankme(embeddings: Tensor, epsilon: float = 1e-7) -> float:
    """
    Calculate the RankMe score (the higher, the better).
    RankMe(Z) = exp (
        - sum_{k=1}^{min(N, K)} p_k * log(p_k)
    ),
    where p_k = sigma_k (Z) / ||sigma_k (Z)||_1 + epsilon
    where sigma_k is the kth singular value of Z.
    where Z is the matrix of embeddings
    RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank
    https://arxiv.org/pdf/2210.02885.pdf
    Args:
        embeddings: the embeddings to calculate the RankMe score for
        epsilon: the epsilon value to use for the calculation. The paper recommends 1e-7 for float32.
    Returns:
        the RankMe score
    """
    # compute the singular values of the embeddings
    _u, s, _vh = torch.linalg.svd(
        embeddings, full_matrices=False
    )  # s.shape = (min(N, K),)

    # normalize the singular values to sum to 1 [[Eq. 2]]
    p = (s / torch.sum(s, axis=0)) + epsilon

    # RankMe score is the exponential of the entropy of the singular values [[Eq. 1]]
    # this is sometimes called the `perplexity` in information theory
    entropy = -torch.sum(p * torch.log(p))
    rankme = torch.exp(entropy).item()

    return rankme


def alpha_req(tensor, s=None, epsilon=1e-12, **_):
    """Implementation of the Alpha-ReQ metric.

    This metric is defined in "α-ReQ: Assessing representation quality in
    self-supervised learning by measuring eigenspectrum decay". Agrawal et al.,
    NeurIPS 2022.

    Args:
      tensor (dense matrix): Input embeddings.
      s (optional, dense vector): Singular values of `tensor`.
      epsilon (float): Numerical epsilon.

    Returns:
      float: Alpha-ReQ metric value.
    """
    s = s.cpu()
    if s is None:
        s = np.linalg.svd(tensor, compute_uv=False)
    n = s.shape[0]
    s = s + epsilon
    features = np.vstack([np.linspace(1, 0, n), np.ones(n)]).T
    a, _, _, _ = np.linalg.lstsq(features, np.log(s), rcond=None)
    return a[0]


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = checkpoint["encoder"]
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading predictor
        pretrained_dict = checkpoint["predictor"]
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint["target_encoder"]
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading optimizer
        opt.load_state_dict(checkpoint["opt"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"loaded optimizers from epoch {epoch}")
        logger.info(f"read-path: {r_path}")
        del checkpoint

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(
    device,
    patch_size=16,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
    use_flash_attn=False,
):
    encoder = vit.__dict__[model_name](img_size=[crop_size], patch_size=patch_size, use_flash_attn=use_flash_attn)
    predictor = vit.__dict__["vit_predictor"](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        use_flash_attn=use_flash_attn
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]

    logger.info("Using AdamW")
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
