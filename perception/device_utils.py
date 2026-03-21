"""
Torch device selection (CUDA, Apple MPS, CPU) for perception models.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_torch_device(*, use_gpu: bool, prefer_mps: bool = True) -> torch.device:
    """
    Pick the best available device for inference.

    Priority when use_gpu is True: CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if not use_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("Using CUDA for perception: %s", dev)
        return dev
    if prefer_mps:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            dev = torch.device("mps")
            logger.info("Using Apple MPS for perception: %s", dev)
            return dev
    logger.info("Using CPU for perception (no CUDA/MPS)")
    return torch.device("cpu")
