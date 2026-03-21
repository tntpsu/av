"""Tests for perception.device_utils.resolve_torch_device."""

from __future__ import annotations

from unittest.mock import patch

from perception.device_utils import resolve_torch_device


def test_resolve_cpu_when_use_gpu_false() -> None:
    d = resolve_torch_device(use_gpu=False, prefer_mps=True)
    assert d.type == "cpu"


@patch("perception.device_utils.torch.cuda.is_available", return_value=True)
def test_cuda_when_available(_mock_cuda: object) -> None:
    d = resolve_torch_device(use_gpu=True, prefer_mps=True)
    assert str(d) == "cuda"


@patch("perception.device_utils.torch.cuda.is_available", return_value=False)
@patch("perception.device_utils.torch.backends.mps.is_available", return_value=True)
def test_mps_when_cuda_unavailable(_mock_cuda: object, _mock_mps: object) -> None:
    d = resolve_torch_device(use_gpu=True, prefer_mps=True)
    assert str(d) == "mps"


@patch("perception.device_utils.torch.cuda.is_available", return_value=False)
@patch("perception.device_utils.torch.backends.mps.is_available", return_value=False)
def test_cpu_when_mps_unavailable(_mock_cuda: object, _mock_mps: object) -> None:
    d = resolve_torch_device(use_gpu=True, prefer_mps=True)
    assert d.type == "cpu"


@patch("perception.device_utils.torch.cuda.is_available", return_value=False)
@patch("perception.device_utils.torch.backends.mps.is_available", return_value=True)
def test_cpu_when_prefer_mps_false(_mock_cuda: object, _mock_mps: object) -> None:
    d = resolve_torch_device(use_gpu=True, prefer_mps=False)
    assert d.type == "cpu"
