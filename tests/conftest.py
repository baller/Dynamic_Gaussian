"""Shared fixtures for W-CVCT-GS tests."""
import pytest
import torch


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _skip_cuda_when_unavailable(request):
    if request.node.get_closest_marker("cuda") and not _cuda_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def device():
    return torch.device("cuda" if _cuda_available() else "cpu")


@pytest.fixture
def small_image():
    """A (1,3,16,16) tensor with a known checkerboard pattern."""
    torch.manual_seed(0)
    return torch.randn(1, 3, 16, 16)


@pytest.fixture
def stereo_pair():
    """A pair of (1,3,16,16) tensors and a (1,1,16,16) disparity in [0,4]."""
    torch.manual_seed(1)
    left = torch.randn(1, 3, 16, 16)
    right = torch.randn(1, 3, 16, 16)
    disp = torch.rand(1, 1, 16, 16) * 4.0
    return left, right, disp
