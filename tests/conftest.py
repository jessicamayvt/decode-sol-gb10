import pytest
import torch


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_without_gpu = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_without_gpu)
