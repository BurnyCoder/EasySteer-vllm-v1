# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms.cuda import NvmlCudaPlatform, pynvml


def test_get_device_capability_returns_none_on_invalid_nvml_device(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)

    def raise_invalid_argument(_: int):
        raise pynvml.NVMLError_InvalidArgument()

    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", raise_invalid_argument)

    assert NvmlCudaPlatform.get_device_capability(99) is None
    assert not NvmlCudaPlatform.has_device_capability(80, device_id=99)
