# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops import attention


def test_asm_dispatch_requires_packaged_architecture(monkeypatch):
    monkeypatch.setattr(attention, "get_gfx", lambda: "gfx90a")

    assert not attention._should_use_asm_kernel(
        num_seqs=512,
        num_heads=4,
        head_size=128,
        kv_cache_tensor_dtype=torch.bfloat16,
        high_precision=1,
    )
