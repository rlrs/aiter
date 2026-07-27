# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

import os


def test_copytree_to_build_keeps_merged_tree_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("AITER_ROOT_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GPU_ARCHS", "gfx90a")

    from csrc.cpp_itfs.utils import copytree_to_build

    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    dst = tmp_path / "dst"
    (src1 / "nested").mkdir(parents=True)
    (src2 / "nested").mkdir(parents=True)
    (src1 / "nested" / "first.h").write_text("first\n")
    (src2 / "nested" / "second.h").write_text("second\n")

    for path in (
        src1 / "nested" / "first.h",
        src2 / "nested" / "second.h",
    ):
        path.chmod(0o440)
    for path in (src1 / "nested", src1, src2 / "nested", src2):
        path.chmod(0o550)

    copytree_to_build(str(src1), str(dst))
    copytree_to_build(str(src2), str(dst))
    (dst / "nested" / "third.h").write_text("third\n")

    assert (dst / "nested" / "first.h").read_text() == "first\n"
    assert (dst / "nested" / "second.h").read_text() == "second\n"
    assert os.access(dst / "nested", os.W_OK)
