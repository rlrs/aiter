# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os
from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist

from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

set_start_method("spawn", force=True)


def _run_rank(
    tp_size: int,
    rank: int,
    iterations: int,
    distributed_init_method: str,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, 1)

    tp_group = get_tp_group()
    group = tp_group.device_group
    custom = tp_group.device_communicator.ca_comm
    if custom is None or custom.disabled:
        raise RuntimeError("custom all-reduce did not initialize")

    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    try:
        shapes = ((1, 4096), (2, 4096), (8, 4096))
        rank_sum = tp_size * (tp_size - 1) // 2
        for iteration in range(iterations):
            for shape in shapes:
                values = torch.arange(
                    shape[0] * shape[1],
                    dtype=torch.int32,
                    device=device,
                ).remainder_(17)
                source = (values + rank).reshape(shape).to(torch.bfloat16)
                expected = (
                    values * tp_size + rank_sum
                ).reshape(shape).to(torch.bfloat16)

                output = custom.custom_all_reduce(source, use_new=False)
                if output is None:
                    raise RuntimeError(
                        f"custom all-reduce rejected {shape=}"
                    )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output,
                    expected,
                    rtol=0,
                    atol=0,
                    msg=(
                        "custom all-reduce input visibility failed for "
                        f"{shape=} {iteration=} {rank=}"
                    ),
                )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check custom all-reduce input visibility across sizes"
    )
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49413"
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=args.tp) as pool:
        results = [
            pool.apply_async(
                _run_rank,
                args=(args.tp, rank, args.iterations, init_method),
            )
            for rank in range(args.tp)
        ]
        for result in results:
            result.get()


if __name__ == "__main__":
    main()
