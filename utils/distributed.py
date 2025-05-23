
"""Distributed utilities for parallel processing.

Supports both Distributed Data Parallel (DDP) and Data Parallel (DP) models.

Examples:
    >>> from utils.distributed import make_ddp, make_dp
    >>> model = make_ddp(model) # for DDP    >>> model = make_dp(model) # for DP

**Note**:
- DDP is not applicable to rehearsal methods (see `make_ddp` for more details).
- When using DDP, you might need the `wait_for_master` function.
    - Synchronization before and after training is handled automatically.
"""
import logging
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank: int, world_size: int) -> None:
    """
    Set up the distributed environment for parallel processing using Distributed Data Parallel (DDP).

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        None
    """
    host = os.environ['SLURM_NODELIST'].split(',')[0]
    ephemeral_port_range = 65535 - 32768
    port = 32768 + int(os.environ['SLURM_JOBID']) % ephemeral_port_range

    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    logging.info(f"Running basic DDP example on rank {rank}/{world_size} (host {host}, node {os.environ['SLURMD_NODENAME']} port {port}).")
    sys.stdout.flush()
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    logging.info("Inited")
    sys.stdout.flush()


def wait_for_master() -> None:
    """
    Wait for the master process to arrive at the barrier.

    - This is a blocking call.
    - The function is a *no-op* if the current process is the master (or DDP is not used).

    Returns:
        None
    """
    if dist.is_initialized():
        dist.barrier()


def make_ddp(model: torch.nn.Module) -> None:
    """
    Create a DistributedDataParallel (DDP) model.


    *Note*: *DDP is not applicable to rehearsal methods* (e.g., GEM, A-GEM, ER, etc.).
    This is because DDP breaks the buffer, which has to be synchronized.
    Ad-hoc solutions are possible, but they are not implemented here.

    Args:
        model: The model to be wrapped with DDP.

    Returns:
        The DDP-wrapped model.
    """
    if not torch.distributed.is_available() or not torch.cuda.is_available():
        raise ValueError("DDP not available!")

    rank_command = f"scontrol show jobid -d {os.environ['SLURM_JOBID']} | grep ' Nodes='"
    rank_data = os.popen(rank_command).read().splitlines()
    world = {x.split("Nodes=")[1].split(" ")[0]: int(x.split('gpu:')[1].split('(')[0]) for x in rank_data}
    world_size = sum(world.values())
    os.environ['MAMMOTH_WORLD_SIZE'] = str(world_size)

    base_rank = sum([w for x, w in world.items() if x < os.environ['SLURMD_NODENAME']])
    local_gpus = world[os.environ['SLURMD_NODENAME']]

    rankno = 0
    for r in range(local_gpus - 1):
        if os.fork() == 0:
            rankno += 1
            setup(rankno + base_rank, world_size)
            model.to(rankno)
            model.device = f"cuda:{rankno}"
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            os.environ['MAMMOTH_RANK'] = str(rankno + base_rank)
            os.environ['MAMMOTH_SLAVE'] = '1'
            ddp_model = DDP(model, device_ids=[rankno])
            return ddp_model

    setup(base_rank, world_size)
    model.to(0)
    model.device = "cuda:0"
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[0])
    os.environ['MAMMOTH_RANK'] = str(base_rank)
    return ddp_model


class CustomDP(DataParallel):
    """
    Custom DataParallel class to avoid using `.module`.

    """

    is_init = False

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        """
        Initialize the CustomDP class.

        Args:
            module: The module to be wrapped with DataParallel.
            device_ids: The device IDs to be used.
            output_device: The output device.
            dim: The dimension to be used for parallelization.

        Returns:
            None
        """
        self.original_names = [name for name in module.__dict__ if not name.startswith('_')]
        super().__init__(module, device_ids, output_device, dim)

        self.is_init = True

    def __getattr__(self, name: str):
        """
        Get attribute value.

        Args:
            name (str): The name of the attribute.

        Returns:
            The value of the attribute.
        """
        if not self.is_init:
            return super().__getattr__(name)

        if name == 'module':
            return super().__getattr__(name)

        if not name.startswith('_') and name not in self.original_names:
            return getattr(self.module, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        """
        Set attribute value.

        Args:
            name (str): The name of the attribute.
            value: The value to be assigned to the attribute.

        Returns:
            None
        """
        if not self.is_init:
            super().__setattr__(name, value)
        else:
            if not name.startswith('_') and name not in self.original_names:
                setattr(self.module, name, value)
            else:
                super().__setattr__(name, value)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Override default load_state_dict to avoid using `.module`.
        """
        return self.module.load_state_dict(state_dict, strict, assign)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override default state_dict to avoid using `.module`.
        """
        return self.module.state_dict(destination, prefix, keep_vars)


def make_dp(model):
    """
    Create a DataParallel (DP) model.

    Args:
        model: The model to be wrapped with DP.

    Returns:
        The DP-wrapped model.
    """
    return CustomDP(model, device_ids=range(torch.cuda.device_count())).to('cuda:0')
