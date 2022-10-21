import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    host = os.environ['SLURM_NODELIST'].split(',')[0]
    ephemeral_port_range = 65535 - 32768
    port = 32768 + int(os.environ['SLURM_JOBID']) % ephemeral_port_range

    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    print(f"Running basic DDP example on rank {rank}/{world_size} (host {host}, node {os.environ['SLURMD_NODENAME']} port {port}).")
    sys.stdout.flush()
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print("Inited")
    sys.stdout.flush()


def wait_for_master():
    if 'MAMMOTH_RANK' in os.environ:
        dist.barrier()


def make_ddp(model):
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

    intercept_names = ['classifier', 'num_classes', 'set_return_prerelu']

    def __getattr__(self, name: str):
        if name in self.intercept_names:
            return getattr(self.module, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.intercept_names:
            setattr(self.module, name, value)
        else:
            super().__setattr__(name, value)


def make_dp(model):
    return CustomDP(model, device_ids=range(torch.cuda.device_count())).to('cuda:0')
