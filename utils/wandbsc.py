
try:
    import wandb
    try:
        import wandbbq
    except ImportError:
        wandbbq = None
except ImportError:
    wandb = None
from argparse import Namespace
from utils import random_id


def innested_vars(args: Namespace):
    new_args = vars(args).copy()
    for key, value in new_args.items():
        if isinstance(value, Namespace):
            new_args[key] = innested_vars(value)
    return new_args


class WandbLogger:
    def __init__(self, args: Namespace, prj='default', entity='regaz', name=None):
        self.active = not args.nowand
        self.run_id = random_id(5)

        if self.active:
            assert wandb is not None, "Wandb not installed, please install it or run without wandb"
            if name is not None:
                name += f'-{self.run_id}'
            if wandbbq is not None:
                wandbbq.init(project=prj, entity=entity, config=innested_vars(args), name=name)
            else:
                wandb.init(project=prj, entity=entity, config=innested_vars(args), name=name)
            self.wandb_url = wandb.run.get_url()
            args.wandb_url = self.wandb_url

    def __call__(self, obj: any, **kwargs):
        if self.active:
            wandb.log(obj, **kwargs)

    def finish(self):
        if self.active:
            wandb.finish()
