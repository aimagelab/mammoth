from .dualcoop import dualcoop
from .build_cfg import setup_cfg


def build_model(args, classnames):
    cfg = setup_cfg(args)
    model = dualcoop(cfg, classnames)
    return model
