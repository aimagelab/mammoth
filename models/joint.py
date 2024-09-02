"""
This module contains the implementation of the Joint CL model.

The Joint model is the upper bound of the CL scenario, as it has access to all the data from all the tasks.
This model is required for the `domain-il` scenario, while `class-il` and `task-il` scenarios can use the `--joint=1` flag.
"""

import math
import torch
from models.utils.continual_model import ContinualModel
from tqdm import tqdm

from utils.conf import create_seeded_dataloader
from utils.schedulers import get_scheduler


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Joint, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.old_data = []
        self.old_labels = []

    def end_task(self, dataset):
        """
        This version of joint training simply saves all data from previous tasks and then trains on all data at the end of the last one.
        """

        self.old_data.append(dataset.train_loader)
        # train
        if len(dataset.test_loaders) != dataset.N_TASKS:
            return

        all_inputs = []
        all_labels = []
        for source in self.old_data:
            for x, l, _ in source:
                all_inputs.append(x)
                all_labels.append(l)
        all_inputs = torch.cat(all_inputs)
        all_labels = torch.cat(all_labels)
        bs = self.args.batch_size
        scheduler = get_scheduler(self.net, self.args, reload_optim=True)

        joint_dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)
        dataloader = create_seeded_dataloader(self.args, joint_dataset, batch_size=self.args.batch_size, shuffle=True)

        with tqdm(total=self.args.n_epochs * len(dataloader)) as pbar:
            for e in range(self.args.n_epochs):
                pbar.set_description(f"Joint - Epoch {e}", refresh=False)
                for i, (inputs, labels) in enumerate(dataloader):
                    if self.args.debug_mode and i > self.get_debug_iters():
                        break
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    pbar.update(bs)
                    pbar.set_postfix({'loss': loss.item()}, refresh=False)

            if scheduler is not None:
                scheduler.step()

    def observe(self, *args, **kwargs):
        # ignore training on task
        return 0
