"""
This module contains the implementation of the Joint CL model.

The Joint model is the upper bound of the CL scenario, as it has access to all the data from all the tasks.
This model is required for the `domain-il` scenario, while `class-il` and `task-il` scenarios can use the `--joint=1` flag.
"""

import math
import torch
from models.utils.continual_model import ContinualModel
from tqdm import tqdm


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)
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
        scheduler = dataset.get_scheduler(self, self.args)

        with tqdm(total=self.args.n_epochs * len(all_inputs), desc="Training joint") as pbar:
            for e in range(self.args.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i + 1) * bs]
                    labels = all_labels[order][i * bs: (i + 1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    pbar.update(bs)
                    pbar.set_postfix({'loss': loss.item()})

            if scheduler is not None:
                scheduler.step()

    def observe(self, *args, **kwargs):
        # ignore training on task
        return 0
