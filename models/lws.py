
from argparse import ArgumentParser
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
try:
    from kmeans_pytorch import kmeans
except ImportError:
    raise ImportError('kmeans_pytorch not installed. Please run `pip install kmeans-pytorch`.')

from utils.args import add_rehearsal_args
from utils.buffer_lws import Buffer
from models.utils.continual_model import ContinualModel


class LwS(ContinualModel):
    """
    Implementation of "Towards Unbiased Continual Learning: Avoiding Forgetting in the Presence of Spurious Correlations"
    """
    NAME = 'lws'
    COMPATIBILITY = ['biased-class-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)

        parser.add_argument('--buf_lambda_logits', type=float, default=1,
                            help='Penalty weight BCE past logits.')
        parser.add_argument('--kd_lambda', type=float, default=1,
                            help='Penalty weight MSE clusters Logits (fixed to 1, not searched)')
        parser.add_argument('--buf_lambda_clusters', type=float, default=1,
                            help='Penalty weight BCE past clusters.')
        parser.add_argument('--gamma', type=float, default=1,
                            help='Weight cluster contribution (Eq. 3 and 4)')
        parser.add_argument('--k', type=int, default=8,
                            help='Number of clusters')
        parser.add_argument('--n_bin', type=int, default=4,
                            help='Number of bins')
        parser.add_argument('--momentum', type=float, default=0.3,
                            help='Momentum for weights update')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(LwS, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size, self.device, n_tasks=self.n_tasks,
                             attributes=['examples', 'labels', 'logits', 'task_labels', 'clusters_labels',
                                         'clusters_logits', 'loss_values'],
                             n_bin=self.args.n_bin)

        self.pseudo_labels = {}
        self.weights = {}
        self.cluster_losses = {}
        self.target_losses = {}
        self.avg_cluster_losses = {}
        self.distances = {}
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_clusters = self.args.k

        self.init_classifiers()

        self.init_net = copy.deepcopy(self.net)
        self.init_net = self.init_net.to(self.device)
        self.init_net.eval()

        self.past_task_idx = 0

    def init_classifiers(self):
        self.net.classifier = torch.nn.Identity()

        if self.args.k is not None:
            self.net.cluster_classifiers = nn.ModuleList()
            self.net.cluster_classifiers.append(nn.ModuleList())  # clusters attr=0
            self.net.cluster_classifiers.append(nn.ModuleList())  # clusters attr=1

        self.net.classifiers = nn.ModuleList()

        # initialize classifiers
        for i in range(self.num_classes):
            self.net.classifiers.append(nn.Linear(512, 1))

        # initialize cluster classifiers, one for each task
        for i in range(self.n_tasks):
            if self.args.k is not None:
                self.net.cluster_classifiers[0].append(nn.Linear(512, self.args.k))
                self.net.cluster_classifiers[1].append(nn.Linear(512, self.args.k))

    def freeze_classifiers(self, task_id, freeze_cluster=False):
        # freeze all classifiers except the current one
        for i, classifier in enumerate(self.net.classifiers):
            if i > task_id:
                for param in classifier.parameters():
                    param.requires_grad = False
            else:
                for param in classifier.parameters():
                    param.requires_grad = True

        if freeze_cluster:
            for i, classifier in enumerate(self.net.cluster_classifiers[0]):
                if i != task_id:
                    for param in classifier.parameters():
                        param.requires_grad = False
                else:
                    for param in classifier.parameters():
                        param.requires_grad = True
            for i, classifier in enumerate(self.net.cluster_classifiers[1]):
                if i != task_id:
                    for param in classifier.parameters():
                        param.requires_grad = False
                else:
                    for param in classifier.parameters():
                        param.requires_grad = True

    def get_classes_and_clusters(self, inputs: torch.Tensor):
        features = self.net(inputs)

        outs = [classifier(features) for classifier in self.net.classifiers]
        out = torch.cat(outs, dim=1)
        # clusters classification
        outs_clusters_0 = torch.stack([classifier(features) for classifier in self.net.cluster_classifiers[0]], dim=0)
        outs_clusters_1 = torch.stack([classifier(features) for classifier in self.net.cluster_classifiers[1]], dim=0)

        # Combine the tensors to get the desired shape
        outs_clusters = torch.stack([outs_clusters_0, outs_clusters_1], dim=0)

        return (out, outs_clusters)

    def cluster_counts(self):
        return self.pseudo_labels[self.current_task].bincount(minlength=self.num_clusters)

    def compute_stats(self):
        for l in range(self.num_clusters * 2):
            ids = (self.pseudo_labels[self.current_task] == l).nonzero()

            if ids.size(0) == 0:
                continue

            cluster_losses = self.cluster_losses[self.current_task][ids]
            cluster_losses_nz = (cluster_losses > 0).nonzero()

            target_losses_ = self.target_losses[self.current_task][ids]
            target_losses_nz = (target_losses_ > 0).nonzero()

            if cluster_losses_nz.size(0) > 0:
                self.avg_cluster_losses[self.current_task][l] = self.args.gamma * (cluster_losses[cluster_losses_nz[:, 0]].float().mean(0)) + target_losses_[target_losses_nz[:, 0]].float().mean(0)

    def update_cluster_weights(self, indexes, epoch):
        # Cluster cardinality
        cluster_counts = self.cluster_counts()
        cluster_weights = cluster_counts.sum() / (cluster_counts.float()).to(self.device)

        # Cluster assignments
        assigns_id = self.pseudo_labels[self.current_task][indexes].to(self.device)

        if (self.cluster_losses[self.current_task] > 0).nonzero().size(0) > 0:
            cluster_losses_ = self.avg_cluster_losses[self.current_task].view(-1).to(self.device)
            losses_weight = cluster_losses_.float() / cluster_losses_.sum().to(self.device)
            weights_ = losses_weight[assigns_id].to(self.device) * cluster_weights[assigns_id].to(self.device)

            weights_ /= weights_.mean()
            if epoch > 0:
                weights_ids_ = (1 - self.args.momentum) * self.weights[self.current_task][indexes] + self.args.momentum * weights_
            else:
                weights_ids_ = weights_

            self.weights[self.current_task][indexes] = weights_ids_

            weights_ids_ /= weights_ids_.mean()
        else:
            weights_ids_ = self.weights[self.current_task][indexes]
            weights_ids_ /= weights_ids_.mean()

        return weights_ids_

    def update(self, target_losses, clusters_losses, indexes, epoch):
        # Get Cluster and Attribute losses
        self.cluster_losses[self.current_task][indexes] = clusters_losses.detach()
        self.target_losses[self.current_task][indexes] = target_losses.detach()

        # Update losses values
        self.compute_stats()

        # Update weights
        self.update_cluster_weights(indexes, epoch)

    def get_weights(self, indexes):
        weights = self.weights[self.current_task][indexes]
        return weights

    def get_task_weights(self):
        weights = self.weights[self.current_task]
        return weights.detach().cpu().numpy()

    def extract_features(self, train_loader):
        # features extraction
        features = []
        labels_ = []
        indexes_ = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(train_loader, desc='Extracting features')):
                inputs, labels, indexes = data[0], data[1], data[-1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.init_net(inputs, returnt='features').float()
                features.append(outputs.detach().cpu())
                # if labels has more than one dimension
                if labels.dim() > 1:
                    labels_.append(labels[:, self.current_task].detach().cpu())
                else:
                    labels_.append(labels.detach().cpu())
                indexes_.append(indexes.detach().cpu())
            features = torch.cat(features, dim=0)
            features = torch.nn.functional.normalize(features, dim=1)
            labels_ = torch.cat(labels_, dim=0)
            indexes_ = torch.cat(indexes_, dim=0)

        return features, labels_, indexes_

    def clustering(self, train_loader):
        features, labels_, indexes = self.extract_features(train_loader)
        self.pseudo_labels[self.current_task] = torch.zeros_like(labels_).to(self.device) - 1
        self.distances[self.current_task] = torch.zeros_like(labels_).float().to(self.device)
        self.weights[self.current_task] = torch.ones_like(labels_).float().to(self.device)
        self.cluster_losses[self.current_task] = torch.zeros_like(labels_).float().to(self.device)
        self.target_losses[self.current_task] = torch.zeros_like(labels_).float().to(self.device)
        self.avg_cluster_losses[self.current_task] = torch.zeros(self.num_clusters * 2).float().to(self.device)
        self.initial_losses = torch.zeros_like(labels_).float().to(self.device)

        for l in range(2):
            target_assigns = (labels_ == l).nonzero().squeeze()
            feautre_assigns = features[target_assigns]
            indexes_assigns = indexes[target_assigns]

            cluster_ids, cluster_centers = kmeans(X=feautre_assigns, num_clusters=self.num_clusters, distance='cosine', device=self.device)
            self.pseudo_labels[self.current_task][indexes_assigns] = cluster_ids.to(self.device) + l * self.num_clusters
            # distances
            similarity = F.cosine_similarity(feautre_assigns, cluster_centers[cluster_ids])
            distances = (1 - similarity).to(self.device)
            self.distances[self.current_task][indexes_assigns] = distances / distances.max()

    def begin_task(self, dataset):
        dataset.train_loader.dataset.indexes = np.arange(len(dataset.train_loader.dataset))
        if dataset.NAME == 'seq-celeba':
            self.freeze_classifiers(self.current_task, freeze_cluster=False)
        self.get_optimizer()
        self.clustering(dataset.train_loader)

    def begin_epoch(self, epoch, dataset):
        if epoch == 5:
            self.get_initial_losses(dataset)

    def end_task(self, dataset):
        self.buffer.reset_bins()

    def get_initial_losses(self, dataset):
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):
                inputs, labels, indexes = data[0], data[1], data[-1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.get_classes_and_clusters(inputs)
                outputs_original = outputs[0].float()

                # Get task specific outputs
                if labels.dim() > 1:
                    task_specific_labels = labels[:, self.current_task]
                    task_specific_outputs = outputs_original[:, self.current_task]
                else:
                    task_specific_labels = labels
                    task_specific_outputs = outputs_original

                task_specific_labels = task_specific_labels.float()

                targets_losses = self.loss(task_specific_outputs.squeeze(), task_specific_labels.squeeze()).detach()
                self.initial_losses[indexes] = targets_losses.detach()

    def forward(self, inputs):
        return self.get_classes_and_clusters(inputs)[0]

    def observe(self, inputs, labels, not_aug_inputs, epoch, indexes):
        task = (torch.ones(labels.shape[0]) * self.current_task).to(self.device, dtype=torch.long)

        self.opt.zero_grad()
        outputs = self.get_classes_and_clusters(inputs)

        # Get net outputs
        outputs_original = outputs[0].float()
        outputs_clusters_ = outputs[1]

        # Get task specific outputs
        if labels.dim() > 1:
            task_specific_labels = labels[:, self.current_task]
            task_specific_outputs = outputs_original[:, self.current_task]
        else:
            task_specific_labels = labels
            task_specific_outputs = outputs_original

        outputs_clusters = outputs_clusters_[task_specific_labels, task, torch.arange(task.size(0))]

        pseudo_labels = self.pseudo_labels[self.current_task][indexes]
        pseudo_labels[task_specific_labels == 1] -= self.num_clusters
        task_specific_labels = task_specific_labels.float()

        # Compute losses
        targets_losses = self.loss(task_specific_outputs.squeeze(), task_specific_labels.squeeze())
        clusters_losses = self.ce(outputs_clusters, pseudo_labels)

        # Get weights per sample
        weights = self.get_weights(indexes)

        # Weighted Target Loss + Cluster Loss
        loss_stream = torch.mean(targets_losses * weights) + self.args.gamma * torch.mean(clusters_losses)

        if epoch > 1:
            # add data to buffer
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 task_labels=task,
                                 logits=outputs_original.data,
                                 clusters_logits=outputs_clusters.data,
                                 clusters_labels=pseudo_labels,
                                 loss_values=self.initial_losses[indexes].detach())

        if not self.buffer.is_empty():
            ################## FIRST FORWARD PASS ##################
            ########################################################

            buf_inputs, buf_labels, _, buf_tasks, clusters_labels, clusters_logits, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

            buf_outputs = self.get_classes_and_clusters(buf_inputs)
            outputs_original = buf_outputs[0].float()
            outputs_clusters_ = buf_outputs[1]

            # 1 ### MSE LOSS PAST TASKS CLUSTERS LOGITS ###
            # SELECT CLUSTERS SETS
            if buf_labels.dim() > 1:
                selected_elements = buf_labels[torch.arange(buf_labels.size(0)), buf_tasks]
            else:
                selected_elements = buf_labels

            outputs_clusters = outputs_clusters_[selected_elements, buf_tasks, torch.arange(buf_tasks.size(0))]

            loss_kl_all = F.mse_loss(outputs_clusters, clusters_logits, reduction='none')
            loss_kl = self.args.kd_lambda * (loss_kl_all.mean(dim=1)).mean()

            ################## SECOND FORWARD PASS #################
            ########################################################

            buffer_indexes, buf_inputs, buf_labels, buf_logits, buf_tasks, clusters_labels, clusters_logits, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)

            buf_outputs = self.get_classes_and_clusters(buf_inputs)
            outputs_original = buf_outputs[0].float()
            outputs_clusters_ = buf_outputs[1]

            # 2 ### BCE LOSS TASK ATTRIBUTE ###
            if buf_labels.dim() > 1:
                loss_buf_labels_all = self.loss(outputs_original[torch.arange(buf_labels.size(0)), buf_tasks], buf_labels[torch.arange(buf_labels.size(0)), buf_tasks].float())
            else:
                loss_buf_labels_all = self.loss(outputs_original.squeeze(), buf_labels.float())
            loss_buf_labels = self.args.buf_lambda_logits * (loss_buf_labels_all).mean()

            # 3 ### CROSS ENTROPY LOSS CLUSTERS ASSIGNMENTS ###
            if buf_labels.dim() > 1:
                selected_elements = buf_labels[torch.arange(buf_labels.size(0)), buf_tasks]
            else:
                selected_elements = buf_labels

            outputs_clusters = outputs_clusters_[selected_elements, buf_tasks, torch.arange(buf_tasks.size(0))]

            loss_buf_clusters_all = self.ce(outputs_clusters, clusters_labels)
            loss_buf_clusters = self.args.buf_lambda_clusters * (loss_buf_clusters_all).mean()

            self.buffer.update_losses(loss_buf_labels_all.detach(), buffer_indexes)
            loss = loss_stream + loss_buf_labels + loss_buf_clusters + loss_kl
        else:
            loss = loss_stream

        loss.backward()
        self.opt.step()
        self.update(targets_losses, clusters_losses, indexes, epoch)

        return loss.item()
