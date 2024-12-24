"""
Adaptation of OpenAI's CLIP.
Requires:
- pip install git+https://github.com/openai/CLIP.git

.. note::
    Checkpoints are loaded from the OpenAI repository.
    * RN50: "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
    * RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"
    * RN50x4: "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt"
    * RN50x16: "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt"
    * RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt"
    * ViT-B/32: "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    * ViT-B/16: "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
    * ViT-L/14: "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
    * ViT-L/14@336px: "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
"""

import torch
import torch.nn as nn

from utils import binary_to_boolean_type
try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.conf import get_device


CUSTOM_TEMPLATES = {  # from https://github.com/KaiyangZhou/CoOp
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class FinalModel(nn.Module):
    @torch.no_grad()
    def __init__(self, clip_model, dataset: ContinualDataset, args) -> None:
        super().__init__()
        self.dataset = dataset
        self.clip_model = clip_model
        self.args = args

        self.classes = self.dataset.get_class_names()
        if args.use_templates:
            templates = self.dataset.get_prompt_templates()
            text_inputs = []
            for t in templates:
                t_inputs = torch.cat([clip.tokenize(t.format(c)) for c in self.classes]).to(get_device())
                t_inputs = self.clip_model.encode_text(t_inputs)
                t_inputs /= t_inputs.norm(dim=-1, keepdim=True)  # double normalization if use templates is expected (see https://github.dev/KaiyangZhou/CoOp)
                text_inputs.append(t_inputs)
            self.text_features = torch.stack(text_inputs).mean(0)
        else:
            template = "a photo of a {}"
            cname = [cname for cname in CUSTOM_TEMPLATES if cname.lower() in self.dataset.NAME.lower().replace('-', '').replace('_', '')]
            if len(cname) > 0:  # if dataset has custom templates
                template = CUSTOM_TEMPLATES[cname[0]]
            text_inputs = torch.cat([clip.tokenize(template.format(c)) for c in self.classes]).to(get_device())
            self.text_features = self.clip_model.encode_text(text_inputs)

        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)  # double normalization if use templates is expected
        self.task_id = 0

    @torch.no_grad()
    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        text_features = self.text_features

        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * (image_features @ text_features.T)).softmax(dim=-1)

        return similarity


class CLIP(ContinualModel):
    """STATIC Continual Learning with CLIP"""
    NAME = 'clip'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(lr=0, n_epochs=0)  # disable training by default
        parser.add_argument('--clip_backbone', type=str, default='ViT-L/14',
                            choices=list(clip.available_models()),
                            help='Backbone architecture for CLIP')
        parser.add_argument('--save_predictions', type=binary_to_boolean_type, default=0,
                            help='Whether to save predictions of the TRAINING set after each task')
        parser.add_argument('--use_templates', type=binary_to_boolean_type, default=0,
                            help='Whether to use prompt templates for CLIP. NOTE: Datasets NEED to have a `get_prompt_templates` method implemented.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        backbone, clip_transform = clip.load(args.clip_backbone, device=get_device())
        n_epochs = 1 if args.save_predictions else 0
        if args.n_epochs != n_epochs:
            print(f"CLIP is a STATIC model, setting n_epochs to {n_epochs}")
            args.n_epochs = n_epochs
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.net = FinalModel(self.net, self.dataset, args)
        self.clip_transform = clip_transform

        self.predictions = []
        self.original_labels = []

    def begin_task(self, dataset):
        dataset.test_loaders[-1].dataset.transform = self.clip_transform
        if self.args.save_predictions:
            dataset.train_loader.dataset.transform = self.clip_transform

        if self.current_task != 0:
            self.net.task_id += 1

        self.eval()

    def end_task(self, dataset: ContinualDataset) -> None:
        if self.args.save_predictions:
            self.predictions = torch.cat(self.predictions, dim=0).cpu()
            self.original_labels = torch.cat(self.original_labels, dim=0).cpu()
            torch.save((self.predictions, self.original_labels), f'predictions_{self.args.dataset}_{self.current_task}.pt')
            print(f"Predictions saved for task {self.current_task} in 'predictions_{self.args.dataset}_{self.current_task}.pt'")
            self.predictions = []
            self.original_labels = []
        return super().end_task(dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.args.save_predictions:
            with torch.no_grad():
                self.predictions.append(self.net(inputs))
                self.original_labels.append(labels)
        return 0

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]
