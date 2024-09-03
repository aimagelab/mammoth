import torch
import torch.nn as nn
from torchvision import transforms

from utils import binary_to_boolean_type
try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
except ImportError:
    raise ImportError("Please install the HuggingFace Transformers package by running: pip install transformers")

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.conf import get_device


class FinalModel(nn.Module):
    @torch.no_grad()
    def __init__(self, backbone, dataset: ContinualDataset, args) -> None:
        super().__init__()
        self.backbone = backbone
        self.preprocess = AutoProcessor.from_pretrained(args.llava_model_name)

        class_names = [c.lower() for c in dataset.get_class_names()]
        if '<classnames>' in args.classification_prompt:
            class_names = [f'({i}) {c}' for i, c in enumerate(class_names)]
            classification_prompt = args.classification_prompt.replace('<classnames>', ', '.join(class_names))
        if '<datasetname>' in args.classification_prompt:
            classification_prompt = args.classification_prompt.replace('<datasetname>', dataset.NAME.replace('seq-', '').replace('-224', '').replace('-', ' '))
        self.prompt = args.base_prompt.replace('<prompt>', classification_prompt)
        self.eye = torch.eye(len(class_names))

    @torch.no_grad()
    def forward(self, x):
        x = self.preprocess(text=[self.prompt] * len(x), images=x, return_tensors='pt').to(self.backbone.device)
        outputs = self.backbone.generate(**x, max_new_tokens=4)

        # Extract the class names from the output
        class_names = [self.preprocess.decode(output, skip_special_tokens=True).split('ASSISTANT:')[-1].strip().lower() for output in outputs]

        # Convert the class names to a prediction tensor
        prediction = torch.tensor([class_names.index(class_name) for class_name in class_names])

        return self.eye[prediction].to(self.backbone.device)


class Llava(ContinualModel):
    """STATIC Continual Learning with LLAVA."""
    NAME = 'llava'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(lr=0, n_epochs=0)  # disable training by default
        parser.add_argument('--llava_model_name', type=str, default='llava-hf/llava-1.5-7b-hf',
                            help='Name of the LLAVA model to use')
        parser.add_argument('--base_prompt', type=str, default="USER: <image>\n<prompt> ASSISTANT:",
                            help='Base prompt for the LLAVA model')
        parser.add_argument('--classification_prompt', type=str,
                            help='Prompt to use for classification. If <classnames> is present, it will be replaced with the class names. If <datasetname> is present, it will be replaced with the dataset name',
                            default="Answer with only a single class name, what class of <datasetname> is in the image?")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        backbone = None
        if args.n_epochs != 0:
            print(f"CLIP is a STATIC model, setting n_epochs to {0}")
            args.n_epochs = 0
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        backbone = LlavaForConditionalGeneration.from_pretrained(args.llava_model_name, torch_dtype=torch.float16).to(self.device, dtype=torch.float16)
        self.net = FinalModel(backbone, self.dataset, args)

        self.predictions = []
        self.original_labels = []

    def begin_task(self, dataset):
        dataset.test_loaders[-1].dataset.transform = transforms.ToTensor()
        self.eval()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        # do nothing
        return 0

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]
