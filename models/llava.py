from argparse import Namespace
import torch
import torch.nn as nn
from torchvision import transforms

try:
    import bitsandbytes
except ImportError:
    raise ImportError("Please install the BitsAndBytes package by running: `pip install -i https://pypi.org/simple/ bitsandbytes`")

try:
    import accelerate
except ImportError:
    raise ImportError("Please install the accelerate package by running: `pip install accelerate`")

try:
    # from transformers import AutoProcessor, LlavaForConditionalGeneration
    from transformers import pipeline, BitsAndBytesConfig
except ImportError:
    raise ImportError("Please install the HuggingFace Transformers package by running: pip install transformers")

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class FinalModel(nn.Module):
    @torch.no_grad()
    def __init__(self, dataset: ContinualDataset, args: Namespace, denorm_transform, device):
        super().__init__()

        self.denorm_transform = denorm_transform
        self.device = device

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        self.pipe = pipeline("image-to-text", model=args.llava_model_name,
                             model_kwargs={"quantization_config": quantization_config})

        class_names = [' '.join(c.lower().split('_')) for c in dataset.get_class_names()]
        self.class_names = [f'({i+1}) {c}' for i, c in enumerate(class_names)]
        if '<classnames>' in args.classification_prompt:
            classification_prompt = args.classification_prompt.replace('<classnames>', ' '.join(class_names))
        if '<datasetname>' in args.classification_prompt:
            classification_prompt = args.classification_prompt.replace('<datasetname>', dataset.NAME.replace('seq-', '').replace('-224', '').replace('-', ' '))
        self.prompt = args.base_prompt.replace('<prompt>', classification_prompt)
        self.eye = torch.eye(len(class_names))

    def get_closest_classname(self, pred_class_name):
        # get the index of the closest class name
        pred_class_name = pred_class_name.lower().replace('_', ' ').strip()
        closest_class_name = [c for c in self.class_names if pred_class_name in c or any(cs for cs in pred_class_name.split('.') if cs.strip() in c)]
        if len(closest_class_name) == 0:
            return -1
        else:
            return self.class_names.index(closest_class_name[0])

    @torch.no_grad()
    def forward(self, x):
        x = self.denorm_transform(x.cpu())
        outputs = []
        for i in range(len(x)):
            x_pil = transforms.ToPILImage()(x[i])
            out = self.pipe(x_pil, prompt=self.prompt, generate_kwargs={"max_new_tokens": 20})
            outputs.append(out)
        outputs = [output[0]['generated_text'] for output in outputs]

        # Extract the class names from the output
        out_class_names = [output.split('ASSISTANT:')[-1].strip().lower() for output in outputs]

        # Convert the class names to a prediction tensor
        prediction = torch.tensor([self.get_closest_classname(class_name) for class_name in out_class_names])

        preds = torch.zeros(len(prediction), len(self.class_names))
        preds[prediction != -1] = self.eye[prediction[prediction != -1]]

        return preds.to(self.device)


class Llava(ContinualModel):
    """STATIC Continual Learning with LLAVA."""
    NAME = 'llava'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(lr=0, n_epochs=0)  # disable training by default
        parser.add_argument('--llava_model_name', type=str, default='llava-hf/llava-1.5-7b-hf',
                            help='Name of the LLAVA model to use')
        parser.add_argument('--base_prompt', type=str, default="USER: <image>\n<prompt>\nASSISTANT:",
                            help='Base prompt for the LLAVA model')
        parser.add_argument('--classification_prompt', type=str,
                            help='Prompt to use for classification. If <classnames> is present, it will be replaced with the class names. If <datasetname> is present, it will be replaced with the dataset name',
                            default="What object is in the photo ? Pick only one and answer with the class name <classnames>")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        backbone = None
        if args.n_epochs != 0:
            print(f"LLAVA is a STATIC model, setting n_epochs to {0}")
            args.n_epochs = 0
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        denorm_transform = self.dataset.get_denormalization_transform()
        self.net = FinalModel(self.dataset, args, denorm_transform=denorm_transform, device=self.device)

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
