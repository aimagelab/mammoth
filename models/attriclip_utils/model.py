import torch
import torch.nn as nn

from copy import deepcopy

from models.attriclip_utils.clip.clip_2 import load, tokenize
from models.attriclip_utils.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import time
from models.attriclip_utils.utils import build_cosine_scheduler


class PromptLearner(nn.Module):
    def __init__(self, device, args, class_names, clip_model, text_prompt, n_ctx=12, prompt_pos=2):
        super().__init__()
        self.device = device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        self.clip_model = clip_model
        self.args = args
        n_cls = len(class_names)
        self.dtype = dtype

        prompt_prefix = ' '.join(['x'] * n_ctx * self.args.text_prompt)
        prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]  # xxxxxx class
        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names]
        self.prompt_pos = prompt_pos

        self.text_prompt = text_prompt
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])  # tokenize class names
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :])  # first token is SOS (start of sequence)
        self.register_buffer('token_suffix', embedding[:, 1 + (n_ctx * self.args.text_prompt):, :])

        nc_prompts = [prompt_prefix + '.']  # xxxxxxxxxxxxxxxxxxxxx.
        nc_tokenized_prompts = torch.cat([tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts.to(self.device)).type(self.dtype)
        self.register_buffer('nc_token_prefix', embedding[:, :1, :])
        self.register_buffer('nc_token_suffix', embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim

    def forward(self, indices, test_class=None, infer=False):
        if test_class is not None:
            prompt_prefix = ' '.join(['x'] * self.n_ctx * self.args.text_prompt)
            prompts = [prompt_prefix + ' ' + name + '.' for name in test_class]
            self.name_lens = [len(_tokenizer.encode(name)) for name in test_class]

            self.prompt_pos = self.prompt_pos

            tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
            self.tokenized_prompts = tokenized_prompts
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)
            self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS, [n_cls, 1, ctx_dim]
            self.register_buffer('token_suffix', embedding[:, 1 + (self.n_ctx * self.args.text_prompt):, :])  # CLS, EOS, [n_cls, -1, ctx_dim]
            self.n_cls = len(test_class)
        batch = indices.shape[0]
        ctx = self.text_prompt[indices].view(batch, self.n_ctx * self.args.text_prompt, self.ctx_dim)
        tokenized_prompts = self.tokenized_prompts.view(self.n_cls, -1)
        n_cls = self.n_cls

        if self.prompt_pos == 2:
            prefix = self.token_prefix.unsqueeze(0).repeat(batch, 1, 1, 1)
            suffix = self.token_suffix.unsqueeze(0).repeat(batch, 1, 1, 1)
            ctx = ctx.unsqueeze(1).repeat(1, n_cls, 1, 1)
            prompts = torch.cat([prefix, ctx, suffix], dim=2)
        elif self.prompt_pos == 1:
            prompts = []
            half_n_ctx = self.n_ctx // 2
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i + 1, :, :].unsqueeze(1)
                class_i = self.token_suffix[i:i + 1, :name_len, :].unsqueeze(1)
                suffix_i = self.token_suffix[i:i + 1, name_len:, :].unsqueeze(1)
                ctx_i_half1 = ctx[:, :half_n_ctx, :].unsqueeze(0)
                ctx_i_half2 = ctx[:, half_n_ctx:, :].unsqueeze(0)
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.prompt_pos == 0:
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i + 1, :, :].unsqueeze(1)
                class_i = self.token_suffix[i:i + 1, :name_len, :].unsqueeze(1)
                suffix_i = self.token_suffix[i:i + 1, name_len:, :].unsqueeze(1)
                ctx_i = ctx.unsqueeze(0)
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        prompts = prompts.squeeze(2).view(batch * self.n_cls, -1, self.ctx_dim)
        tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(batch, 1, 1).view(batch * self.n_cls, -1)
        self.prompts = prompts
        self.prompts_token = tokenized_prompts
        if infer:
            return prompts, tokenized_prompts
        else:
            nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

    def only_prefix(self):
        ctx = self.text_prompt
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return nc_prompts, nc_tokenized_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CLIP(nn.Module):
    def __init__(self, device, args, class_names, clip_model, text_key, text_prompt, n_ctx=12):
        super().__init__()
        self.n_class = len(class_names)
        self.device = device
        self.args = args

        # text enoder
        self.text_encoder = TextEncoder(clip_model)
        # if torch.cuda.device_count() > 1:
        #     self.text_encoder = nn.DataParallel(self.text_encoder)

        self.prompt_learner = PromptLearner(self.device, self.args, class_names, clip_model, text_prompt, n_ctx=n_ctx)
        self.text_key = text_key
        # image encoder
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

    def forward(self, image, test_class=None, test=False):

        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()

        if test:
            n_test = len(test_class)
            probability = image_features @ self.text_key.t()
            _, indices = probability.topk(k=min(self.args.text_prompt, probability.shape[1]), dim=1, largest=True)
            text_prompt, tokenized_prompts = self.prompt_learner(indices, test_class, test)
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            text_features = text_features.view(image_features.shape[0], n_test, -1)
            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)
            return logits

        else:
            n_class = self.n_class
            probability = image_features @ self.text_key.t()
            _, indices = probability.topk(k=min(self.args.text_prompt, probability.shape[1]), dim=1, largest=True)
            key_choose = self.text_key[indices]
            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts = self.prompt_learner(indices)
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(image_features.shape[0], n_class, -1)
            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)

            nc_text_features = self.text_encoder(nc_prompts, nc_tokenized_prompts)
            nc_text_features = nc_text_features / nc_text_features.norm(dim=-1, keepdim=True)
            dis = nc_text_features @ nc_text_features.permute(1, 0)
            loss_m = dis[~torch.eye(self.args.num_prompt, dtype=torch.bool, device=self.device)].abs().mean()

            return logits, image_features, key_choose, loss_m

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype


class CoOp:
    def __init__(self, device, prev_key, prev_prompt, args, n_ctx=12, use_float32=False, use_grad_checkpoint=False, keep=False):
        super().__init__()
        self.device = device
        self.args = args
        clip_model, _ = load('ViT-L/14', device=device)
        clip_model.eval()
        if use_float32:
            clip_model.float()

        if self.args.freeze_clip:
            for param in clip_model.parameters():
                param.requires_grad = False

        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint
        self.num_prompt = args.num_prompt
        self.n_ctx = n_ctx
        self.lr = args.lr * args.batch_size / 20
        self.wd = args.optim_wd
        self.epochs = args.n_epochs
        self.train_batch = args.batch_size
        self.args = args
        dtype = clip_model.dtype
        self.dtype = dtype
        # prompt learner
        ctx_dim = clip_model.ln_final.weight.shape[0]
        text_key = torch.empty(self.num_prompt, ctx_dim, dtype=self.dtype).to(self.device)
        nn.init.normal_(text_key, std=0.02)
        text_prompt = torch.empty(self.num_prompt, n_ctx, ctx_dim, dtype=self.dtype).to(self.device)
        nn.init.normal_(text_prompt, std=0.02)
        if keep == True:
            self.text_key = nn.Parameter(prev_key)
            self.text_prompt = nn.Parameter(prev_prompt)
        else:
            self.text_key = nn.Parameter(text_key)
            self.text_prompt = nn.Parameter(text_prompt)

    def init_model(self, class_names, text_key, text_prompt):

        self.n_class = len(class_names)
        clip_model = deepcopy(self.clip_model)

        self.model = CLIP(self.device, self.args, class_names, clip_model, text_key, text_prompt, self.n_ctx)
        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True
            except BaseException:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True

    def get_optimizer(self, per_epoch_steps):
        Other_params = [param for name, param in self.model.named_parameters() if 'text_key' in name]
        param_dict = [{'params': [p for p in self.model.prompt_learner.parameters() if p.requires_grad]},
                      {'params': Other_params}]

        optimizer = torch.optim.SGD(param_dict, lr=self.lr, weight_decay=self.wd)
        scheduler = build_cosine_scheduler(
            optimizer,
            lr=self.lr,
            total_step=self.epochs * per_epoch_steps)

        return optimizer, scheduler

    @property
    def training(self):
        return self.model.training

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)

    def parameters(self):
        return self.model.parameters()
