import torch
import torch.nn.functional as F
from copy import deepcopy
import types

from backbone import get_backbone
from models.twf_utils.afd import MultiTaskAFDAlternative


@torch.no_grad()
def init_twf(model, dataset):
    model.teacher = get_backbone(model.args)
    if isinstance(model.net, torch.nn.DataParallel):
        st = deepcopy(model.net.module.state_dict())
    else:
        st = deepcopy(model.net.state_dict())

    for k in list(st.keys()):
        if 'classifier' in k:
            st.pop(k)
    unknown, missing = model.teacher.load_state_dict(st, strict=False)
    assert len(missing) == 0
    assert len([x for x in unknown if 'classifier' not in x]) == 0
    model.teacher.to(model.device)

    model.net.set_return_prerelu(True)
    model.teacher.set_return_prerelu(True)

    # Set new forward for teacher
    @torch.no_grad()
    def _teacher_forward(self, x):
        ret = []
        x = x.to(self.device)
        x = self.bn1(self.conv1(x))
        ret.append(x.clone().detach())
        x = F.relu(x)

        if hasattr(self, 'maxpool'):
            x = self.maxpool(x)
        x = self.layer1(x)
        ret.append(self.layer1[-1].prerelu.clone().detach())

        x = self.layer2(x)
        ret.append(self.layer2[-1].prerelu.clone().detach())

        x = self.layer3(x)
        ret.append(self.layer3[-1].prerelu.clone().detach())

        x = self.layer4(x)
        ret.append(self.layer4[-1].prerelu.clone().detach())

        return ret

    if isinstance(model.teacher, torch.nn.DataParallel):
        model.teacher.module.forward = types.MethodType(
            _teacher_forward, model.teacher.module)
    else:
        model.teacher.forward = types.MethodType(
            _teacher_forward, model.teacher)

    # Initialize classifier
    model.net.classifier = torch.nn.Linear(
        model.net.classifier.in_features, model.num_classes).to(model.device)

    # --- Create adapters ---
    # Retrieve features to get shapes
    x = next(iter(dataset.train_loader))[0].to(model.device)
    _, feats_t = model.net(x, returnt='full')
    teacher_input = x
    pret_feats_t = model.teacher(teacher_input)

    # Initialize adapters
    for i, (x, pret_x) in enumerate(zip(feats_t, pret_feats_t)):
        # clear_grad=self.args.detach_skip_grad
        adapt_shape = x.shape[1:]
        pret_shape = pret_x.shape[1:]
        if len(adapt_shape) == 1:
            adapt_shape = (adapt_shape[0], 1, 1)  # linear is a cx1x1
            pret_shape = (pret_shape[0], 1, 1)

        setattr(model.net, f"adapter_{i+1}", MultiTaskAFDAlternative(
            adapt_shape, model.N_TASKS, model.cpt, clear_grad=False,
            teacher_forcing_or=False,
            lambda_forcing_loss=model.args.lambda_fp_replay,
            use_overhaul_fd=True, use_hard_softmax=True,
            lambda_diverse_loss=model.args.lambda_diverse_loss,
            attn_mode="chsp",
            min_resize_threshold=model.args.min_resize_threshold,
            resize_maps=model.args.resize_maps,
        ).to(model.device))

    # Freeze teacher
    for p in model.teacher.parameters():
        p.requires_grad = False
