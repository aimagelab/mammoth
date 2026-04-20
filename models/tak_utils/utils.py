from typing import Literal, overload

import torch
import math
import os
import hashlib
import json
import logging
import re
import warnings
import urllib.request
from urllib.error import HTTPError
from urllib.parse import quote, urljoin

from utils import binary_to_boolean_type
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset

from tqdm import tqdm

try:
    import clip  # noqa: F401
except ImportError:
    raise ImportError(
        "Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git"
    )

import numpy as np
from utils.conf import get_checkpoint_path


def set_requires_grad_to(model, namevars, mode: bool):
    for n, p in model.named_parameters():
        if n in namevars:
            p.requires_grad = mode


def add_clip_args(parser):
    parser.add_argument(
        "--clip_backbone",
        type=str,
        default="ViT-B/16",
        help="Backbone architecture for CLIP",
        choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"],
    )
    parser.add_argument(
        "--ft_linears",
        type=binary_to_boolean_type,
        default=1,
        help="Set to 1 fine-tune linear layers",
    )
    parser.add_argument(
        "--ft_attention",
        type=binary_to_boolean_type,
        default=1,
        help="Set to 1 fine-tune attention layers",
    )
    parser.add_argument(
        "--ft_ln",
        type=binary_to_boolean_type,
        default=1,
        help="Set to 1 fine-tune layer norm",
    )
    parser.add_argument(
        "--ft_class_embed",
        type=binary_to_boolean_type,
        default=1,
        help="Set to 1 fine-tune class embedding layers",
    )
    parser.add_argument(
        "--ft_proj",
        type=binary_to_boolean_type,
        default=1,
        help="Set to 1 fine-tune projection layers",
    )

    parser.add_argument(
        "--ft_pos_embed",
        type=binary_to_boolean_type,
        default=0,
        help="Set to 1 fine-tune posistional embedding",
    )
    parser.add_argument(
        "--ft_conv",
        type=binary_to_boolean_type,
        default=0,
        help="Set to 1 fine-tune convolutional layers",
    )


class OptimizerBuilder:
    def __init__(self, cmd_args):
        self.args = cmd_args

    def build_opt_and_sched(self, all_params, num_batches):
        opt, sched = None, None

        if self.args.optimizer == "adamw":
            opt = torch.optim.AdamW(
                all_params, lr=self.args.lr, weight_decay=self.args.optim_wd
            )
        elif self.args.optimizer == "sgd":
            opt = torch.optim.SGD(
                all_params,
                lr=self.args.lr,
                momentum=self.args.optim_mom,
                weight_decay=self.args.optim_wd,
            )
        else:
            raise ValueError

        reduction_factor = getattr(self.args, "epochs_factor_reduction", 1)

        if self.args.scheduler_ntk == "none":
            pass
        elif self.args.scheduler_ntk == "cosine":
            num_total_steps = self.args.n_epochs * (
                num_batches // self.args.virtual_bs_n
            )
            sched = cosine_lr(
                opt, self.args.lr, 500 / reduction_factor, num_total_steps, 0
            )
        elif self.args.scheduler_ntk == "cosine_talos":
            num_total_steps = self.args.n_epochs * (
                num_batches // self.args.virtual_bs_n
            )
            sched = cosine_lr(opt, self.args.lr, 200, num_total_steps, 0)
        elif self.args.scheduler_ntk == "cosine_plus":
            num_total_steps = self.args.n_epochs * (
                num_batches // self.args.virtual_bs_n
            )
            warmup_steps = int(0.1 * num_total_steps)
            sched = cosine_lr(
                opt, self.args.lr, warmup_steps, num_total_steps, 0.1 * self.args.lr
            )
        elif self.args.scheduler_ntk == "decay":
            sched = cosine_lr(opt, self.args.lr, 0, self.args.n_epochs * num_batches, 0)
        elif self.args.scheduler_ntk == "step":
            num_steps = self.args.n_epochs * num_batches // self.args.virtual_bs_n
            warmup_steps = int(0.1 * num_steps)

            sched = step_lr_decay(opt, self.args.lr, warmup_steps, num_steps)
        else:
            raise ValueError

        return opt, sched

    def build_opt_and_sched_multiple_lr(
        self, params_group_1, params_group_2, num_batches
    ):
        opt, sched = None, None

        lr_group_1 = self.args.lr
        lr_group_2 = getattr(self.args, "lr2", None)
        if lr_group_2 is None:
            lr_group_2 = getattr(self.args, "lr_lin", None)
        if lr_group_2 is None:
            lr_group_2 = getattr(self.args, "lr_second", None)
        if lr_group_2 is None or lr_group_2 == 0:
            lr_group_2 = self.args.lr

        param_groups = [
            {"params": params_group_1, "lr": lr_group_1},
            {"params": params_group_2, "lr": lr_group_2},
        ]

        if self.args.optimizer == "adamw":
            opt = torch.optim.AdamW(
                param_groups, lr=lr_group_1, weight_decay=self.args.optim_wd
            )
        elif self.args.optimizer == "sgd":
            opt = torch.optim.SGD(
                param_groups,
                lr=lr_group_1,
                momentum=self.args.optim_mom,
                weight_decay=self.args.optim_wd,
            )
        else:
            raise ValueError

        base_lrs = [lr_group_1, lr_group_2]

        if self.args.scheduler_ntk == "none":
            pass
        elif self.args.scheduler_ntk == "cosine":
            num_total_steps = self.args.n_epochs * (
                num_batches // self.args.virtual_bs_n
            )
            sched = cosine_lr(opt, base_lrs, 500, num_total_steps, 0)
        elif self.args.scheduler_ntk == "cosine_talos":
            num_total_steps = self.args.n_epochs * (
                num_batches // self.args.virtual_bs_n
            )
            sched = cosine_lr(opt, base_lrs, 200, num_total_steps, 0)
        elif self.args.scheduler_ntk == "cosine_plus":
            num_total_steps = self.args.n_epochs * (
                num_batches // self.args.virtual_bs_n
            )
            warmup_steps = int(0.1 * num_total_steps)
            min_lrs = [0.1 * lr for lr in base_lrs]
            sched = cosine_lr(opt, base_lrs, warmup_steps, num_total_steps, min_lrs)
        elif self.args.scheduler_ntk == "decay":
            sched = cosine_lr(opt, base_lrs, 0, self.args.n_epochs * num_batches, 0)
        elif self.args.scheduler_ntk == "step":
            num_steps = self.args.n_epochs * num_batches // self.args.virtual_bs_n
            warmup_steps = int(0.1 * num_steps)

            sched = step_lr_decay(opt, base_lrs, warmup_steps, num_steps)
        else:
            raise ValueError

        return opt, sched


@torch.no_grad()
def compute_acc_on_last_task(model: ContinualModel, dataset: ContinualDataset):
    test_loader = dataset.test_loaders[-1]
    total_len = len(test_loader) if hasattr(test_loader, "__len__") else None

    pbar = tqdm(
        test_loader, total=total_len, desc="Evaluating", disable=model.args.non_verbose
    )

    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    test_iter = iter(test_loader)
    i = 0

    num_classes = dataset.N_CLASSES
    while True:
        try:
            data = next(test_iter)
        except StopIteration:
            break
        if model.args.debug_mode and i > model.get_debug_iters():
            break
        inputs, labels = data[0], data[1]
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model.forward(inputs)

        assert outputs.shape[1] == num_classes

        _, pred = torch.max(outputs, 1)

        correct += torch.sum(pred == labels).item()
        total += labels.shape[0]
        i += 1
        pbar.set_postfix(
            {f"acc_task_{model.current_task + 1}": max(0, correct / total * 100)},
            refresh=False,
        )
        pbar.set_description(f"Evaluating Task {model.current_task + 1}", refresh=False)
        pbar.update(1)

        start_c, end_c = dataset.get_offsets(model.current_task)
        outputs[:, :start_c] = -float("inf")
        outputs[:, end_c:num_classes] = -float("inf")
        _, pred = torch.max(outputs.data, 1)
        correct_mask_classes += torch.sum(pred == labels).item()

    acc = correct / total * 100
    acc_mask_classes = correct_mask_classes / total * 100

    pbar.close()

    return acc, acc_mask_classes


def make_psd(x, to64=False):
    orig_dtype = x.dtype
    if to64:
        x = x.to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(x)
    eigvals_clamped = torch.clamp(eigvals, min=0.0)
    x_psd = (eigvecs * eigvals_clamped) @ eigvecs.t()
    return x_psd.to(orig_dtype) if to64 else x_psd


class FisherLoader:
    def __init__(
        self,
        fisher_cache,
        dataset_name,
        device,
        fp_precision="fp32",
        fallback_dataset_name=None,
    ):
        self.dataset_name = dataset_name
        self.fallback_dataset_name = fallback_dataset_name
        self.device = device
        self.fisher_cache = fisher_cache
        self.fp_precision = fp_precision
        self.postprocessing = None
        self._dataset_name_hints: list[str] | None = None

    @staticmethod
    def _append_unique(items: list[str], value: str | None) -> None:
        if value and value not in items:
            items.append(value)

    def _extract_path_dataset_hints(self) -> list[str]:
        hints: list[str] = []
        if not isinstance(self.fisher_cache, str):
            return hints

        parts = self.fisher_cache.replace("@", "/").split("/")
        for part in parts:
            if not part.startswith("fisher_"):
                continue
            dataset_hint = part[len("fisher_") :].strip()
            if dataset_hint in {"", "cache"}:
                continue
            self._append_unique(hints, dataset_hint)
            if not dataset_hint.startswith("seq-"):
                self._append_unique(hints, f"seq-{dataset_hint}")

        return hints

    def _list_cache_entries(self) -> list[str]:
        if self._is_hf_source():
            repo_id, base_path, revision = self._parse_hf_source(self.fisher_cache)
            encoded_repo = quote(repo_id, safe="")
            encoded_revision = quote(revision, safe="")
            encoded_base_path = quote(base_path.strip("/"), safe="/")
            endpoint = (
                f"https://huggingface.co/api/models/{encoded_repo}/tree/{encoded_revision}"
            )
            if encoded_base_path:
                endpoint = f"{endpoint}/{encoded_base_path}"
            try:
                with urllib.request.urlopen(endpoint) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    return []
                return [
                    item["path"]
                    for item in payload
                    if isinstance(item, dict)
                    and item.get("type") == "file"
                    and isinstance(item.get("path"), str)
                ]
            except Exception:
                return []

        if self._is_http_source():
            return []

        try:
            return [
                os.path.join(self.fisher_cache, name)
                for name in os.listdir(self.fisher_cache)
            ]
        except Exception:
            return []

    def _extract_dataset_hints_from_entries(self) -> list[str]:
        hints: list[str] = []
        pattern = re.compile(
            r"^(?P<dataset>.+)_task_\d+_(?:num_(?:aaT|ggT)|aaT|ggT|ffT)\.pt$"
        )
        for entry in self._list_cache_entries():
            filename = os.path.basename(entry)
            match = pattern.match(filename)
            if not match:
                continue
            dataset_hint = match.group("dataset")
            self._append_unique(hints, dataset_hint)
        return hints

    def _get_dataset_name_hints(self) -> list[str]:
        if self._dataset_name_hints is not None:
            return self._dataset_name_hints

        hints: list[str] = []
        for hint in self._extract_path_dataset_hints():
            self._append_unique(hints, hint)
        for hint in self._extract_dataset_hints_from_entries():
            self._append_unique(hints, hint)

        self._dataset_name_hints = hints
        return hints

    def _dataset_name_candidates(self) -> list[str]:
        names = [self.dataset_name]
        if (
            self.fallback_dataset_name is not None
            and self.fallback_dataset_name not in names
        ):
            names.append(self.fallback_dataset_name)
        for hint in self._get_dataset_name_hints():
            self._append_unique(names, hint)
        return names

    def _try_resolve_file(self, filename: str) -> str | None:
        try:
            file_path = self._resolve_file(filename)
        except FileNotFoundError:
            return None
        except HTTPError as e:
            if e.code == 404:
                return None
            raise
        except Exception as e:
            if e.__class__.__name__ in {"EntryNotFoundError", "LocalEntryNotFoundError"}:
                return None
            raise

        if os.path.exists(file_path):
            return file_path
        return None

    def _resolve_count_paths(self, task_id: int) -> tuple[str, str, str]:
        for dataset_name in self._dataset_name_candidates():
            base_name = f"{dataset_name}_task_{task_id}"
            aaT_count_path = self._try_resolve_file(f"{base_name}_num_aaT.pt")
            ggT_count_path = self._try_resolve_file(f"{base_name}_num_ggT.pt")

            if aaT_count_path is None and ggT_count_path is None:
                continue

            if aaT_count_path is None or ggT_count_path is None:
                raise FileNotFoundError(
                    f"Incomplete Fisher counts for `{base_name}` in `{self.fisher_cache}`. "
                    "Expected both `_num_aaT.pt` and `_num_ggT.pt`."
                )

            return base_name, aaT_count_path, ggT_count_path

        expected = [f"{name}_task_{task_id}" for name in self._dataset_name_candidates()]
        raise FileNotFoundError(
            f"Fisher cache for task {task_id} not found in `{self.fisher_cache}`. "
            f"Tried dataset prefixes: {expected}."
        )

    def has_task(self, task_id: int) -> bool:
        try:
            self._resolve_count_paths(task_id)
            return True
        except FileNotFoundError:
            return False

    def get_available_task_ids(self, max_tasks: int) -> list[int]:
        task_ids = []
        for task_id in range(max_tasks):
            if self.has_task(task_id):
                task_ids.append(task_id)
        return task_ids

    def _is_hf_source(self) -> bool:
        return isinstance(self.fisher_cache, str) and self.fisher_cache.startswith(
            "hf://"
        )

    def _is_http_source(self) -> bool:
        return isinstance(self.fisher_cache, str) and self.fisher_cache.startswith(
            ("http://", "https://")
        )

    def _is_remote_source(self) -> bool:
        return self._is_hf_source() or self._is_http_source()

    def _remote_cache_dir(self) -> str:
        src_hash = hashlib.sha256(self.fisher_cache.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join(
            get_checkpoint_path(), "fisher_remote_cache", self.dataset_name, src_hash
        )
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def _assert_not_lfs_pointer(file_path: str) -> None:
        with open(file_path, "rb") as f:
            header = f.read(256)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"Downloaded file `{file_path}` is a Git LFS pointer, not the binary artifact. "
                "Use a direct/raw artifact URL (or Hugging Face resolve URL) instead of a pointer URL."
            )

    @staticmethod
    def _parse_hf_source(spec: str) -> tuple[str, str, str]:
        assert spec.startswith("hf://")
        payload = spec[len("hf://") :]
        if "@" in payload:
            payload, revision = payload.rsplit("@", 1)
        else:
            revision = "main"

        parts = payload.split("/")
        if len(parts) < 2:
            raise ValueError(
                "Invalid HF source format. Use `hf://<owner>/<repo>/<optional/subpath>@<optional_revision>`"
            )

        repo_id = "/".join(parts[:2])
        base_path = "/".join(parts[2:])
        return repo_id, base_path, revision

    def _download_http_file(self, remote_url: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with (
            urllib.request.urlopen(remote_url) as source,
            open(local_path, "wb") as output,
        ):
            output.write(source.read())
        self._assert_not_lfs_pointer(local_path)
        return local_path

    def _resolve_file(self, filename: str) -> str:
        if not self._is_remote_source():
            return os.path.join(self.fisher_cache, filename)

        cache_dir = self._remote_cache_dir()
        local_path = os.path.join(cache_dir, filename)
        if os.path.exists(local_path):
            return local_path

        if self._is_hf_source():
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(
                    "huggingface_hub is required for `hf://` Fisher cache sources. "
                    "Install it with `pip install huggingface_hub`."
                ) from e

            repo_id, base_path, revision = self._parse_hf_source(self.fisher_cache)
            hf_filename = f"{base_path}/{filename}" if base_path else filename
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                revision=revision,
                repo_type="model",
                cache_dir=cache_dir,
            )
            self._assert_not_lfs_pointer(downloaded_path)
            logging.info(
                f"Downloaded Fisher file from HF: {repo_id}/{hf_filename}@{revision}"
            )
            return downloaded_path

        remote_url = urljoin(self.fisher_cache.rstrip("/") + "/", filename)
        logging.info(f"Downloading Fisher file from URL: {remote_url}")
        return self._download_http_file(remote_url, local_path)

    @overload
    def load_kfac(
        self, task_id: int, only_counts: Literal[True]
    ) -> tuple[int, int]: ...

    @overload
    def load_kfac(
        self, task_id: int, only_counts: Literal[False] = False
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        int,
        int,
    ]: ...

    def load_kfac(
        self, task_id: int, only_counts: bool = False
    ) -> (
        tuple[
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
            int,
            int,
        ]
        | tuple[int, int]
    ):
        base_name, fisher_cache_path_num_aaT, fisher_cache_path_num_ggT = (
            self._resolve_count_paths(task_id)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cur_num_aaT: int = torch.load(
                fisher_cache_path_num_aaT, map_location="cpu"
            ).item()
            cur_num_ggT: int = torch.load(
                fisher_cache_path_num_ggT, map_location="cpu"
            ).item()

        if only_counts:
            logging.info(
                f"Loaded Fisher counts for task {task_id} from `{self.fisher_cache}`"
            )
            return cur_num_ggT, cur_num_aaT

        fisher_cache_path_aaT = self._try_resolve_file(f"{base_name}_aaT.pt")
        fisher_cache_path_ggT = self._try_resolve_file(f"{base_name}_ggT.pt")
        fisher_cache_path_ffT = self._try_resolve_file(f"{base_name}_ffT.pt")

        if (
            fisher_cache_path_aaT is None
            or fisher_cache_path_ggT is None
            or fisher_cache_path_ffT is None
        ):
            raise FileNotFoundError(
                f"Incomplete Fisher tensors for `{base_name}` in `{self.fisher_cache}`. "
                "Expected `_aaT.pt`, `_ggT.pt`, and `_ffT.pt`."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aaT: dict = torch.load(fisher_cache_path_aaT, map_location=self.device)
            ggT: dict = torch.load(fisher_cache_path_ggT, map_location=self.device)
            ffT: dict = torch.load(fisher_cache_path_ffT, map_location=self.device)

        for key in aaT.keys():
            if self.fp_precision == "fp64":
                aaT[key] = aaT[key].to(torch.float64)
                ggT[key] = ggT[key].to(torch.float64)
            elif self.fp_precision == "fp32":
                aaT[key] = aaT[key].to(torch.float32)
                ggT[key] = ggT[key].to(torch.float32)
            else:
                raise NotImplementedError

        for key in ffT.keys():
            if self.fp_precision == "fp64":
                # ffT[key] = ffT[key].to(torch.float64)
                ffT[key] = ffT[key].to(torch.float64)
            elif self.fp_precision == "fp32":
                # ffT[key] = ffT[key].to(torch.float32)
                ffT[key] = ffT[key].to(torch.float32)
            else:
                raise NotImplementedError

        logging.info(
            f"Loaded Fisher tensors for task {task_id} from `{self.fisher_cache}`"
        )
        return ggT, aaT, ffT, cur_num_ggT, cur_num_aaT

    def store_kfac(self, task_id, ggT, aaT, ffT, num_ggT, num_aaT):
        if self._is_remote_source():
            raise ValueError(
                "Cannot store Fisher cache to remote sources. "
                "Set `--fisher_cache` to a local directory when `--load_fisher=0`."
            )
        os.makedirs(self.fisher_cache, exist_ok=True)
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        torch.save(ggT, fisher_cache_path.replace(".pt", "_ggT.pt"))
        torch.save(aaT, fisher_cache_path.replace(".pt", "_aaT.pt"))
        torch.save(ffT, fisher_cache_path.replace(".pt", "_ffT.pt"))
        torch.save(
            torch.tensor([num_ggT]), fisher_cache_path.replace(".pt", "_num_ggT.pt")
        )
        torch.save(
            torch.tensor([num_aaT]), fisher_cache_path.replace(".pt", "_num_aaT.pt")
        )

    def load_ekfac(
        self, task_id, only_counts=False
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        int,
        int,
    ]:
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        fisher_cache_path_num_of_examples = fisher_cache_path.replace(
            ".pt", "_num_of_examples.pt"
        )

        assert os.path.exists(fisher_cache_path_num_of_examples), (
            f"File {fisher_cache_path_num_of_examples} not found."
        )

        num_of_examples = torch.load(
            fisher_cache_path_num_of_examples, map_location="cpu"
        ).item()

        if only_counts:
            return num_of_examples

        fisher_cache_path_UA = fisher_cache_path.replace(".pt", "_UA.pt")
        fisher_cache_path_UG = fisher_cache_path.replace(".pt", "_UG.pt")
        fisher_cache_path_D = fisher_cache_path.replace(".pt", "_D.pt")
        fisher_cache_path_ffT = fisher_cache_path.replace(".pt", "_ffT.pt")

        assert os.path.exists(fisher_cache_path_UA)
        assert os.path.exists(fisher_cache_path_UG)
        assert os.path.exists(fisher_cache_path_D)
        assert os.path.exists(fisher_cache_path_ffT)

        UA = torch.load(fisher_cache_path_UA, map_location=self.device)
        UG = torch.load(fisher_cache_path_UG, map_location=self.device)
        D = torch.load(fisher_cache_path_D, map_location=self.device)
        ffT = torch.load(fisher_cache_path_ffT, map_location=self.device)

        assert UA.keys() == UG.keys() == D.keys()

        for key in UA.keys():
            if self.fp_precision == "fp64":
                UA[key] = UA[key].to(torch.float64)
                UG[key] = UG[key].to(torch.float64)
                D[key] = D[key].to(torch.float64)
            elif self.fp_precision == "fp32":
                UA[key] = UA[key].to(torch.float32)
                UG[key] = UG[key].to(torch.float32)
                D[key] = D[key].to(torch.float32)
            else:
                raise NotImplementedError

        for key in ffT.keys():
            if self.fp_precision == "fp64":
                ffT[key] = ffT[key].to(torch.float64)
            elif self.fp_precision == "fp32":
                ffT[key] = ffT[key].to(torch.float32)
            else:
                raise NotImplementedError

        return UA, UG, D, ffT, num_of_examples

    def load_diff_ekfac(
        self, task_id, only_counts=False
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        int,
        int,
    ]:
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        fisher_cache_path_num_of_examples = fisher_cache_path.replace(
            ".pt", "_num_of_examples.pt"
        )

        assert os.path.exists(fisher_cache_path_num_of_examples)

        num_of_examples = torch.load(
            fisher_cache_path_num_of_examples, map_location="cpu"
        ).item()

        if only_counts:
            return num_of_examples

        fisher_cache_universe_path = (
            f"{self.fisher_cache}/{self.dataset_name}_universe.pt"
        )
        fisher_cache_path_UA = fisher_cache_universe_path.replace(".pt", "_UA.pt")
        fisher_cache_path_UG = fisher_cache_universe_path.replace(".pt", "_UG.pt")

        fisher_cache_path_D = fisher_cache_path.replace(".pt", "_D.pt")
        fisher_cache_path_ffT = fisher_cache_path.replace(".pt", "_ffT.pt")

        assert os.path.exists(fisher_cache_path_UA)
        assert os.path.exists(fisher_cache_path_UG)
        assert os.path.exists(fisher_cache_path_D)
        assert os.path.exists(fisher_cache_path_ffT)
        if task_id == 0:
            UA = torch.load(fisher_cache_path_UA, map_location=self.device)
            UG = torch.load(fisher_cache_path_UG, map_location=self.device)
        else:
            UA = {}
            UG = {}
        D = torch.load(fisher_cache_path_D, map_location=self.device)
        ffT = torch.load(fisher_cache_path_ffT, map_location=self.device)

        if task_id == 0:
            assert UA.keys() == UG.keys() == D.keys()

        for key in UA.keys():
            if self.fp_precision == "fp64":
                UA[key] = UA[key].to(torch.float64)
                UG[key] = UG[key].to(torch.float64)
                D[key] = D[key].to(torch.float64)
            elif self.fp_precision == "fp32":
                UA[key] = UA[key].to(torch.float32)
                UG[key] = UG[key].to(torch.float32)
                D[key] = D[key].to(torch.float32)
            else:
                raise NotImplementedError

        for key in ffT.keys():
            if self.fp_precision == "fp64":
                ffT[key] = ffT[key].to(torch.float64)
            elif self.fp_precision == "fp32":
                ffT[key] = ffT[key].to(torch.float32)
            else:
                raise NotImplementedError

        return UA, UG, D, ffT, num_of_examples


def get_parameter(
    shape,
    device,
    type_init: str = "orto",
    transpose: bool = False,
    requires_grad: bool = True,
):
    param = torch.zeros(*shape, dtype=torch.float32, device=device)
    if type_init == "orto":
        torch.nn.init.orthogonal_(param)
    if type_init == "gaussian":
        torch.nn.init.normal_(param, mean=0.0, std=0.1)
    if type_init == "kernel":
        torch.nn.init.normal_(param, mean=0.0, std=0.036)
    if type_init == "attn":
        torch.nn.init.normal_(param, mean=1.0, std=0.03)
    if type_init == "kaiming":
        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    if type_init == "ones":
        torch.nn.init.ones_(param)
    if transpose:
        param = torch.transpose(param, 1, 2)
    return torch.nn.Parameter(param, requires_grad=requires_grad)


def get_params(
    net, features=True, classifier=False, offset_1=-1, offset_2=-1
) -> torch.Tensor:
    params = []
    for name, param in net.named_parameters():
        if "head" in name:
            if classifier:
                assert offset_1 > -1 and offset_2 > -1
                params.append(param[offset_1:offset_2].view(-1))
        elif features:
            params.append(param.view(-1))

    if len(params):
        return torch.cat(params)
    else:
        return torch.tensor([0.0])


def set_params(
    net,
    new_params: torch.Tensor,
    features=True,
    classifier=False,
    offset_1=-1,
    offset_2=-1,
) -> None:
    progress = 0
    for name, param in net.named_parameters():
        if "head" in name:
            if classifier:
                assert offset_1 > -1 and offset_2 > -1
                cur_size = torch.tensor(param.data[offset_1:offset_2].size()).prod()
                param.data[offset_1:offset_2] = new_params[
                    progress : progress + cur_size
                ].view(param.data[offset_1:offset_2].size())
                progress += cur_size
        elif features:
            cur_size = torch.tensor(param.size()).prod()
            cand_params = new_params[progress : progress + cur_size].view(param.size())
            param.data = cand_params
            progress += cur_size


def get_delta_w_backbone(named_params, delta_w, delta_w_names, training_type, device):
    params = []
    for name, param in named_params():
        name = name.replace("visual_encoder.", "")
        if "head" not in name:
            if name in delta_w_names:
                index = delta_w_names.index(name)
                cur_delta_w = delta_w[index]
                params.append(cur_delta_w.view(-1).to(device))
            elif name == "logit_scale":
                # else:
                # params.append(torch.zeros_like(param).view(-1).to(device))
                print(name)
                print("ops siamo finiti in sto posto strano ma non facciamo nulla")
                # params.append(torch.clone(param).view(-1).to(device))

    if len(params):
        return torch.cat(params)
    else:
        return torch.tensor([0.0]).to(device)


def get_delta_w_parameterlist(named_params, delta_w, delta_w_names, peft_type, device):
    params = []
    for name, param in named_params():
        if name in delta_w_names:
            index = delta_w_names.index(name)
            cur_delta_w = None
            if peft_type == "lora":
                cur_delta_w = delta_w[index][0] @ delta_w[index][1]
            elif peft_type == "full":
                cur_delta_w = delta_w[index]
            assert cur_delta_w
            params.append(cur_delta_w.to(device))
        else:
            params.append(torch.zeros_like(param).to(device))

    return params


def replace_non_dynamically_quantizable_linear(module):
    """Recursively replace all NonDynamicallyQuantizableLinear layers with Linear layers in a model."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
            # Replace with an equivalent Linear layer
            new_layer = torch.nn.Linear(
                child.in_features, child.out_features, bias=child.bias is not None
            )
            new_layer.weight = torch.nn.Parameter(child.weight.clone())  # Copy weights
            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(child.bias.clone())  # Copy bias
            setattr(module, name, new_layer)
        else:
            replace_non_dynamically_quantizable_linear(
                child
            )  # Recursively process children

    return module


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    if not isinstance(min_lr, list):
        min_lr_list = [min_lr for _ in optimizer.param_groups]
    else:
        min_lr_list = min_lr
    assert len(base_lrs) == len(optimizer.param_groups) == len(min_lr_list)

    def _lr_adjuster(step):
        for param_group, base_lr, group_min_lr in zip(
            optimizer.param_groups, base_lrs, min_lr_list
        ):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = group_min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def step_lr_decay(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                progress = step / steps
                if progress < 0.70:
                    lr = base_lr
                elif progress < 0.90:
                    lr = base_lr * 0.5
                else:
                    lr = base_lr * 0.1
            assign_learning_rate(param_group, lr)

    return _lr_adjuster
