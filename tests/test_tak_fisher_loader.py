import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from models.tak_utils.utils import FisherLoader


def _write_fake_fisher_files(root: str, dataset_name: str, task_id: int = 0) -> None:
    base = os.path.join(root, f"{dataset_name}_task_{task_id}")
    ggT = {"w": torch.eye(2)}
    aaT = {"w": torch.eye(2) * 2}
    ffT = {"w": torch.eye(2) * 3}
    torch.save(ggT, base + "_ggT.pt")
    torch.save(aaT, base + "_aaT.pt")
    torch.save(ffT, base + "_ffT.pt")
    torch.save(torch.tensor([7]), base + "_num_ggT.pt")
    torch.save(torch.tensor([7]), base + "_num_aaT.pt")


@pytest.fixture
def fisher_cache_dir(tmp_path):
    dataset_name = "dummyset"
    _write_fake_fisher_files(str(tmp_path), dataset_name, task_id=0)
    return str(tmp_path), dataset_name


def test_fisher_loader_local(fisher_cache_dir):
    cache_dir, dataset_name = fisher_cache_dir
    loader = FisherLoader(
        cache_dir, dataset_name, torch.device("cpu"), fp_precision="fp32"
    )

    num_ggT, num_aaT = loader.load_kfac(0, only_counts=True)
    assert num_ggT == 7
    assert num_aaT == 7

    ggT, aaT, ffT, num_ggT, num_aaT = loader.load_kfac(0)
    assert num_ggT == 7
    assert num_aaT == 7
    assert "w" in ggT and "w" in aaT and "w" in ffT
    assert ggT["w"].dtype == torch.float32
    assert aaT["w"].dtype == torch.float32
    assert ffT["w"].dtype == torch.float32


def test_fisher_loader_http(fisher_cache_dir):
    cache_dir, dataset_name = fisher_cache_dir

    handler = partial(SimpleHTTPRequestHandler, directory=cache_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        loader = FisherLoader(
            base_url, dataset_name, torch.device("cpu"), fp_precision="fp32"
        )

        num_ggT, num_aaT = loader.load_kfac(0, only_counts=True)
        assert num_ggT == 7
        assert num_aaT == 7

        ggT, aaT, ffT, num_ggT, num_aaT = loader.load_kfac(0)
        assert num_ggT == 7
        assert num_aaT == 7
        assert "w" in ggT and "w" in aaT and "w" in ffT
    finally:
        server.shutdown()
        server.server_close()


def test_fisher_loader_fallback_dataset_name(tmp_path):
    fallback_dataset_name = "seq-8vision"
    runtime_dataset_name = "seq-cifar100-224"
    _write_fake_fisher_files(str(tmp_path), fallback_dataset_name, task_id=0)

    loader = FisherLoader(
        str(tmp_path),
        runtime_dataset_name,
        torch.device("cpu"),
        fp_precision="fp32",
        fallback_dataset_name=fallback_dataset_name,
    )

    num_ggT, num_aaT = loader.load_kfac(0, only_counts=True)
    assert num_ggT == 7
    assert num_aaT == 7

    ggT, aaT, ffT, num_ggT, num_aaT = loader.load_kfac(0)
    assert num_ggT == 7
    assert num_aaT == 7
    assert "w" in ggT and "w" in aaT and "w" in ffT


def test_fisher_loader_available_tasks_single_checkpoint(tmp_path):
    dataset_name = "seq-8vision"
    _write_fake_fisher_files(str(tmp_path), dataset_name, task_id=0)

    loader = FisherLoader(
        str(tmp_path), dataset_name, torch.device("cpu"), fp_precision="fp32"
    )

    assert loader.has_task(0)
    assert not loader.has_task(1)
    assert loader.get_available_task_ids(4) == [0]
