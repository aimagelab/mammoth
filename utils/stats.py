try:
    from resource import getrusage, RUSAGE_CHILDREN, RUSAGE_SELF

    def get_memory_mb():
        """
        Get the memory usage of the current process and its children.

        Returns:
            dict: A dictionary containing the memory usage of the current process and its children.

            The dictionary has the following keys:
                - self: The memory usage of the current process.
                - children: The memory usage of the children of the current process.
                - total: The total memory usage of the current process and its children.
        """
        res = {
            "self": getrusage(RUSAGE_SELF).ru_maxrss / 1024,
            "children": getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024,
            "total": getrusage(RUSAGE_SELF).ru_maxrss / 1024 + getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024
        }
        return res
except BaseException:
    get_memory_mb = None

import torch

try:
    if torch.cuda.is_available():
        from utils.conf import get_alloc_memory_all_devices

        def get_memory_gpu_mb(avail_devices=None):
            """
            Get the memory usage of the selected GPUs in MB.
            """

            return [d / 1024 / 1024 for d in get_alloc_memory_all_devices(avail_devices=avail_devices)]
    else:
        get_memory_gpu_mb = None
except BaseException:
    get_memory_gpu_mb = None

import logging
from utils.loggers import Logger


def _parse_device_ids(device):
    """
    Normalize a device specification to a list of CUDA ids.
    """
    if device is None:
        return None

    if isinstance(device, torch.device):
        if device.type != 'cuda':
            return None
        if device.index is None:
            return list(range(torch.cuda.device_count()))
        if 0 <= device.index < torch.cuda.device_count():
            return [device.index]
        logging.warning(f"Requested device index {device.index} is out of range.")
        return None

    if isinstance(device, str):
        if 'cuda' not in device:
            return None
        parts = [p for p in device.split(',') if p.strip() != '']
        if len(parts) == 0:
            return list(range(torch.cuda.device_count()))
        ids = []
        for p in parts:
            try:
                ids.append(int(p.split(':')[-1]))
            except ValueError:
                logging.warning(f"Could not parse device id from `{p}`, skipping.")
        ids = [i for i in ids if 0 <= i < torch.cuda.device_count()]
        if len(ids) == 0:
            logging.warning("No valid CUDA device ids parsed, falling back to all visible devices.")
            return list(range(torch.cuda.device_count()))
        return ids

    if isinstance(device, (list, tuple)):
        ids = []
        for d in device:
            if isinstance(d, int):
                ids.append(d)
            elif isinstance(d, torch.device) and d.type == 'cuda' and d.index is not None:
                ids.append(d.index)
        ids = [i for i in ids if 0 <= i < torch.cuda.device_count()]
        return ids or None

    return None


class track_system_stats:
    """
    A context manager that tracks the memory usage of the system.
    Tracks both CPU and GPU memory usage if available.

    Usage:

    .. code-block:: python

        with track_system_stats() as t:
            for i in range(100):
                ... # Do something
                t()

            cpu_res, gpu_res = t.cpu_res, t.gpu_res

        Args:
            logger (Logger): external logger.
            device: Device (or list of devices) to monitor. Defaults to all visible CUDA devices.
            disabled (bool): If True, the context manager will not track the memory usage.
    """

    def get_stats(self):
        """
        Get the memory usage of the system.

        Returns:
            tuple: (cpu_res, gpu_res) where cpu_res is the memory usage of the CPU and gpu_res is the memory usage of the GPU.
        """
        cpu_res = None
        if get_memory_mb is not None:
            cpu_res = get_memory_mb()['total']

        gpu_res = None
        if get_memory_gpu_mb is not None:
            gpu_res = get_memory_gpu_mb(self.gpu_ids)
            gpu_res = self._zip_gpu_res(gpu_res)

        return cpu_res, gpu_res

    def __init__(self, logger: Logger = None, device=None, disabled=False):
        self.logger = logger
        self.disabled = disabled
        self._it = 0
        self.gpu_ids = _parse_device_ids(device) if torch.cuda.is_available() else None

    def __enter__(self):
        if self.disabled:
            return self
        self.initial_cpu_res, self.initial_gpu_res = self.get_stats()
        if self.initial_cpu_res is None and self.initial_gpu_res is None:
            self.disabled = True
        else:
            self.avg_gpu_res = self.initial_gpu_res
            self.avg_cpu_res = self.initial_cpu_res

            self.max_cpu_res = self.initial_cpu_res
            self.max_gpu_res = self.initial_gpu_res

            if self.logger is not None:
                self.logger.log_system_stats(self.initial_cpu_res, self.initial_gpu_res)

        return self

    def __call__(self):
        if self.disabled:
            return

        cpu_res, gpu_res = self.get_stats()
        self.update_stats(cpu_res, gpu_res)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disabled:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # this allows to raise errors triggered previously by the GPU

        cpu_res, gpu_res = self.get_stats()
        self.update_stats(cpu_res, gpu_res)

    def update_stats(self, cpu_res, gpu_res):
        """
        Update the memory usage statistics.

        Args:
            cpu_res (float): The memory usage of the CPU.
            gpu_res (dict): The memory usage of the GPUs keyed by device id.
        """
        if self.disabled:
            return

        self._it += 1

        alpha = 1 / self._it
        if self.initial_cpu_res is not None:
            self.avg_cpu_res = self.avg_cpu_res + alpha * (cpu_res - self.avg_cpu_res)
            self.max_cpu_res = max(self.max_cpu_res, cpu_res)

        if self.initial_gpu_res is not None:
            self.avg_gpu_res = {g: (g_res + alpha * (g_res - self.avg_gpu_res[g])) for g, g_res in gpu_res.items()}
            self.max_gpu_res = {g: max(self.max_gpu_res[g], g_res) for g, g_res in gpu_res.items()}

        if self.logger is not None:
            self.logger.log_system_stats(cpu_res, gpu_res)

    def print_stats(self):
        """
        Print the memory usage statistics.
        """

        cpu_res, gpu_res = self.get_stats()

        # Print initial, average, final, and max memory usage
        logging.info("System stats:")
        if cpu_res is not None:
            logging.info(f"\tInitial CPU memory usage: {self.initial_cpu_res:.2f} MB")
            logging.info(f"\tAverage CPU memory usage: {self.avg_cpu_res:.2f} MB")
            logging.info(f"\tFinal CPU memory usage: {cpu_res:.2f} MB")
            logging.info(f"\tMax CPU memory usage: {self.max_cpu_res:.2f} MB")

        if gpu_res is not None:
            for gpu_id, g_res in gpu_res.items():
                logging.info(f"\tInitial GPU {gpu_id} memory usage: {self.initial_gpu_res[gpu_id]:.2f} MB")
                logging.info(f"\tAverage GPU {gpu_id} memory usage: {self.avg_gpu_res[gpu_id]:.2f} MB")
                logging.info(f"\tFinal GPU {gpu_id} memory usage: {g_res:.2f} MB")
                logging.info(f"\tMax GPU {gpu_id} memory usage: {self.max_gpu_res[gpu_id]:.2f} MB")

    def _zip_gpu_res(self, gpu_res):
        """
        Zip a list of GPU stats to a dict keyed by the selected GPU ids.
        """
        if gpu_res is None:
            return None

        keys = self.gpu_ids if self.gpu_ids is not None else list(range(len(gpu_res)))
        if len(keys) != len(gpu_res):
            logging.warning("Mismatch between provided GPU ids and measured GPUs. Falling back to enumeration.")
            keys = list(range(len(gpu_res)))
        return {g: g_res for g, g_res in zip(keys, gpu_res)}
