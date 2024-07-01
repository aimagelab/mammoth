try:
    from resource import getrusage, RUSAGE_CHILDREN, RUSAGE_SELF

    def get_memory_mb():
        """
        Get the memory usage of the current process and its children.

        Returns:
            dict: A dictionary containing the memory usage of the current process and its children.

            The dictionary has the following
            keys:
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

try:
    import torch

    if torch.cuda.is_available():
        from utils.conf import get_alloc_memory_all_devices

        def get_memory_gpu_mb():
            """
            Get the memory usage of all GPUs in MB.
            """

            return [d / 1024 for d in get_alloc_memory_all_devices()]
    else:
        get_memory_gpu_mb = None
except BaseException:
    get_memory_gpu_mb = None

from utils.loggers import Logger


class track_system_stats:
    """
    A context manager that tracks the memory usage of the system.
    Tracks both CPU and GPU memory usage if available.

    Usage:
    with track_system_stats() as t:
        for i in range(100):
            ... # Do something
            t()

    cpu_res, gpu_res = t.cpu_res, t.gpu_res

    Args:
        logger (Logger): external logger.
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
            gpu_res = get_memory_gpu_mb()

        return cpu_res, gpu_res

    def __init__(self, logger: Logger = None, disabled=False):
        self.logger = logger
        self.disabled = disabled
        self._it = 0

    def __enter__(self):
        if self.disabled:
            return self
        self.initial_cpu_res, self.initial_gpu_res = self.get_stats()
        self.initial_gpu_res = {g: g_res for g, g_res in enumerate(self.initial_gpu_res)}

        self.avg_gpu_res = self.initial_gpu_res
        self.avg_cpu_res = self.initial_cpu_res

        self.max_cpu_res = self.initial_cpu_res
        self.max_gpu_res = self.initial_gpu_res

        if self.initial_cpu_res is None and self.initial_gpu_res is None:
            self.disabled = True

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

        cpu_res, gpu_res = self.get_stats()
        self.update_stats(cpu_res, gpu_res)

    def update_stats(self, cpu_res, gpu_res):
        """
        Update the memory usage statistics.

        Args:
            cpu_res (float): The memory usage of the CPU.
            gpu_res (list): The memory usage of the GPUs.
        """
        if self.disabled:
            return

        self._it += 1

        alpha = 1 / self._it
        if self.initial_cpu_res is not None:
            self.avg_cpu_res = self.avg_cpu_res + alpha * (cpu_res - self.avg_cpu_res)
            self.max_cpu_res = max(self.max_cpu_res, cpu_res)

        if self.initial_gpu_res is not None:
            self.avg_gpu_res = {g: (g_res + alpha * (g_res - self.avg_gpu_res[g])) for g, g_res in enumerate(gpu_res)}
            self.max_gpu_res = {g: max(self.max_gpu_res[g], g_res) for g, g_res in enumerate(gpu_res)}

        if self.logger is not None:
            self.logger.log_system_stats(cpu_res, gpu_res)

    def print_stats(self):
        """
        Print the memory usage statistics.
        """

        cpu_res, gpu_res = self.get_stats()

        # Print initial, average, final, and max memory usage
        print("System stats:")
        if cpu_res is not None:
            print(f"\tInitial CPU memory usage: {self.initial_cpu_res:.2f} MB", flush=True)
            print(f"\tAverage CPU memory usage: {self.avg_cpu_res:.2f} MB", flush=True)
            print(f"\tFinal CPU memory usage: {cpu_res:.2f} MB", flush=True)
            print(f"\tMax CPU memory usage: {self.max_cpu_res:.2f} MB", flush=True)

        if gpu_res is not None:
            for gpu_id, g_res in enumerate(gpu_res):
                print(f"\tInitial GPU {gpu_id} memory usage: {self.initial_gpu_res[gpu_id]:.2f} MB", flush=True)
                print(f"\tAverage GPU {gpu_id} memory usage: {self.avg_gpu_res[gpu_id]:.2f} MB", flush=True)
                print(f"\tFinal GPU {gpu_id} memory usage: {g_res:.2f} MB", flush=True)
                print(f"\tMax GPU {gpu_id} memory usage: {self.max_gpu_res[gpu_id]:.2f} MB", flush=True)
