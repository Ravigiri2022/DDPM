import os
import socket
import torch as th
import torch.distributed as dist

# Set up for single-node, single-process mode (no distributed training)
def setup_dist():
    """
    Setup a fake distributed process group for single-process (non-distributed) training.
    """
    if dist.is_initialized():
        return

    print("Running in single-process (non-distributed) mode.")

    # Simulate a single process group with rank 0 and world size 1
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Set backend depending on whether CUDA is available
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    
    # Dummy setup for process group, since we are in single-process mode
    dist.init_process_group(backend=backend, init_method="file:///tmp/tmpfile", rank=0, world_size=1)


def dev():
    """
    Get the device to use for training.
    """
    # In single-process mode, always use rank 0
    return th.device("cuda:0" if th.cuda.is_available() else "cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    In single-process mode, just load the file directly.
    """
    if dist.get_rank() == 0:
        with open(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = dist.broadcast(data, src=0)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks.
    Since we're in single-process mode, just return without any synchronization.
    """
    pass  # No need to sync params in single-process mode.


def _find_free_port():
    """
    Find a free port on the machine to use for communication (for distributed training).
    Since we're not using distributed training, this function is not used, but kept for completeness.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
