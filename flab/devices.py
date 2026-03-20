import numpy as np
import torch


def as_ndarray(x, dtype=np.float64, clone=False):
    """Convert x to a numpy ndarray.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    result = np.asarray(x, dtype=dtype)
    if clone and result is x:
        return result.copy()
    return result


def as_tensor(x, dtype=torch.float32, clone=False, device=None):
    """
    Convert array-like to torch.Tensor on the desired device/dtype.
    """
    device = get_device(device)
    if not isinstance(x, torch.Tensor):
        return torch.as_tensor(x, dtype=dtype, device=device)
    if (x.device != device) or (x.dtype != dtype):
        return x.to(device=device, dtype=dtype)
    return x.clone() if clone else x


def get_device(device=None):
    """Return a torch.device, auto-selecting CUDA/MPS/CPU if none is specified."""
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif isinstance(device, int):
        return torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        return torch.device(device)
    else:
        raise ValueError(f"Invalid device: {device}")


def set_global_seed(seed, deterministic=False):
    """Seed NumPy and PyTorch RNGs globally, optionally enabling deterministic cuDNN mode."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # current device
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
