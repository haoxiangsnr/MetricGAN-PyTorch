import importlib
import time

import torch


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg):
    """
    According to the configuration items, load specific module dynamically.

    e.g. configuration as follow：

        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }

        1. Load the module corresponding to the "module" param.
        2. Call function (or instantiate class) corresponding to the "main" param.
        3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\t "
        f"max: {floor_tensor(torch.max(tensor))}, "
        f"min: {float(torch.min(tensor))}, "
        f"mean: {floor_tensor(torch.mean(tensor))}, "
        f"std: {floor_tensor(torch.std(tensor))}"
    )


def set_requires_grad(nets, requires_grad=False):
    """
    Args:
        nets(list): networks
        requires_grad(bool): True or False
    """
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad