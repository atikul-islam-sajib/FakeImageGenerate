import torch
import pickle
import os
import torch.nn as nn


def create_pickle(value=None, filename=None):
    """
    Serializes and saves a Python object to a file using pickle.

    This function takes a Python object and a filename, and serializes the object to a file with the specified name.
    If either the value or the filename is not provided, the function raises an exception.

    Parameters:
    - value: The Python object to serialize. Must not be None for the operation to proceed.
    - filename: The name of the file where the serialized object should be saved. Must not be None for the operation to proceed.

    Raises:
    - Exception: If either `value` or `filename` is None, indicating incomplete arguments for the operation.
    """
    if value is not None and filename is not None:
        with open(filename, "wb") as file:
            pickle.dump(value, file)
    else:
        raise Exception("Pickle file is empty".capitalize())


def total_params(model):
    """
    Calculates and prints the total number of parameters in a PyTorch model.

    Parameters:
        model (torch.nn.Module): The model to calculate parameters for.

    Returns:
        None. This function prints the total number of parameters directly.

    Example:
        >>> model = torch.nn.Linear(10, 5)
        >>> total_params(model)
        55
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    return total_params


def weight_init(m):
    """
    Applies custom weight initialization to convolutional and batch normalization layers of a neural network.

    This function initializes the weights of convolutional layers (`Conv2d`, `ConvTranspose2d`, etc.) with a normal distribution
    centered at 0 with a standard deviation of 0.02. For batch normalization layers (`BatchNorm2d`, etc.), it initializes the weights
    with a normal distribution centered at 1 with a standard deviation of 0.02, and sets the bias to 0.

    This initialization can help in stabilizing the learning process in generative adversarial networks (GANs) or other deep learning models.

    Parameters:
        m (torch.nn.Module): A PyTorch module, typically a layer of a neural network model. The function checks the class name of `m`
                             to determine whether it is a convolutional layer or a batch normalization layer and applies the appropriate
                             initialization.

    Example:
        >>> model = MyModel()
        >>> model.apply(weight_init)

    Note:
        The function directly modifies the input module `m` and does not return any value. It is designed to be used with the `apply`
        method of `torch.nn.Module`, which applies a function recursively to every submodule.
    """
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(tensor=m.weight.data, mean=0.0, std=0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(tensor=m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(tensor=m.bias.data, val=0)


def device_init(device=None):
    """
    Initializes and returns a PyTorch device based on the specified preference and availability.

    This function is designed to facilitate the dynamic selection of the computational device for PyTorch operations,
    allowing the user to preferentially select 'mps' (Apple Silicon GPU) or 'cuda' (NVIDIA GPU) devices. If the preferred
    device is not available, or if no preference is specified, it defaults to using the CPU.

    Parameters:
        device (str, optional): The preferred device to use for PyTorch operations. Valid options are 'mps', 'cuda', or None.
                                If 'mps' is selected, the function will check for Apple Silicon GPU availability. For 'cuda',
                                it checks for NVIDIA GPU availability. If None or if the preferred device is not available,
                                it defaults to 'cpu'. Defaults to None.

    Returns:
        torch.device: The initialized PyTorch device object, ready to be used for model operations or data manipulation.

    Example:
        >>> device = device_init("cuda")
        >>> print(device)
        device(type='cuda')  # This output depends on the availability of CUDA.

    Note:
        - The 'mps' device option is specifically for machines with Apple Silicon GPUs.
        - It's recommended to explicitly specify your device preference when calling this function to ensure your computations
          are performed on the desired hardware, when available.
    """
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")
