import torch
import torch.nn.functional as F


def stack_and_pad_1d_tensors(tensors, padding_value=-1):
    """
    Stack a list of 1D tensors with different lengths by padding them to the same length.

    Args:
        tensors (list of torch.Tensor): List of 1D tensors to stack.
        padding_value (int, float): Value to use for padding.

    Returns:
        torch.Tensor: A single 2D tensor containing all the padded tensors stacked along the first dimension.
    """
    # Determine the maximum length among all tensors
    max_length = max(tensor.size(0) for tensor in tensors)

    padded_tensors = []
    for tensor in tensors:
        # Calculate the padding size
        pad_size = max_length - tensor.size(0)

        # Pad the tensor (on the right)
        padded_tensor = F.pad(tensor, (0, pad_size), value=padding_value)
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors along a new dimension
    stacked_tensor = torch.stack(padded_tensors, dim=0)
    return stacked_tensor


def pad_width(tensor, target_width, padding_value=-1):
    """
    Pad only the width (columns) of a 2D tensor to the specified target width.

    Args:
        tensor (torch.Tensor): Input 2D tensor (shape: [H, W]).
        target_width (int): Desired width after padding.
        padding_value (int, float): Value used for padding.

    Returns:
        torch.Tensor: Padded 2D tensor with the same height but padded width.
    """
    current_height, current_width = tensor.size()

    # Calculate the padding size for width
    pad_width = target_width - current_width

    # Ensure the padding is non-negative
    assert (
        pad_width >= 0
    ), "Target width must be greater than or equal to the input width."

    # Padding is applied as (left, right, top, bottom)
    padding = (0, pad_width, 0, 0)  # Only pad the width (right side)

    # Apply the padding
    padded_tensor = F.pad(tensor, padding, value=padding_value)
    return padded_tensor
