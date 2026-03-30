import numpy as np
import torch


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
