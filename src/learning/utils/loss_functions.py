import torch
import torch.nn.functional as F


def alignment_loss(o: torch.Tensor, t: torch.Tensor):

    ot = torch.transpose(o, 0, 1)
    tt = torch.transpose(t, 0, 1)

    O = o - ot
    T = t - tt

    align = torch.mul(O, T)
    align = F.sigmoid(align)
    loss = -torch.mean(align)

    return loss
