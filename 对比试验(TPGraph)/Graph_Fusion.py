import torch
import numpy as np
import torch.nn as nn
import torch


def get_normalize(graph, normalize=True):
    """
    return the laplacian of the graph.

    :param graph: the graph structure without self loop, [N, N].
    :param normalize: whether to used the normalized laplacian.
    :return: graph laplacian.
    """
    if normalize:
        D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
        I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        A1 = graph + I
        L = torch.mm(torch.mm(D, A1), D)

    return L


class GF(nn.Module):
    def __init__(self):
        super(GF, self).__init__()
        self.weight = nn.Parameter(torch.randn(2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, B):
        print(self.weight)
        w1, w2 = self.sigmoid(self.weight)
        print(w1)
        print(w2)
        out=w1 * A + w2 * B
        print(out.shape)
        print(out)
        out = torch.add(w1 * A, w2 * B)
        print(out)
        print(out.shape)

