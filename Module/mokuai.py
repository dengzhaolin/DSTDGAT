import math
import torch
import torch.nn as nn

from torch.nn.utils import weight_norm
class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

import numpy as np
import torch
from torch import nn
from torch.nn import init



class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, l, n, c = x.size()
        x_=x.permute(0,3,2,1)
        y = self.avg_pool(x_).view(b, c)
        y = self.fc(y).view(b, c, 1, 1).permute(0,3,2,1)
        return x * y.expand_as(x)



class AdaptiveMixtureUnits(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(AdaptiveMixtureUnits, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.adaptive_act_fn = torch.sigmoid
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, input_tensor, global_output, local_output): ### B L N D
        input_tensor_avg = torch.mean(input_tensor, dim=[1,2])  # [B, D]
        ada_score_alpha = self.adaptive_act_fn(self.linear(input_tensor_avg)).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        ada_score_beta = 1 - ada_score_alpha

        mixture_output = torch.mul(global_output, ada_score_beta) + torch.mul(local_output,
                                                                              ada_score_alpha)  # [B, N, D]


        output = self.LayerNorm(self.dropout(self.linear_out(mixture_output)) + input_tensor)  # [B, N, D]
        return output


class SqueezeExcitationAttention(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super(SqueezeExcitationAttention, self).__init__()
        self.dense_1 = nn.Linear(channels, channels // reduction_ratio)
        self.squeeze_act_fn = nn.ReLU()

        self.dense_2 = nn.Linear(channels // reduction_ratio, channels)
        self.excitation_act_fn = torch.sigmoid

    def forward(self, input_tensor):
        input_tensor_avg = torch.mean(input_tensor, dim=-1, keepdim=True)  # [B, N, 1]

        hidden_states = self.dense_1(input_tensor_avg.permute(0, 2, 1))  # [B, 1, N] -> [B, 1, N/r]
        hidden_states = self.squeeze_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)  # [B, 1, N/r] -> [B, 1, N]
        att_score = self.excitation_act_fn(hidden_states)  # sigmoid

        # reweight
        input_tensor = torch.mul(input_tensor, att_score.permute(0, 2, 1))  # [B, N, D]
        return input_tensor



"""
class AdaptedLinear(nn.Module):
    ##Modified linear layer

    def __init__(self, in_features, out_features, bias=True,
                gamma=4):
        super(AdaptedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(
            out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(
                out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        adaptor_size = torch.Size([self.weight.size(0), self.weight.size(1), 2])
        self.adaptorFFN= self.weight.data.new(adaptor_size)

        self.adaptorFFN[:,:,0].fill_(gamma)
        self.adaptorFFN[:,:,1].fill_(-gamma)
        self.adaptorFFN = Parameter(self.adaptorFFN)

    def forward(self, input):

        if self.training:
            adaptor_thresholded = F.gumbel_softmax(self.adaptorFFN, hard=True)
            adaptor_thresholded = adaptor_thresholded[..., 0]

        else:
            adaptor_thresholded = self.adaptor[...,0] > self.adaptor[...,1]

        weight_thresholded = adaptor_thresholded * self.weight
        return F.linear(input, weight_thresholded, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)


"""
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size  # 这个chomp_size就是padding的值

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 这个就是TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 里面提到的downsample就是下采样，其实就是实现残差链接的部分。不理解的可以无视这个
class Causal(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(Causal, self).__init__()

        padding=kernel_size-1


        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1,self.dropout1, self.norm1,
                                 self.conv2, self.chomp2, self.dropout2, self.norm2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):  ### b l d

        x = x.permute(0, 2, 1)

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
"""
Ada=torch.randn(325,64)
se=SE(64,2,Ada_embed=Ada)
x=torch.randn(64,12,325,64)
y=se(x)
print(y.shape)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
def asym_adj(adj):
    #adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= np.diag(d_inv)
    return d_mat.dot(adj)


def compute_support_gwn(adj):
    adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
    support = [torch.tensor(i) for i in adj_mx]
    return support



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
