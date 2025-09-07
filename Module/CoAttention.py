import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def sim_global(flow_data, sim_type='cos'):
    # Calculate the global similarity of traffic flow data.
    #:param flow_data: tensor, original flow [n,l,v,c] or location embedding [n,v,c]
    #:param type: str, type of similarity, attention or cosine. ['att', 'cos']
    #:return sim: tensor, symmetric similarity, [v,v]

    if len(flow_data.shape) == 4:
        n, l, v, c = flow_data.shape
        att_scaling = n * l * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('btnc, btmc->nm', flow_data, flow_data)
    elif len(flow_data.shape) == 3:
        n, v, c = flow_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('bnc, bmc->nm', flow_data, flow_data)
    else:
        raise ValueError('sim_global only support shape length in [3, 4] but got {}.'.format(len(flow_data.shape)))

    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        sim = sim * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only support sim_type in [att, cos].')

    return sim





class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8,mask=None):
        super().__init__()

        self.mask=mask

        self.model_dim = model_dim
        self.num_heads = num_heads


        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value,mask):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if mask is not  None:

            attn_score=attn_score.view(batch_size,self.num_heads,-1,tgt_length,tgt_length)

            attn_score = attn_score * mask

            attn_score = attn_score.masked_fill(mask == 0, -1e9)

            attn_score=attn_score.view(batch_size*self.num_heads,-1,tgt_length,tgt_length)

        if self.mask is not None:
            mask = self.mask.bool() # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out

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


class ComAttention(nn.Module):

    def __init__(self,d_model, head, sim_score:float,sem_mask,dropout_prob):
        super(ComAttention, self).__init__()
        self.d_k = d_model // head
        self.head = head

        self.sem_mask=sem_mask

        self.fusion = nn.Sequential(
            nn.Conv2d(head,2*head,1),
            nn.Conv2d(2*head,1,1),
        )
        self.sim_score=sim_score
        self.sem_attn = AttentionLayer(d_model, head,mask=None)
        self.spa_attn = AttentionLayer(d_model, head,mask=None)

        self.norm_sim=nn.Linear(d_model,1)

        self.FC_Q = nn.Linear(d_model, d_model)
        self.FC_K = nn.Linear(d_model, d_model)
        self.FC_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.AdaMix=AdaptiveMixtureUnits(d_model,dropout_prob)

        self.fusion_weight=nn.Parameter(torch.randn(2))

    def forward(self, x):



        batch_size = x.shape[0]   ###b l n c
        tgt_length = x.shape[-2]

        query = self.FC_Q(x).view(-1, tgt_length, self.head, self.d_k).transpose(1, 2)
        key = self.FC_K(x).view(-1, tgt_length, self.head, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = torch.sigmoid(self.fusion(scores))
        zero = torch.zeros_like(scores).to(x.device)
        one = torch.ones_like(scores).to(x.device)
        p_mask = (torch.where(scores > 0.5, one, zero) * scores).view(batch_size,-1,x.shape[1],tgt_length,tgt_length)  ###b 1 l n n
        n_mask = (torch.where(scores <= 0.5, one, zero) * scores).view(batch_size,-1,x.shape[1],tgt_length,tgt_length)
        """

        sim_score= sim_global(query,'cos')
        print(sim_score)

        sim_mask=torch.where(sim_score < self.sim_score, torch.tensor(0, device=query.device, dtype=query.dtype),
                    torch.tensor(1, device=query.device, dtype=query.dtype))
        sim_mask=sim_mask.bool()
        

        
        value = self.FC_V(x)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.d_k, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d_k, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d_k, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                     query @ key
                     ) / self.d_k ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if sim_mask is not  None:
            mask = sim_mask # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        

        out_sim = self.out_proj(out)
        

        out_sem=self.sem_attn(x,x,x)

        out_spa=self.spa_attn(x,x,x)
        
        """

        out_sem =self.sem_attn(x,x,x,p_mask*self.sem_mask)
        out_sim=self.spa_attn(x,x,x,n_mask*self.sem_mask)

        out=self.fusion_weight[0]*out_sem + self.fusion_weight[-1]*out_sim



        return out

class SelfAttentionLayer_Co(nn.Module):
    def __init__(
        self,d_model,feed_forward_dim, head, sim_score:float,sem_mask,dropout_prob
    ):
        super().__init__()
        self.attn = ComAttention(d_model, head, sim_score, sem_mask,dropout_prob)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

"""
from lib.metrics import *
adj = get_adjacency_matrix1('../data/adj_bay.npy')
adj = torch.tensor(adj, dtype=torch.float32).bool()
co=ComAttention(64,4,0.2,adj,0.)
x=torch.randn(64,12,325,64)
y=co(x)
print(y.shape)
"""
