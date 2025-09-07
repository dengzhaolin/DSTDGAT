import torch.nn as nn
import torch
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
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

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out
class SelfAdaAttentionLayer(nn.Module):
    def __init__(
        self, model_dim,len, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AdaAttentionLayer(model_dim,len, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)

        return out





class AdaAttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, len,num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

        self.position_bias = nn.Parameter(torch.randn(len, len), requires_grad=True)



    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]



        position_bias=self.position_bias.to(query.device)



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

        #position_bias = position_bias.expand(attn_score.shape[0],attn_score.shape[1], *position_bias.shape)
        position_bias = position_bias.unsqueeze(0).unsqueeze(0)

        attn_score=attn_score+position_bias

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out



class Causal_Aware_Attention(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=True):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads


        self.global_q = nn.Linear(model_dim, model_dim)
        self.global_k = nn.Linear(model_dim, model_dim)
        self.global_v= nn.Linear(model_dim, model_dim)
        self.causal_q = nn.Linear(model_dim, model_dim)
        self.mask_q = nn.Linear(model_dim, model_dim)
        self.mask_k = nn.Linear(model_dim, model_dim)
        self.causal_k = nn.Linear(model_dim, model_dim)
        self.causal_v = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        global_query = self.global_q(query)
        global_key = self.global_k(key)
        global_value=self.global_v(value)

        causal_query = self.causal_q(query)
        causal_key = self.causal_k(key)
        causal_value = self.causal_v(value)

        mask_q=self.mask_q(query)

        mask_k=self.mask_k(key)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        global_query= torch.cat(torch.split(global_query, self.head_dim, dim=-1), dim=0)
        global_key = torch.cat(torch.split(global_key, self.head_dim, dim=-1), dim=0)
        global_value = torch.cat(torch.split(global_value, self.head_dim, dim=-1), dim=0)
        local_query = torch.cat(torch.split(causal_query, self.head_dim, dim=-1), dim=0)
        local_key = torch.cat(torch.split(causal_key, self.head_dim, dim=-1), dim=0)
        local_value = torch.cat(torch.split(causal_value, self.head_dim, dim=-1), dim=0)
        mask_q = torch.cat(torch.split(mask_q, self.head_dim, dim=-1), dim=0).reshape(-1,tgt_length,self.head_dim)
        mask_k = torch.cat(torch.split(mask_k, self.head_dim, dim=-1), dim=0).reshape(-1,tgt_length,self.head_dim)


        global_key = global_key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        global_attn_weight = (
            global_query @ global_key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)


        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            global_attn_weight.masked_fill_(~mask, -torch.inf)  # fill in-place

        local_key = local_key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        local_attn_weight = (
                                local_query @ local_key
                             ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        local_att_mask = self.scan_local_masking(mask_q, mask_k)

        local_att_mask=local_att_mask.reshape(batch_size*self.num_heads,-1,local_att_mask.shape[-2],local_att_mask.shape[-1])

        masked_local_attn_weight = local_attn_weight * local_att_mask

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            masked_local_attn_weight.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_weights = 0.1*global_attn_weight  + masked_local_attn_weight  # 0.1


        attn_score = torch.softmax(attn_weights, dim=-1)
        out = attn_score @ global_value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out

    def scan_local_masking(self, q_mask, k_mask):

        attention = torch.bmm(q_mask, k_mask.transpose(1, 2))

        time_len = attention.shape[1]

        key = attention.clone()


        _, max_indices = torch.max(attention, dim=-1)

        key = F.relu(key)

        max_indices = max_indices.unsqueeze(2).expand(*max_indices.shape, attention.shape[-1])


        dis_mask = torch.ones(time_len, device=q_mask.device, dtype=q_mask.dtype)

        for i in range(time_len):

            dis_mask[i] = i

        dis_mask = dis_mask.expand(attention.shape[0], time_len, time_len)

        dis_mask = F.normalize(torch.abs(dis_mask - max_indices), p=1, dim=-1)
        mean_value = torch.mean((key * dis_mask), dim=-1)

        mean_value = mean_value.unsqueeze(2).expand(*attention.shape)
        key = dis_mask * attention
        query = key - mean_value

        query = torch.where(query < 0, torch.tensor(0, device=q_mask.device, dtype=q_mask.dtype),
                            torch.tensor(1, device=q_mask.device, dtype=q_mask.dtype))



        return query
class Temporal_Aware_Learning(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = Causal_Aware_Attention(model_dim, num_heads)


        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):

        b,l,n,c=x.shape


        x = x.transpose(dim, -2)

        residual = x
        # x: (batch_size, ..., length, model_dim)


        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)

        out = self.dropout1(out)

        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)

        return out
