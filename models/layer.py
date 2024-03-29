import torch
import torch.nn as nn
import torch.nn.functional as F

### follow the implementation of nanoGPT
### https://github.com/karpathy/nanoGPT/


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """follow the implementation of nanoGPT
    https://github.com/karpathy/nanoGPT/"""

    def __init__(self, config):
        super().__init__()
        assert (config.in_channels) % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.in_channels, 3 * config.in_channels, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.in_channels, config.in_channels, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.n_ma, config.n_ma))
                                        .view(1, 1, config.n_ma, config.n_ma))

    def forward(self, x):
        B, n_ma, CB = x.size() 
        # batch size, number of moving averages (n_ma), number of channels (open, high, low, close, volume) * block_size
        # dim1 is typically the sequence length and dim2 is the embedding size of language models

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).chunk(3, dim=-1) # (B, n_ma, 3 * hs) -> 3 * (B, n_ma, hs
        k = k.view(B, n_ma, self.n_head, CB // self.n_head).transpose(1, 2) # (B, nh, n_ma, hs)
        q = q.view(B, n_ma, self.n_head, CB // self.n_head).transpose(1, 2) # (B, nh, n_ma, hs)
        v = v.view(B, n_ma, self.n_head, CB // self.n_head).transpose(1, 2) # (B, nh, n_ma, hs)

        # causal self-attention; Self-attend: (B, nh, n_ma, hs) x (B, nh, hs, n_ma) -> (B, nh, n_ma, n_ma)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)) ** 0.5)
            att = att.masked_fill(self.bias[:,:,:n_ma,:n_ma] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, n_ma, n_ma) x (B, nh, n_ma, hs) -> (B, nh, n_ma, hs)

        y = y.transpose(1, 2).contiguous().view(B, n_ma, CB) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y)) # (B, n_ma, nh * hs = in_channels) 
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.in_channels, 4 * config.in_channels, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.in_channels, config.in_channels, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.in_channels, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.in_channels, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
