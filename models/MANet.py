import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import LayerNorm, TransformerBlock

### The implementation of Moving Average Neural Net, currently only support torch ###

class MANet:
    def __new__(cls, model_config):
        if model_config["lib"] == "tensorflow" or model_config["lib"] == "numpy":
            raise NotImplementedError(f"MANet currently not implemented for {model_config.lib}.")
        elif model_config["lib"] == "torch":
            return MANet_torch(model_config)
    
class MANet_torch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # kernel size is the length of the moving average window, dist is the distance between the windows
        # stride is the time steps between each time moving averaging is applied
        self.average_pools = nn.ModuleList([nn.AvgPool1d(kernel_size = kernel_size, stride = config.stride) 
                                            for kernel_size in range(config.min_kernel, config.max_kernel + 1, config.dist)])

        
        self.transformer = nn.ModuleDict(dict(
            # pe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.in_channels, bias = config.bias),
        ))
        self.lm_head = nn.Linear(config.in_channels * config.n_ma, 4, bias = False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        B, C, T = idx.shape # batch size, number of channels, sequence length
        # pos = torch.arange(0, self.config.block_size, self.config.stride, dtype=torch.long, device=device) # shape (block_size,)

        xs = []
        for pool in self.average_pools:
            temp = pool(idx[..., -(pool.kernel_size[0] + pool.stride[0] * (self.config.block_size - 1)):]) # shape (B, C, block_size)
            xs.append(temp)
        
        x = torch.stack(xs, dim=1) # shape (B, n_ma, C, block_size)
        x = x.view(B, self.config.n_ma, -1) # shape (B, n_ma, C * block_size = in_channels)

        # pos_emb = self.transformer.pe(pos) # position embeddings of shape (t, n_ma)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # shape (B, n_ma, in_channels)
        x = x.view(B, -1)
        x = self.lm_head(x) # shape (B, n_ma, in_channels)

        return x