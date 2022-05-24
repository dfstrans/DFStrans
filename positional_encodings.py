import torch
import torch.nn as nn

class VanillaEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(VanillaEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_multidim = self.pe[:x.size()[0], :].unsqueeze(-1)

        return x + pe_multidim


class DFTEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(DFTEncoding, self).__init__()
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        w_s = 2 * torch.pi / d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.arange(0, d_model, 2).float() // 2
        pe[:, 1::2] = torch.sin(position * w_s * div_term)
        pe[:, 0::2] = torch.cos(position * w_s * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.d_model = d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_multidim = self.pe[1:x.size()[0] + 1, :].unsqueeze(-1)
        pe_multidim[0] = math.sqrt(1 / self.d_model) * pe_multidim[0]
        pe_multidim[-1] = math.sqrt(1 / self.d_model) * pe_multidim[-1]
        pe_multidim[1:-1] = torch.mul(pe_multidim[1:-1], math.sqrt(2 / self.d_model))

        pe_multidim.repeat(1, x.size()[1], 1, x.size()[3]).size()

        return x + pe_multidim