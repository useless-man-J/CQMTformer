from layers.Tools.RevIN import RevIN
from layers.My_model.Decomp import SeriesDecomp
import torch.nn as nn
from math import sqrt
import torch
from layers.Tools.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.My_model.Transformer_EncDec import Decoder, DecoderLayer_Single, Encoder, DecoderLayer, EncoderLayer,BiEncoderLayer
from layers.Attention.SelfAttention_Family import FullAttention, AttentionLayer
import numpy as np
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba

class Model(nn.Module):

    def __init__(self, configs, **kwargs):
        super().__init__()
        
        # Input
        self.c_in = configs.enc_in
        self.c_out = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch_size = configs.batch_size

        # Decomp
        self.kernel_size = configs.kernel_size
        self.decomp = SeriesDecomp(kernel_size=self.kernel_size)

        # Norm
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last
        self.revin = RevIN(int(self.c_in), affine=self.affine, subtract_last=self.subtract_last)

        # Model_size
        self.d_model = configs.d_model

        # Patch
        self.patch_len = configs.patch_len
        self.patchmamba = PatchModel(self.c_in, self.pred_len, self.patch_len, configs.d_state, self.d_model, configs.dropout, configs.factor, configs.n_heads, model='mamba', flatten='p')

        # Tokenization
        self.enc_embedding_i = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Mamba
        self.mamba1 = Mamba(
            d_model=configs.d_model,  # Model dimension d_model
            d_state=configs.d_state,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor
        )
        self.mamba2 = Mamba(
            d_model=configs.d_model,  # Model dimension d_model
            d_state=configs.d_state,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor
        )

        self.attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),configs.d_model, configs.n_heads)
        
        self.norm = nn.LayerNorm(self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.gate = GateFusion(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_p = nn.Parameter(torch.zeros(configs.batch_size, configs.enc_in + 4, configs.d_model), requires_grad=True)
        nn.init.xavier_normal_(self.channel_p)

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        batch_x = self.revin(batch_x, 'norm')
        _,_,N = batch_x.shape
        x_enc = self.enc_embedding_i(batch_x, batch_x_mark)
        x1 = x_enc + self.dropout(self.attention(self.channel_p, x_enc, x_enc, attn_mask=None)[0])
        x1 = self.norm1(x1)
        x2 = x_enc + self.dropout(self.mamba1(x_enc) + self.mamba2(x_enc.flip(dims=[-1])).flip(dims=[-1]))
        x2 = self.norm(x2)
        x = self.gate(x1, x2)
        x = x_enc + x
        x = self.norm(x)
        x = self.patchmamba(x)
        out = x.permute(0, 2, 1)[:, :, :N]
        out = self.revin(out, 'denorm')
        return out
    

class PatchModel(nn.Module):
    def __init__(self, c_in, pred_len, patch_len, d_state, d_model, dropout, factor, n_heads, model:str, flatten:str):
        super().__init__()
        self.patch_num = int(d_model // patch_len)
        self.head_nf = d_model * self.patch_num
        self.w_p = nn.Linear(patch_len, d_model)
        self.head = FlattenHead(c_in, self.head_nf, pred_len, head_dropout=dropout)
        self.head_d = FlattenHead(c_in, self.head_nf, d_model, head_dropout=dropout)
        self.model = model
        self.flatten = flatten
        self.patch_len = patch_len

        #model
        self.mlp = MLP(d_model, d_model * 2, d_model)

        self.attention = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads)
        
        self.mamba = Mamba(
            d_model=d_model,    # Model dimension d_model
            d_state=d_state,    # SSM state expansion factor
            d_conv=2,           # Local convolution width
            expand=1,           # Block expansion factor
        )

    def forward(self, x):                                                                       # x: [bs x nvars x d_model]
        x_p = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                  # x: [bs x nvars x patch_num x patch_len]
        n_vars = x_p.shape[1]
        x_p = self.w_p(x_p)                                                                     # x: [bs x nvars x patch_num x d_model]
        x_p = torch.reshape(x_p, (x_p.shape[0] * x_p.shape[1], x_p.shape[2], x_p.shape[3]))     # x: [bs * nvars x patch_num x d_model]

        if self.model == 'mlp':
            x_p = self.mlp(x_p)
        elif self.model == 'attention':
            x_p, _ = self.attention(x_p, x_p, x_p, attn_mask=None)
        elif self.model == 'mamba':
            x_p = self.mamba(x_p)

        x_p = torch.reshape(x_p, (-1, n_vars, x_p.shape[-2], x_p.shape[-1]))                    # x: [bs x nvars x patch_num x d_model]
        x_p = x_p.permute(0, 1, 3, 2)
        if self.flatten == 'd':                                                                    # x: [bs x nvars x d_model x patch_num]
            x_p = self.head_d(x_p)
        elif self.flatten == 'p':
            x_p = self.head(x_p)
        return x_p

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        output = self.model(x)
        return output


class GateFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    def forward(self, x1, x2):
        fused = torch.cat((x1, x2), dim=-1)
        gate = self.gating(fused)
        x = gate * x1 + (1 - gate) * x2
        return x
 
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


