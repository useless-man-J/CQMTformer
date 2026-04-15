import torch
import torch.nn as nn
from layers.Tools.Embed import DataEmbedding
from layers.Tools.Decomp import Seasonal_Prediction, series_decomp_multi


class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__()
        self.c_out = configs.c_out
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.device = configs.devices

        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.d_layers = configs.d_layers
        self.dropout = configs.dropout
        self.embed = configs
        self.freq = configs.freq
        self.mode = configs.mode

        self.decomp_kernel = configs.decomp_kernel
        self.conv_kernel = configs.conv_kernel
        self.isometric_kernel = configs.isometric_kernel

        self.decomp_multi = series_decomp_multi(self.decomp_kernel)

        # embedding
        self.dec_embedding = DataEmbedding(configs.dec_in, self.d_model, self.embed, self.freq, self.dropout)

        self.conv_trans = Seasonal_Prediction(
            embedding_size=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
            d_layers=self.d_layers,
            decomp_kernel=self.decomp_kernel,
            c_out=self.c_out,
            conv_kernel=self.conv_kernel,
            isometric_kernel=self.isometric_kernel,
            device=self.device
        )

        self.regression = nn.Linear(self.seq_len, self.pred_len)
        self.regression.weight = nn.Parameter((1 / self.pred_len) * torch.ones([self.pred_len, self.seq_len]),
                                              requires_grad=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # trend-cyclical prediction block: regre or mean
        if self.mode == 'regre':
            seasonal_init_enc, trend = self.decomp_multi(x_enc)
            trend_permuted = trend.permute(0, 2, 1)
            trend = self.regression(trend_permuted).permute(0, 2, 1)

        elif self.mode == 'mean':
            seasonal_init_enc, trend = self.decomp_multi(x_enc)
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            trend = torch.cat([trend[:, -self.seq_len:, :], mean], dim=1)

        # embedding
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.label_len:, :], zeros], dim=1)
        dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]

        return dec_out