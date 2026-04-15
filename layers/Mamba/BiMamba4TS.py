import torch
import torch.nn as nn
import numpy as np
from layers.Mamba.BiMamba4TS_layers import Encoder, EncoderLayer
from layers.Mamba.BiMamba_Embed import PatchEmbedding, TruncateModule
from einops import rearrange
from mamba_ssm import Mamba


class BiMamba_enc(nn.Module):
    def __init__(self, configs, corr=None):
        super(BiMamba_enc, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin = configs.revin
        self.d_model = configs.d_model
        self.embed_type = configs.embed_type
        if configs.SRA:
            self.ch_ind = self.SRA(corr, configs.threshold)
        else:
            self.ch_ind = configs.ch_ind

        # patching
        if configs.seq_len % configs.stride == 0:
            self.patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 1)
            process_layer = nn.Identity()
        else:
            if configs.padding_patch == "end":
                padding_length = configs.stride - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 2)
                process_layer = nn.ReplicationPad1d((0, padding_length))
            else:
                truncated_length = configs.seq_len - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 1)
                process_layer = TruncateModule(truncated_length)

        self.process_layer = process_layer

        self.local_token_layer = PatchEmbedding(
            configs.seq_len,
            configs.d_model,
            configs.patch_len,
            configs.stride,
            configs.dropout,
            process_layer=self.process_layer,
            ch_ind=self.ch_ind
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=configs.d_model,
                          d_state=configs.d_state,
                          d_conv=configs.d_conv,
                          expand=configs.e_fact,
                          use_fast_path=True),
                    Mamba(d_model=configs.d_model,
                          d_state=configs.d_state,
                          d_conv=configs.d_conv,
                          expand=configs.e_fact,
                          use_fast_path=True),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    bi_dir=configs.bi_dir,
                    residual=configs.residual == 1
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 修改head，输出pred_len长度
        head_nf = configs.d_model * self.patch_num
        self.head = Flatten_Head(False, configs.enc_in, head_nf, self.pred_len)  # 改为pred_len

    def SRA(self, corr, threshold):
        high_corr_matrix = corr >= threshold
        num_high_corr = np.maximum(high_corr_matrix.sum(axis=1) - 1, 0)

        positive_corr_matrix = corr >= 0
        num_positive_corr = np.maximum(positive_corr_matrix.sum(axis=1) - 1, 0)
        max_high_corr = num_high_corr.max()
        max_positive_corr = num_positive_corr.max()
        r = max_high_corr / max_positive_corr
        print('SRA -> channel mixing' if r >= 1 - threshold else 'channel independent')
        return 0 if r >= 1 - threshold else 1

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # B L M
        B, L, M = x_enc.shape

        # [B, M, L] -> [B*M, N, D] or [B*(M+4), N, D]
        enc_out, _ = self.local_token_layer(x_enc.permute(0, 2, 1),
                                            x_mark_enc.permute(0, 2, 1) if self.embed_type == 2 else None)
        if not self.ch_ind:
            enc_out = rearrange(enc_out, '(B M) N D -> (B N) M D', B=B)

        enc_out = self.encoder(enc_out)

        # output: [B*M, N, D] or [B*N, M, D] -> [B x M x H]
        if not self.ch_ind:
            dec_out = rearrange(enc_out, '(B N) M D -> B M N D', B=B)
        else:
            dec_out = rearrange(enc_out, '(B M) N D -> B M N D', B=B)

        # 修改这里：输出形状应该是 [B, pred_len, M]
        dec_out = self.head(dec_out).permute(0, 2, 1)

        return dec_out


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0.):
        super().__init__()
        self.ch_ind = individual
        self.n_vars = n_vars

        if self.ch_ind:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B, C, D, N]
        if self.ch_ind:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [B, D * N]
                z = self.linears[i](z)  # z: [B, target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [B, C, target_window]
        else:
            x = self.flatten(x)  # x: [B, C, D * N]
            x = self.linear(x)  # x: [B, C, target_window]
            x = self.dropout(x)
        return x