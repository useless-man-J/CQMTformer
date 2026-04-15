import torch.nn as nn
from layers.Mamba.Mamba_Family import Mamba_Layer
from layers.Tools.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp, DataEmbedding_onlypos
from mamba_ssm import Mamba
class Mamba_enc(nn.Module):
    """
    Mamba
    """
    def __init__(self, configs):
        super(Mamba_enc, self).__init__()
        self.configs = configs
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding_onlypos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.mamba_layers = nn.Sequential(*[
            Mamba_Layer(
                Mamba(configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv),
                configs.d_model
            )
            for i in range(configs.d_layers)
        ])
        self.linear = nn.Linear(configs.seq_len, configs.pred_len)
        self.dropout = nn.Dropout(configs.dropout)
    def forward(self, x_enc, x_mark_enc):
        x = self.enc_embedding(x_enc, x_mark_enc)
        cross_in = self.mamba_layers(x)
        cross_in = cross_in.permute(0, 2, 1)
        cross_out = self.linear(cross_in)
        cross_out = cross_out.permute(0, 2, 1)
        cross_out = self.dropout(cross_out)
        return cross_out