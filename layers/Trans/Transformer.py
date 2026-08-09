import torch
import torch.nn as nn
from layers.My_model.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.My_model.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Tools.Embed import DataEmbedding_onlypos

class Transformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding_onlypos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding_onlypos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=False,
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.dec_in, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):  # 将x_dec设为可选参数
        enc_out = self.enc_embedding(x_enc,x_mark_enc)
        enc_out, attns = self.encoder(enc_out)
        dec_out = self.dec_embedding(x_dec,x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)
        return dec_out[:, -self.pred_len:, :]