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
        #Input
        self.c_in = configs.enc_in
        self.c_out = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch_size = configs.batch_size

        #Decomp
        self.kernel_size = configs.kernel_size
        self.decomp = SeriesDecomp(kernel_size=self.kernel_size)

        #Norm
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last
        self.revin = RevIN(int(self.c_in), affine=self.affine, subtract_last=self.subtract_last)

        #Model
        self.d_model = configs.d_model

        #Patch
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding_patch = configs.padding_patch
        self.patch_num = int(configs.d_model // configs.patch_len)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(self.c_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)
        self.head_d = FlattenHead(self.c_in, self.head_nf, self.d_model, head_dropout=configs.dropout)
        #Tokenization
        self.enc_embedding_i = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.enc_embedding_i_trend = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding_i = DataEmbedding_inverted(configs.pred_len + configs.label_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.enc_no_embedding_i = nn.Linear(self.seq_len, self.d_model)
        self.enc_embedding = nn.Linear(1, self.d_model)

        #MLP
        self.mlp = MLP(self.d_model, self.d_model * 2, self.d_model)
        self.mlp_trend = MLP(self.seq_len, self.d_model, self.pred_len)
        self.fc = MLP(self.d_model, self.d_model * 2, self.d_model)
        
        self.feed = nn.Linear(self.d_model, 1)
        #Mamba
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
        self.mamba3 = Mamba(
            d_model=configs.d_model,  # Model dimension d_model
            d_state=configs.d_state,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor
        )
        #Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.biencoder = self.encoder = Encoder(
            [
                BiEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                     AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.prediction = nn.Linear(self.d_model, self.pred_len)

        #Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.pred_len, bias=True)
        )

        self.decoder_no_mask = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.pred_len, bias=True)
        )

        self.attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),configs.d_model, configs.n_heads)
        
        self.attention_m = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),configs.d_model, configs.n_heads)
        
        self.decoder_single = Decoder([
            DecoderLayer_Single(
                AttentionLayer(
                    FullAttention(False,
                                  configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model,
                    configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for l in range(configs.d_layers)
        ],
        norm_layer=torch.nn.LayerNorm(configs.d_model),
        projection=nn.Linear(configs.d_model, configs.pred_len, bias=True)
        )
        
        self.norm = nn.LayerNorm(self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(configs.dropout)
        self.w_p = nn.Linear(self.patch_len, self.d_model)
        self.position = PositionalEmbedding(self.d_model)
        self.channel_p1 = nn.Parameter(torch.zeros(configs.batch_size, configs.enc_in + 4, configs.d_model), requires_grad=True)
        self.channel_p3 = nn.Parameter(torch.zeros(configs.enc_in + 4, configs.d_model), requires_grad=True)
        self.channel_p2 = nn.Parameter(torch.zeros(configs.batch_size * (configs.enc_in + 4), self.patch_num, configs.d_model))
        self.channel_trend = nn.Parameter(torch.zeros(configs.enc_in, configs.d_model))
        nn.init.xavier_normal_(self.channel_p1)
        nn.init.xavier_normal_(self.channel_p2)
        nn.init.xavier_normal_(self.channel_trend)
        
        self.cnn_original = nn.Conv1d(self.c_in, self.c_out, kernel_size=3, padding=1)

        self.conv1 = nn.Conv1d(self.seq_len, configs.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(configs.d_ff, self.d_model, kernel_size=1)
        self.gate_conv = nn.Conv1d(self.d_model*2, self.d_model, kernel_size=1)
        self.gelu = nn.GELU()
        self.cnn = nn.Conv1d(self.c_in + 4, self.c_in + 4, kernel_size=3, padding=1)

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):

        #Test1 only_encoder
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x, _ = self.encoder(x)
        # x = self.prediction(x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test2 only_decoder
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = self.decoder(x, x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test3 only_decoder_single
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = self.decoder_single(x, x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test4 only_decoder_single + mlp 
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))

        # x1 = x + self.dropout(self.mlp(x))
        # x1 = self.norm(x1)

        # out = self.decoder_single(x1, x)
        # out = out.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test5 only_decoder_no_mask
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = self.decoder_no_mask(x, x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test6 Patch + only_encoder
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # x, _ = self.encoder(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)                                                                     # x: [bs x nvars x pred_len]
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test7 only_encoder + Patch + mlp
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x, _ = self.encoder(x)
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)                                                                     # x: [bs x nvars x pred_len]
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test8 Patch + mlp
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)                                                                     # x: [bs x nvars x pred_len]
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test9 Patch + mlp + only_decoder_single
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x_cross = x
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head_d(x)                                                                   # x: [bs x nvars x d_model]

        # out = self.decoder_single(x_cross, x)                                                
        # out = out.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test10 only_encoder + Patch + mlp + only_decoder_single
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x_cross,_ = self.encoder(x)
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head_d(x)                                                                   # x: [bs x nvars x d_model]

        # out = self.decoder_single(x_cross, x)                                                
        # out = out.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test11 Learnable parameters + encoder
        # batch_x = self.revin(batch_x,'norm')
        # x_learn = nn.Parameter(torch.zeros(self.c_in, self.d_model), requires_grad=True).to(batch_x.device)
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = x - x_learn
        # x, _ = self.encoder(x)
        # x = x + x_learn
        # x = self.prediction(x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test12 Learnable parameters + encoder + patch + Learnable parameters + mlp
        # batch_x = self.revin(batch_x,'norm')
        # x_learn1 = self.channel_p1
        # x_learn2 = self.channel_p2
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = x + x_learn1
        # x, _ = self.encoder(x)
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # x = x + x_learn2
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)                                                                     # x: [bs x nvars x pred_len]
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test13 Learnable parameters + encoder + patch + mlp
        # batch_x = self.revin(batch_x,'norm')
        # x_learn1 = self.channel_p1
        # # x_learn2 = self.channel_p2
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = x + x_learn1
        # x, _ = self.encoder(x)
        # x = x - x_learn1
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # # x = x + x_learn2
        # x = self.mlp(x)
        # # x = x - x_learn2
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)                                                                     # x: [bs x nvars x pred_len]
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test14  encoder + patch + Learnable parameters + mlp
        # batch_x = self.revin(batch_x,'norm')
        # # x_learn1 = self.channel_p1
        # x_learn2 = self.channel_p2
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # # x = x + x_learn1
        # x, _ = self.encoder(x)
        # # x = x - x_learn1
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # x = x + x_learn2
        # x = self.mlp(x)
        # x = x - x_learn2
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)                                                                     # x: [bs x nvars x pred_len]
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test15 encoder + only_decoder_single
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x_cross,_ = self.encoder(x)
        # out = self.decoder_single(x_cross, x)                                                
        # out = out.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test16 patch + mlp + only_encoder
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head_d(x)                                                                   # x: [bs x nvars x d_model]
        # x, _ = self.encoder(x)
        # x = self.prediction(x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')


        #Test17 CNN_original
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = self.cnn_original(x)
        # x = self.prediction(x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test18 CNN_Enchance
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x = self.cnn_enhance(x)
        # x = self.prediction(x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        # Test19 only_encoder + CNN
        # batch_x = self.revin(batch_x,'norm')
        # x = self.enc_no_embedding_i(batch_x.permute(0, 2, 1))
        # x, _ = self.encoder(x)
        # x = self.cnn_enhance(x)
        # x = self.prediction(x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test20 encoder + decomp + patch + mlp
        # batch_x = self.revin(batch_x,'norm')
        # res, trend = self.decomp(batch_x)
        # trend = self.mlp_trend(trend.permute(0, 2, 1))
        # x = self.enc_no_embedding_i(res.permute(0, 2, 1))
        # x, _ = self.encoder(x)
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.dropout(x + self.position(x))                                               # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head_d(x) + trend                                                           # x: [bs x nvars x pred_len]
        # x = self.prediction(x)
        # out = x.permute(0, 2, 1)
        # out = self.revin(out,'denorm')

        #Test21 attntion + mamba + Learnable parameters + patch + mlp
        # batch_x = self.revin(batch_x,'norm')
        # _, _, N = batch_x.shape
        # x = self.enc_embedding_i(batch_x, batch_x_mark)
        # x1, _ = self.attention(x, x, x,attn_mask = None)
        # x2 = self.mamba2(x.flip(dims=[1])).flip(dims=[1])
        # new_x = x1 + x2
        # x = x + new_x
        # y = x = self.norm(x)
        # x = x + self.channel_p1
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)
        # out = x.permute(0, 2, 1)[:, :, :N]
        # out = self.revin(out,'denorm')

        #Test22  decomp + Learnable parameters + attntion + mamba + patch + mlp
        # batch_x = self.revin(batch_x,'norm')
        # res, trend = self.decomp(batch_x)
        # trend = self.mlp_trend(trend.permute(0, 2, 1))
        # _, _, N = batch_x.shape
        # x = self.enc_embedding_i(res, batch_x_mark)
        # x1, _ = self.attention(x, x, x,attn_mask = None)
        # x2 = self.mamba2(x.flip(dims=[1])).flip(dims=[1])
        # new_x = x1 + x2
        # x = x + new_x
        # x = self.norm(x)
        # x = x + self.channel_p1
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)
        # out = x.permute(0, 2, 1)[:, :, :N] + trend.permute(0, 2, 1)
        # out = self.revin(out,'denorm')
        
        #Test23  Learnable parameters + attntion + mamba + patch + mlp
        # batch_x = self.revin(batch_x,'norm')
        # _, _, N = batch_x.shape
        # x = self.enc_embedding_i(batch_x, batch_x_mark)
        # x1, _ = self.attention(self.channel_p1, x, x, attn_mask = None)
        # x2 = self.mamba1(x.flip(dims=[1])).flip(dims=[1])
        # new_x = x1 + x2
        # x = x + new_x
        # x = self.norm(x)
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)
        # out = x.permute(0, 2, 1)[:, :, :N] 
        # out = self.revin(out,'denorm')

        #Test24 Learnable parameters + attntion
        # batch_x = self.revin(batch_x,'norm')
        # _, _, N = batch_x.shape
        # x_enc = self.enc_embedding_i(batch_x, batch_x_mark)
        # x = x_enc + self.channel_p1
        # new_x, _ = self.attention(x, x_enc, x_enc, attn_mask = None)
        # x = x_enc + new_x
        # x = self.norm(x)
        # x = self.prediction(x)
        # out = x.permute(0, 2, 1)[:, :, :N] 
        # out = self.revin(out,'denorm')

        #Test25 
        # batch_x = self.revin(batch_x,'norm')
        # _, _, N = batch_x.shape
        # x = self.enc_embedding_i(batch_x, batch_x_mark)
        # x1, _ = self.attention(self.channel_p1, x, x, attn_mask = None)
        # x2 = self.mamba1(x) + self.mamba2(x.flip(dims=[-1])).flip(dims=[-1])
        # new_x = x1 + x2
        # x = x + new_x
        # x = self.norm(x)
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.mlp(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)
        # out = x.permute(0, 2, 1)[:, :, :N] 
        # out = self.revin(out,'denorm')

        #Test26
        # batch_x = self.revin(batch_x,'norm')
        # _, _, N = batch_x.shape
        # x_enc = self.enc_embedding_i(batch_x, batch_x_mark)
        # x1 = x_enc + self.dropout(self.attention(self.channel_p1, x_enc, x_enc, attn_mask=None)[0])
        # x1 = self.norm1(x1)

        # B, M, D = x_enc.shape
        # x_g = x_enc.reshape(B*M, D, -1)
        # x_g = self.enc_embedding(x_g)
        # x2 = self.mamba1(x_g) + self.mamba2(x_g.flip(dims=[1])).flip(dims=[1])
        # x2 = self.feed(x2).reshape(B, -1, D)

        # new_x = x1 + x2
        # x = x_enc + new_x
        # x = self.norm(x)

        # gate_conv = self.gate_conv(torch.cat([x1, x2],dim=-1).permute(0,2,1)).permute(0, 2, 1)
        # new_x = torch.sigmoid(gate_conv) * x1 + (1 - torch.sigmoid(gate_conv)) * x2
        # x_p = x_enc + new_x
        # x_p = self.norm(x_p)

        # new_x = self.dropout(self.attention_m(x1, x2, x2, attn_mask=None)[0])
        # x = self.norm(x + new_x)

        # x = x_p.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                 # x: [bs x nvars x patch_num x patch_len]
        # n_vars = x.shape[1]
        # x = self.w_p(x)                                                                      # x: [bs x nvars x patch_num x d_model]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))                # x: [bs * nvars x patch_num x d_model]
        # x = self.mamba3(x)
        # x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))                         # x: [bs x nvars x patch_num x d_model]
        # x = x.permute(0, 1, 3, 2)                                                            # x: [bs x nvars x d_model x patch_num]
        # x = self.head(x)
        # out = x.permute(0, 2, 1)[:, :, :N] 
        # out = self.revin(out,'denorm')

        #Test27
        # batch_x = self.revin(batch_x,'norm')
        # _, _, N = batch_x.shape
        # x_enc = self.enc_embedding_i(batch_x, batch_x_mark)
        # x1 = x_enc + self.dropout(self.attention(self.channel_p, x_enc, x_enc, attn_mask=None)[0])
        # x1 = self.norm(x1)
        # x2 = x_enc + self.dropout(self.patchmamba(x_enc))
        # x2 = self.norm(x2)
        # gate_conv = self.gate_conv(torch.cat([x1, x2], dim=-1).permute(0,2,1)).permute(0, 2, 1)
        # new_x = torch.sigmoid(gate_conv) * x1 + torch.sigmoid(gate_conv) * x2
        # x_p = x_enc + new_x
        # x_p = self.norm(x_p)
        # x = self.mamba1(x_p) + self.mamba2(x_p.flip(dims=[-1])).flip(dims=[-1])
        # x = self.prediction(x)
        # out = x.permute(0, 2, 1)[:, :, :N] 
        # out = self.revin(out,'denorm')

        #Test28
        batch_x = self.revin(batch_x, 'norm')
        _,_,N = batch_x.shape
        x_enc = self.enc_embedding_i(batch_x, batch_x_mark)
        x1 = x_enc + self.dropout(self.attention(self.channel_p, x_enc, x_enc, attn_mask=None)[0])
        x1 = self.norm1(x1)
        x2 = x_enc + self.dropout(self.mamba1(x_enc) + self.mamba2(x_enc.flip(dims=[-1])).flip(dims=[-1]))
        x2 = self.norm(x2)
        x = x1 + x2
        x = x_enc + x
        x = self.norm(x)

        x = self.patch3_2(x)
        out = x.permute(0, 2, 1)[:, :, :N]
        out = self.revin(out, 'denorm')
        return out
    


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

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.cnn = nn.Conv1d(1, hidden_dim, kernel_size, padding = padding)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x_concat = []
        for i in range(self.input_dim):
            x_split = x[:, i, :]
            x_split = x_split.unsqueeze(1)
            x_split = self.cnn(x_split)
            x_split = self.fc(x_split.permute(0, 2, 1))
            x_split = x_split.squeeze(-1)
            x_concat.append(x_split) 
        output = torch.stack(x_concat, dim=1)
        return output

    
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


