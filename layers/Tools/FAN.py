import time
import torch
import torch.nn as nn


def main_freq_part(x, k, rfft=True):
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    k_values = torch.topk(xf.abs(), k, dim=1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    norm_input = x - x_filtered
    return norm_input, x_filtered


def low_pass_filter(x, cutoff_ratio=0.3, rfft=True):
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
        seq_len_rfft = xf.shape[1]
        cutoff_index = int(seq_len_rfft * cutoff_ratio)

        mask = torch.zeros_like(xf)
        mask[:, :cutoff_index, :] = 1

    else:
        xf = torch.fft.fft(x, dim=1)
        seq_len = xf.shape[1]
        cutoff_index = int(seq_len * cutoff_ratio)

        mask = torch.zeros_like(xf)
        mask[:, :cutoff_index, :] = 1
        mask[:, -cutoff_index:, :] = 1

    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1, n=x.shape[1]).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    residual = x - x_filtered
    return residual, x_filtered


def combined_freq_filter(x, k, cutoff_ratio=0.3, rfft=True, mode='weighted'):
    if mode == 'combined':
        return combined_lowpass_then_topk(x, k, cutoff_ratio, rfft)
    elif mode == 'sequential':
        return sequential_combination(x, k, cutoff_ratio, rfft)
    elif mode == 'weighted':
        return weighted_combination(x, k, cutoff_ratio, rfft)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def combined_lowpass_then_topk(x, k, cutoff_ratio=0.3, rfft=True):
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
        seq_len_rfft = xf.shape[1]
        cutoff_index = int(seq_len_rfft * cutoff_ratio)

        # Create a low-pass filter
        lowpass_mask = torch.zeros_like(xf)
        lowpass_mask[:, :cutoff_index, :] = 1
        xf_lowpass = xf * lowpass_mask

    else:
        xf = torch.fft.fft(x, dim=1)
        seq_len = xf.shape[1]
        cutoff_index = int(seq_len * cutoff_ratio)

        lowpass_mask = torch.zeros_like(xf)
        lowpass_mask[:, :cutoff_index, :] = 1
        lowpass_mask[:, -cutoff_index:, :] = 1
        xf_lowpass = xf * lowpass_mask

    k_values = torch.topk(xf_lowpass.abs(), k, dim=1)
    indices = k_values.indices

    topk_mask = torch.zeros_like(xf)
    topk_mask.scatter_(1, indices, 1)

    combined_mask = lowpass_mask * topk_mask
    xf_filtered = xf * combined_mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1, n=x.shape[1]).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    residual = x - x_filtered
    return residual, x_filtered


def sequential_combination(x, k, cutoff_ratio=0.3, rfft=True):

    _, x_topk = main_freq_part(x, k, rfft)

    _, x_lowpass = low_pass_filter(x, cutoff_ratio, rfft)

    x_combined = (x_topk + x_lowpass) / 2 

    residual = x - x_combined
    return residual, x_combined


def weighted_combination(x, k, cutoff_ratio=0.3, rfft=True):

    if rfft:
        xf = torch.fft.rfft(x, dim=1)
        seq_len_rfft = xf.shape[1]
        cutoff_index = int(seq_len_rfft * cutoff_ratio)

        freq_indices = torch.arange(seq_len_rfft, device=x.device).float()
        low_freq_weight = 1.0 - (freq_indices / cutoff_index).clamp(0, 1) 

        weight_matrix = low_freq_weight.unsqueeze(0).unsqueeze(-1).expand_as(xf)

    else:
        xf = torch.fft.fft(x, dim=1)
        seq_len = xf.shape[1]
        cutoff_index = int(seq_len * cutoff_ratio)

        freq_indices = torch.arange(seq_len, device=x.device).float()

        pos_freq_weight = 1.0 - (freq_indices[:cutoff_index] / cutoff_index).clamp(0, 1)

        neg_freq_weight = pos_freq_weight.flip(0)

        low_freq_weight = torch.cat([pos_freq_weight, neg_freq_weight])
        weight_matrix = low_freq_weight.unsqueeze(0).unsqueeze(-1).expand_as(xf)

    weighted_magnitude = xf.abs() * weight_matrix

    k_values = torch.topk(weighted_magnitude, k, dim=1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1, n=x.shape[1]).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    residual = x - x_filtered
    return residual, x_filtered


class FAN(nn.Module):
    """FAN with combined frequency filtering"""

    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20, cutoff_ratio=0.7,
                 rfft=True, filter_mode='combined', **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        self.cutoff_ratio = cutoff_ratio
        self.filter_mode = filter_mode  # 'topk', 'lowpass', 'combined', 'sequential', 'weighted'
        self.rfft = rfft

        print(f"Frequency filtering mode: {filter_mode}")
        if filter_mode in ['topk', 'combined', 'sequential', 'weighted']:
            print(f"freq_topk: {self.freq_topk}")
        if filter_mode in ['lowpass', 'combined', 'sequential', 'weighted']:
            print(f"cutoff_ratio: {self.cutoff_ratio}")

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)

    def _apply_frequency_filter(self, x):
        if self.filter_mode == 'topk':
            return main_freq_part(x, self.freq_topk, self.rfft)
        elif self.filter_mode == 'lowpass':
            return low_pass_filter(x, self.cutoff_ratio, self.rfft)
        elif self.filter_mode in ['combined', 'sequential', 'weighted']:
            return combined_freq_filter(x, self.freq_topk, self.cutoff_ratio, self.rfft, self.filter_mode)
        else:
            raise ValueError(f"Unsupported filter mode: {self.filter_mode}")

    def loss(self, true):
        B, O, N = true.shape
        residual, pred_main = self._apply_frequency_filter(true)

        lf = nn.functional.mse_loss
        return lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual)

    def normalize(self, input):
        bs, len, dim = input.shape
        norm_input, x_filtered = self._apply_frequency_filter(input)
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1, 2), input.transpose(1, 2)).transpose(1, 2)

        return norm_input.reshape(bs, len, dim)

    def denormalize(self, input_norm):
        bs, len, dim = input_norm.shape
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal

        return output.reshape(bs, len, dim)

    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x)


class MLPfreq(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )

        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)