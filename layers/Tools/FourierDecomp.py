import torch
import torch.nn as nn
import torch.fft


class FourierProcessor(nn.Module):
    """Fourier Transform Processor - Standalone Module"""

    def __init__(self, seq_len, pred_len, keep_ratio=0.5, mode='low_high'):
        """
        Initialize Fourier Processor

        Args:
            seq_len: Length of input sequence
            pred_len: Length of prediction sequence
            keep_ratio: Ratio of frequency components to retain
            mode: Decomposition mode ('low_high', 'band_pass', 'multi_scale')
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.keep_ratio = keep_ratio
        self.mode = mode

        # Calculate cutoff frequency index
        self.cutoff_index = int(seq_len * keep_ratio)

        # Precompute filters
        self._init_filters()

    def _init_filters(self):
        """Initialize frequency filters"""
        if self.mode == 'low_high':
            # Low-pass and High-pass filters
            self.low_pass_filter = self._create_low_pass_filter()
            self.high_pass_filter = 1 - self.low_pass_filter

        elif self.mode == 'band_pass':
            # Band-pass filters (low + mid + high frequencies)
            self.low_pass_filter = self._create_low_pass_filter(cutoff_ratio=0.3)
            self.mid_pass_filter = self._create_band_pass_filter(low_ratio=0.3, high_ratio=0.7)
            self.high_pass_filter = self._create_high_pass_filter(cutoff_ratio=0.7)

        elif self.mode == 'multi_scale':
            # Multi-scale frequency decomposition
            self.filters = self._create_multi_scale_filters()

    def _create_low_pass_filter(self, cutoff_ratio=None):
        """Create low-pass frequency filter"""
        if cutoff_ratio is None:
            cutoff_ratio = self.keep_ratio

        cutoff_index = int(self.seq_len * cutoff_ratio)
        filter_tensor = torch.zeros(self.seq_len)
        filter_tensor[:cutoff_index] = 1
        filter_tensor[-cutoff_index:] = 1
        return filter_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]

    def _create_high_pass_filter(self, cutoff_ratio=None):
        """Create high-pass frequency filter"""
        low_pass = self._create_low_pass_filter(cutoff_ratio)
        return 1 - low_pass

    def _create_band_pass_filter(self, low_ratio, high_ratio):
        """Create band-pass frequency filter"""
        low_cutoff = int(self.seq_len * low_ratio)
        high_cutoff = int(self.seq_len * high_ratio)

        filter_tensor = torch.zeros(self.seq_len)
        filter_tensor[low_cutoff:high_cutoff] = 1
        filter_tensor[-high_cutoff:-low_cutoff] = 1
        return filter_tensor.unsqueeze(0).unsqueeze(0)

    def _create_multi_scale_filters(self, scales=[0.2, 0.4, 0.6, 0.8]):
        """Create multi-scale filter bank"""
        filters = []
        prev_ratio = 0
        for ratio in scales:
            if prev_ratio == 0:
                # First filter is low-pass
                filters.append(self._create_low_pass_filter(ratio))
            else:
                # Middle filters are band-pass
                filters.append(self._create_band_pass_filter(prev_ratio, ratio))
            prev_ratio = ratio
        # Last filter is high-pass
        filters.append(self._create_high_pass_filter(scales[-1]))
        return filters

    def forward(self, x, return_components=False):
        """
        Forward pass

        Args:
            x: Input tensor [bs, seq_len, nvars]
            return_components: Whether to return all frequency components

        Returns:
            Decomposed signals
        """
        # Fourier transform
        x_permuted = x.permute(0, 2, 1)  # [bs, nvars, seq_len]
        fft_result = torch.fft.fft(x_permuted, dim=2)

        if self.mode == 'low_high':
            return self._decompose_low_high(fft_result, return_components)
        elif self.mode == 'band_pass':
            return self._decompose_band_pass(fft_result, return_components)
        elif self.mode == 'multi_scale':
            return self._decompose_multi_scale(fft_result, return_components)

    def _decompose_low_high(self, fft_result, return_components):
        """Low-frequency and high-frequency decomposition"""
        # Apply filters
        trend_fft = fft_result * self.low_pass_filter.to(fft_result.device)
        residual_fft = fft_result * self.high_pass_filter.to(fft_result.device)

        if return_components:
            return {
                'trend': self.inverse_transform(trend_fft),
                'residual': self.inverse_transform(residual_fft)
            }
        return self.inverse_transform(residual_fft), self.inverse_transform(trend_fft)

    def _decompose_band_pass(self, fft_result, return_components):
        """Band-pass decomposition (low + mid + high frequencies)"""
        low_fft = fft_result * self.low_pass_filter.to(fft_result.device)
        mid_fft = fft_result * self.mid_pass_filter.to(fft_result.device)
        high_fft = fft_result * self.high_pass_filter.to(fft_result.device)

        if return_components:
            return {
                'low_freq': self.inverse_transform(low_fft),
                'mid_freq': self.inverse_transform(mid_fft),
                'high_freq': self.inverse_transform(high_fft)
            }
        # Default: return residual (mid+high) and trend (low)
        residual = self.inverse_transform(mid_fft + high_fft)
        trend = self.inverse_transform(low_fft)
        return residual, trend

    def _decompose_multi_scale(self, fft_result, return_components):
        """Multi-scale frequency decomposition"""
        components = {}
        for i, filter in enumerate(self.filters):
            component_fft = fft_result * filter.to(fft_result.device)
            components[f'component_{i}'] = self.inverse_transform(component_fft)

        if return_components:
            return components

        # Default: lowest frequency as trend, others as residual
        trend = components['component_0']
        residual = sum(components[f'component_{i}'] for i in range(1, len(self.filters)))
        return residual, trend

    def inverse_transform(self, fft_signal):
        """Inverse Fourier transform"""
        reconstructed = torch.fft.ifft(fft_signal, dim=2)
        reconstructed_real = torch.real(reconstructed)
        return reconstructed_real.permute(0, 2, 1)  # [bs, seq_len, nvars]

    def frequency_analysis(self, x):
        """
        Frequency analysis utility function
        Returns: Frequency magnitude and phase information
        """
        x_permuted = x.permute(0, 2, 1)
        fft_result = torch.fft.fft(x_permuted, dim=2)

        # Calculate magnitude and phase
        magnitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)

        return {
            'magnitude': magnitude.permute(0, 2, 1),  # [bs, seq_len, nvars]
            'phase': phase.permute(0, 2, 1),  # [bs, seq_len, nvars]
            'frequencies': torch.fft.fftfreq(self.seq_len)
        }


# Usage Example
if __name__ == "__main__":
    # Test Fourier Processor
    batch_size, seq_len, n_vars = 32, 100, 5
    pred_len = 20

    # Create processor instance
    fourier_processor = FourierProcessor(
        seq_len=seq_len,
        pred_len=pred_len,
        keep_ratio=0.3,
        mode='band_pass'
    )

    # Test data
    test_input = torch.randn(batch_size, seq_len, n_vars)

    # Test decomposition
    residual, trend = fourier_processor(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Residual shape: {residual.shape}")
    print(f"Trend shape: {trend.shape}")

    # Test frequency analysis
    analysis = fourier_processor.frequency_analysis(test_input)
    print(f"Magnitude shape: {analysis['magnitude'].shape}")