import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeAttention(nn.Module):
    """
    Temporal Attention Module
    Input shape: [batch_size, time_steps, channels] or [batch_size, channels, time_steps]
    Output shape: Same as input
    """

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3, mode='channels_last'):
        """
        Args:
            in_channels: Number of input feature channels
            reduction_ratio: Compression ratio for fully connected layers
            kernel_size: Size of 1D convolution kernel
            mode: 'channels_last' [B, T, C] or 'channels_first' [B, C, T]
        """
        super(TimeAttention, self).__init__()
        self.mode = mode

        # Method 1: 1D Convolution
        self.conv = nn.Conv1d(
            in_channels=2,  # Concatenation of avg pool and max pool
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        # Method 2: Small Fully Connected Network (Alternative)
        self.fc_net = nn.Sequential(
            nn.Linear(2, 8),  # 2 features -> 8 hidden units
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),  # 8 hidden units -> 1 output
        )

        # Flag to use convolution or fully connected network
        self.use_conv = True

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor, shape [B, T, C] or [B, C, T]
        """
        batch_size, time_steps, channels = 0, 0, 0

        # Handle different input formats
        if self.mode == 'channels_last':
            # Input shape: [B, T, C]
            batch_size, time_steps, channels = x.size()
            x_permuted = x.permute(0, 2, 1)  # [B, C, T] for pooling
        else:
            # Input shape: [B, C, T]
            batch_size, channels, time_steps = x.size()
            x_permuted = x

        # Step 1: Apply average pooling and max pooling along channel dimension
        # Input: [B, C, T], pool along dim=1 (channels) -> [B, 1, T]
        avg_pool = torch.mean(x_permuted, dim=1, keepdim=True)  # [B, 1, T]
        max_pool, _ = torch.max(x_permuted, dim=1, keepdim=True)  # [B, 1, T]

        # Step 2: Concatenate pooled features -> [B, 2, T]
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, T]

        # Step 3: Generate attention weights via 1D conv or FC network
        if self.use_conv:
            # Use 1D convolution: [B, 2, T] -> [B, 1, T]
            attention_weights = self.conv(pooled)  # [B, 1, T]
        else:
            # Use fully connected network
            # Reshape: [B, 2, T] -> [B, T, 2] -> [B, T, 1] -> [B, 1, T]
            attention_weights = self.fc_net(pooled.permute(0, 2, 1))  # [B, T, 1]
            attention_weights = attention_weights.permute(0, 2, 1)  # [B, 1, T]

        # Step 4: Apply Sigmoid activation
        attention_weights = self.sigmoid(attention_weights)  # [B, 1, T]

        # Step 5: Reshape attention weights for broadcast multiplication
        if self.mode == 'channels_last':
            # [B, 1, T] -> [B, T, 1] to match [B, T, C]
            attention_weights = attention_weights.permute(0, 2, 1)  # [B, T, 1]
            output = x * attention_weights  # Broadcast multiply: [B, T, C] * [B, T, 1]
        else:
            # [B, 1, T] directly multiplies with [B, C, T]
            output = x_permuted * attention_weights  # Broadcast multiply: [B, C, T] * [B, 1, T]

        return output