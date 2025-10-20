import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, config, first=False):
        super().__init__()
        self.config = config
        in_channels = self.config.in_channels if first else self.config.hidden_channels
        self.conv_0 = nn.Conv2d(in_channels=in_channels, 
                                out_channels=self.config.hidden_channels, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1)
        
        self.conv_1 = nn.Conv2d(in_channels=self.config.hidden_channels, 
                                out_channels=self.config.hidden_channels, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, 
                                 stride=2)

        self.bn = nn.BatchNorm2d(num_features=self.config.hidden_channels)


    def forward(self, x):
        x = F.gelu(self.conv_0(x))
        x = F.gelu(self.conv_1(x))

        x = self.pool(x)

        x = self.bn(x)

        return x

######################################
# Modified ConvBlock for concatenated input
######################################
class CatConvBlock(nn.Module):
    """
    The first convolutional block for CNNCatCross.
    Expects concatenated input so that the number of input channels is doubled.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_0 = nn.Conv2d(
            in_channels=self.config.in_channels * 2,  # doubled input channels
            out_channels=self.config.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv_1 = nn.Conv2d(
            in_channels=self.config.hidden_channels,
            out_channels=self.config.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(num_features=self.config.hidden_channels)

    def forward(self, x):
        x = F.gelu(self.conv_0(x))
        x = F.gelu(self.conv_1(x))
        x = self.pool(x)
        x = self.bn(x)
        return x

######################################
# CNNCatCross Model
######################################
class CNNCatCross(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Build backbone:
        # First block uses concatenated input channels.
        self.first_block = CatConvBlock(config)
        # Remaining blocks: a sequence of standard ConvBlocks.
        self.remaining_blocks = nn.Sequential(
            *[ConvBlock(config) for _ in range(self.config.num_layers)]
        )
        
        # After the backbone, feature maps have shape [B, hidden_channels, H_final, W_final].
        # We will split the channels into two halves (make sure hidden_channels is even).
        half_channels = self.config.hidden_channels // 2
        
        # Cross-attention layer: will operate on sequences with embedding dimension = half_channels.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=half_channels,
            num_heads=self.config.attention_heads
        )
        
        self.dropout = nn.Dropout(p=0.3)
        
        # Calculate final spatial dimensions after downsampling:
        # Total blocks = first block + num_layers; each block halves H and W.
        num_blocks = self.config.num_layers + 1
        final_dim = self.config.image_size // (2 ** num_blocks)
        S = final_dim * final_dim  # number of spatial locations
        
        # Fully-connected layer maps the flattened (attended) features to output dimension.
        self.fc = nn.Linear(in_features=S * half_channels, out_features=self.config.output_dim)

    def forward(self, xray_image, drr_image):
        # Concatenate the two images along the channel dimension.
        # Each image is assumed to have shape [B, in_channels, H, W].
        x = torch.cat([xray_image, drr_image], dim=1)  # shape: [B, 2*in_channels, H, W]
        
        # Pass through the backbone.
        x = self.first_block(x)
        x = self.remaining_blocks(x)
        # x now has shape: [B, hidden_channels, H_final, W_final]
        B, C, H, W = x.shape
        
        # Split feature maps along channel dimension into two halves.
        half = C // 2  # ensure that C is even
        feat_first = x[:, :half, :, :]  # assumed to correspond more to one modality
        feat_second = x[:, half:, :, :]  # the other modality
        
        # Reshape each feature map to a sequence for attention.
        # Each becomes [B, half, S] then permuted to [S, B, half].
        S = H * W
        feat_first_seq = feat_first.view(B, half, S).permute(2, 0, 1)
        feat_second_seq = feat_second.view(B, half, S).permute(2, 0, 1)
        
        # Apply cross-attention:
        # 1. Let first features query the second features.
        attn_first, _ = self.cross_attention(
            query=feat_first_seq,
            key=feat_second_seq,
            value=feat_second_seq
        )
        # 2. Let second features query the first features.
        attn_second, _ = self.cross_attention(
            query=feat_second_seq,
            key=feat_first_seq,
            value=feat_first_seq
        )
        # Fuse the two outputs by averaging.
        fused = (attn_first + attn_second) / 2.0  # shape: [S, B, half]
        
        # Permute and flatten: [B, S, half] -> [B, S*half]
        fused = fused.permute(1, 0, 2).contiguous().view(B, -1)
        
        # Apply dropout and final fully connected layer.
        x = self.dropout(fused)
        x = self.fc(x)
        return x