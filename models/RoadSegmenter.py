import torch
import torch.nn as nn
from transformers import ViTModel
import torch.nn.functional as F


class FeatureFusionModule(nn.Module):
    def __init__(self, vit_channels, cnn_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv_vit = nn.Conv2d(vit_channels, 256, kernel_size=1)
        self.conv_cnn = nn.Conv2d(cnn_channels, 256, kernel_size=1)
        self.conv_out = nn.Conv2d(512, 768, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, vit_features, cnn_features):
        vit_features = self.conv_vit(vit_features)
        cnn_features = self.conv_cnn(cnn_features)
        combined = torch.cat([vit_features, cnn_features], dim=1)
        out = self.conv_out(combined)
        return self.relu(out)

class FeatureAggregationModule(nn.Module):
    """This module will aggregate features from different decoder branches."""
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeatureAggregationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        # Concatenate along the channel dimension and process
        x = torch.cat((x1, x2), dim=1)  # Concatenation along channels
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
    
    
class RoadSegmenter(nn.Module):
    def __init__(self):
        super(RoadSegmenter, self).__init__()
        
        # Pretrained ViT as the feature extractor
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        # CNN branch to complement ViT
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Downsample from 224x224 to 112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample from 112x112 to 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample from 56x56 to 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Downsample from 28x28 to 14x14
            nn.ReLU(inplace=True)
        )
        
        # MLP branch to complement CNN and ViT
        self.mlp_branch = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 768),  # Match the hidden size of ViT
            nn.ReLU(inplace=True)
        )

        # Decoder branch 1 (high-resolution branch)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2),  # Upsample from 14x14 to 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder branch 2 (mid-resolution branch)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),  # Upsample from 14x14 to 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder branch 3 (low-resolution branch)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(768, 128, kernel_size=2, stride=2),  # Upsample from 14x14 to 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Inter-branch communication (aggregation at multiple stages)
        self.aggregation1 = FeatureAggregationModule(256, 128, 128)  # Combine decoder1 and decoder2
        self.aggregation2 = FeatureAggregationModule(128, 64, 64)    # Combine aggregate1 with decoder3
        self.aggregation3 = FeatureAggregationModule(256, 64, 128)   # Corrected channels: 256 (from decoder1) + 64 (from aggregate2)
        self.feature_fusion = FeatureFusionModule(768, 512)

        # Final decoder to upsample to original input size (224x224)
        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample from 28x28 to 56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # Upsample from 56x56 to 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # Upsample from 112x112 to 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),  # Output channel for segmentation mask
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features using ViT
        vit_features = self.vit(x)['last_hidden_state']  # Shape: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = vit_features.size()

        # Remove [CLS] token and reshape
        vit_features = vit_features[:, 1:, :]  # Shape: [batch_size, seq_len-1, hidden_size]
        height = width = int((seq_len - 1) ** 0.5)  # Typically 14x14 for ViT-base
        vit_features = vit_features.permute(0, 2, 1).contiguous().view(batch_size, hidden_size, height, width)

        # Pass through CNN branch
        cnn_features = self.cnn_branch(x)  # Shape: [batch_size, 512, 14, 14]

        # Flatten CNN output and pass through MLP branch
        cnn_features_flattened = cnn_features.view(batch_size, -1)  # Flatten the CNN output
        mlp_features = self.mlp_branch(cnn_features_flattened)  # Shape: [batch_size, 768]
        mlp_features = mlp_features.view(batch_size, 768, 1, 1)  # Reshape to match ViT feature dimensions

        # Combine ViT and MLP features
        combined_features = self.feature_fusion(vit_features, cnn_features)
        #combined_features = vit_features + mlp_features  # Element-wise addition of ViT and MLP features

        # Branch 1 - higher resolution
        decoder1_output = self.decoder1(combined_features)
        
        # Branch 2 - mid resolution
        decoder2_output = self.decoder2(combined_features)

        # Branch 3 - lower resolution
        decoder3_output = self.decoder3(combined_features)

        # Aggregation of features from different branches
        aggregate1 = self.aggregation1(decoder1_output, decoder2_output)
        aggregate2 = self.aggregation2(aggregate1, decoder3_output)
        final_aggregate = self.aggregation3(decoder1_output, aggregate2)

        # Final decoder to generate segmentation mask
        segmentation_mask = self.final_decoder(final_aggregate)

        return segmentation_mask
