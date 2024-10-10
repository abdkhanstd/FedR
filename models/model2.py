import torch
import torch.nn as nn
from transformers import ViTModel
import torchvision.models as models

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.attention = AttentionModule(out_channels)

    def forward(self, x, skip=None):
        x = self.conv(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.attention(x)
        return x

class RoadSegmenter(nn.Module):
    def __init__(self):
        super(RoadSegmenter, self).__init__()
        
        # ViT as the main feature extractor
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Modified ResNet as the CNN branch
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.cnn_branch = nn.Sequential(
            *list(resnet.children())[:-3],  # Remove the last three layers to keep spatial dimensions
            nn.Conv2d(256, 512, kernel_size=1),  # Adjust channels to match ViT
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks
        self.decoder4 = DecoderBlock(768 + 512, 512)  # 768 from ViT, 512 from modified ResNet
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)
        
        # Final layers
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extract features using ViT
        vit_outputs = self.vit(x)
        if 'last_hidden_state' not in vit_outputs or vit_outputs['last_hidden_state'] is None:
            raise ValueError("ViT model did not return a valid 'last_hidden_state'. Check model output.")
        
        vit_features = vit_outputs['last_hidden_state']
        batch_size, seq_len, hidden_size = vit_features.size()
        
        # Reshape ViT features
        vit_features = vit_features[:, 1:, :].permute(0, 2, 1).contiguous().view(batch_size, hidden_size, 14, 14)
        
        # Extract features using CNN branch
        cnn_features = self.cnn_branch(x)
        
        # Combine ViT and CNN features
        combined_features = torch.cat([vit_features, cnn_features], dim=1)
        
        # Decoder with skip connections
        x = self.decoder4(combined_features)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)
        
        # Final layers
        x = self.final_conv(x)
        segmentation_mask = self.sigmoid(x)
        
        return segmentation_mask

'''
    def forward.old(self, x):
        # Extract features using ViT
        vit_features = self.vit(x)['last_hidden_state']
        batch_size, seq_len, hidden_size = vit_features.size()
        vit_features = vit_features[:, 1:, :].permute(0, 2, 1).contiguous().view(batch_size, hidden_size, 14, 14)
        
        # Extract features using CNN branch
        cnn_features = self.cnn_branch(x)
        
        # Combine ViT and CNN features
        combined_features = torch.cat([vit_features, cnn_features], dim=1)
        
        # Decoder with skip connections
        x = self.decoder4(combined_features)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)
        
        # Final layers
        x = self.final_conv(x)
        segmentation_mask = self.sigmoid(x)
        
        return segmentation_mask
'''
# Usage example (you can keep this for testing, but it's not necessary in your actual model file)
if __name__ == "__main__":
    model = RoadSegmenter()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Should be torch.Size([1, 1, 224, 224])