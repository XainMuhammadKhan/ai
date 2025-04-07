import torch
import torch.nn as nn
import torch.nn.functional as F

class HDRAttention(nn.Module):
    def __init__(self, in_channels):
        super(HDRAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling
        avg_pool = F.avg_pool2d(x, x.size()[2:])
        # Channel attention
        channel_att = self.conv1(avg_pool)
        channel_att = F.relu(channel_att)
        channel_att = self.conv2(channel_att)
        channel_att = self.sigmoid(channel_att)
        # Apply attention
        return x * channel_att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Max and average pooling along channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Concatenate
        pool_cat = torch.cat([max_pool, avg_pool], dim=1)
        # Apply convolution
        spatial_att = self.conv(pool_cat)
        spatial_att = self.sigmoid(spatial_att)
        # Apply attention
        return x * spatial_att

class HDRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HDRBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        # HDR-specific attention
        self.channel_att = HDRAttention(out_channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention
        out = self.channel_att(out)
        out = self.spatial_att(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ImageQualityModel(nn.Module):
    def __init__(self):
        super(ImageQualityModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Quality score prediction
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Attention
        attention = self.attention(features)
        features = features * attention
        
        # Quality score
        quality_score = self.quality_head(features)
        return quality_score 