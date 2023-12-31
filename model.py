import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.model(x)
    
class UNet(nn.Module):
    def __init__(
        self, 
        in_channels = 3, 
        out_channels = 11, 
        features = [64, 128, 256, 512]
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size = 2,
                    stride = 2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def get_in_channels(self):
        return self.in_channels
    
    def get_out_channels(self):
        return self.out_channels

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = F.resize(x, size = skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[i + 1](concat_skip)
        return self.final(x)