import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x) 

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32,64,128,256]):
        super(UNet3D, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder_blocks.append(DoubleConv3D(in_channels, feature))
            in_channels = feature
        
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        self.decoder_upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for feature in reversed(features):
            self.decoder_upsample.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder_blocks.append(DoubleConv3D(feature * 2, feature))
        
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        for i in range(len(self.decoder_blocks)):
            x = self.decoder_upsample[i](x)
            x = torch.cat((x, encoder_outputs[-(i + 1)]), dim=1)
            x = self.decoder_blocks[i](x)
        
        return self.final_conv(x)
