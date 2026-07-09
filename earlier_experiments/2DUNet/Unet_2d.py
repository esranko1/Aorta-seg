import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.decoder = UNetDecoder(out_channels, features)

        for feature in features:
            self.encoder_blocks.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        x = self.decoder(x, encoder_outputs)
        return x

class UNetDecoder(nn.Module):
    def __init__(self, out_channels, features=[64, 128, 256, 512]):
        super(UNetDecoder, self).__init__()
        self.decoder_upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        for feature in reversed(features):
            # upsamples spatially and halves channels
            self.decoder_upsample.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            # concatenates with encoder output, so doubles channels
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))
    
    def forward(self, x, encoder_outputs):
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_upsample[i](x)
            enc_out = encoder_outputs[-(i + 1)]
            x = torch.cat((x, enc_out), dim=1)
            x = self.decoder_blocks[i](x)
        return self.final_conv(x)



