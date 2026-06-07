import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Zastępuje zwykłe DoubleConv. Uczy się 'różnicy' (residual), 
    co pozwala sieci lepiej skupić się na krawędziach obiektów.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # Jeśli liczba kanałów wejściowych różni się od wyjściowych, musimy dopasować wymiary
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.shortcut(x))

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # ENCODER (Feature Extraction)
        self.down1 = ResidualBlock(in_channels, 64)
        self.down2 = ResidualBlock(64, 128)
        self.down3 = ResidualBlock(128, 256)
        self.down4 = ResidualBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # BOTTLENECK
        self.bottleneck = ResidualBlock(512, 1024)
        
        # DECODER z Upsamplingiem i Skip Connections
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = ResidualBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ResidualBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ResidualBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ResidualBlock(128, 64)
        
        # FINAL PREDICTION
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Zejście w dół (zapisywanie skip connections)
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        
        # Wąskie gardło
        b = self.bottleneck(self.pool(d4))
        
        # Wyjście w górę (łączenie cech z Encodera)
        u4 = self.up4(b)
        u4 = torch.cat((d4, u4), dim=1)
        u4 = self.conv4(u4)
        
        u3 = self.up3(u4)
        u3 = torch.cat((d3, u3), dim=1)
        u3 = self.conv3(u3)
        
        u2 = self.up2(u3)
        u2 = torch.cat((d2, u2), dim=1)
        u2 = self.conv2(u2)
        
        u1 = self.up1(u2)
        u1 = torch.cat((d1, u1), dim=1)
        u1 = self.conv1(u1)
        
        return self.final_conv(u1)