import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = BasicBlock(3, 32)                      #--- input: [B, 3, 32, 32] -> output: [B, 32, 32, 32]
        self.block2 = BasicBlock(32, 64, downsample=True)    #--- input: [B, 32, 32, 32] -> output: [B, 64, 16, 16]
        self.block3 = BasicBlock(64, 128, downsample=True)   #--- input: [B, 64, 16, 16] -> output: [B, 128, 8, 8]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))             #--- input: [B, 128, 8, 8] -> output: [B, 128, 1, 1]
        self.dropout = nn.Dropout(0.3)                       #--- input: [B, 128] (after flatten)
        self.fc = nn.Linear(128, 10)                         #--- input: [B, 128] -> output: [B, 10]

    def forward(self, x):
        x = self.block1(x)                                   #--- [B, 32, 32, 32]
        x = self.block2(x)                                   #--- [B, 64, 16, 16]
        x = self.block3(x)                                   #--- [B, 128, 8, 8]
        x = self.pool(x).view(x.size(0), -1)                 #--- [B, 128, 1, 1] -> [B, 128]
        x = self.dropout(x)                                  #--- [B, 128]
        return self.fc(x)                                    #--- [B, 10] -> logits per class
