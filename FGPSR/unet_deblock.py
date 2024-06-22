import torch
from torch import nn


class unet_block(nn.Module):
    def __init__(self):
        super(unet_block, self).__init__()
        c = 16
        self.prelu = nn.PReLU()
        self.down1 = nn.Sequential(
            nn.Conv2d(c, c, 2, 2, 0),
            nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c, c, 2, 2, 0),
            nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(c, c, 2, 2, 0),
            nn.PReLU()
        )
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.up1 = nn.Sequential(
            nn.Conv2d(c*2, c*4, 3, 1, 1),
            nn.PReLU(),
            nn.PixelShuffle(2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c*2, c*4, 3, 1, 1),
            nn.PReLU(),
            nn.PixelShuffle(2),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(c*2, c*4, 3, 1, 1),
            nn.PReLU(),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down2(x2)
        out=self.conv1(x3)
        out=self.up1(torch.cat((out,x3),1))
        out = self.up2(torch.cat((out, x2), 1))
        out = self.up3(torch.cat((out, x1), 1))
        return out


class deblock_unet(nn.Module):
    def __init__(self):
        super(deblock_unet, self).__init__()
        self.rec_block1 = unet_block()
        # self.rec_block2 = unet_block()
        # self.rec_block3 = res_block()
        # self.rec_block4 = res_block()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 3, 3, 1, 1)
        self.prelu = nn.PReLU()

    def segment1(self, x):
        x = self.rec_block1(x)
        x = self.rec_block2(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        resdual = x
        # x.requires_grad_(True)
        x = self.rec_block1(x)
        # x = self.rec_block2(x)
        # x = self.rec_block3(x)
        # x = self.rec_block4(x)
        x = self.conv2(x + resdual)
        return x
