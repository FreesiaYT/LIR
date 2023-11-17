import torch
import torch.nn as nn
from utils.registry import ARCH_REGISTRY
from archs.arch_utils import ConvBlock, EAABlock


class LIR(nn.Module):
    def __init__(self, img_channel=3, dim=48, left_blk_num=[3, 3], bottom_blk_num=3, right_blk_num=[4, 3], training=False):
        super().__init__()
        self.training = training
        self.Conv_head = ConvBlock(img_channel, dim)

        self.EAA1 = nn.Sequential(*[EAABlock(dim, 8, 16, training=self.training) for i in range(left_blk_num[0])])
        self.downsample1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

        self.EAA2 = nn.Sequential(*[EAABlock(dim, 8, 8, training=self.training) for i in range(left_blk_num[1])])
        self.downsample2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

        self.EAA3 = nn.Sequential(*[EAABlock(dim, 4, 8, training=self.training) for i in range(bottom_blk_num)])
        self.upsample2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.down = ConvBlock(dim*2, dim)

        self.EAA4 = nn.Sequential(*[EAABlock(dim, 8, 8, training=self.training) for i in range(right_blk_num[1])])
        self.upsample1 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

        self.EAA5 = nn.Sequential(*[EAABlock(dim, 8, 16, training=self.training) for i in range(right_blk_num[0])])
        self.transpose1 = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.transpose2 = nn.ConvTranspose2d(dim, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.Conv_tail = ConvBlock(dim, img_channel)
        
        
    def forward(self, x):
        img = x
        x = self.Conv_head(x)
        res = x

        x = self.EAA1(x)
        lq = self.downsample1(x)

        x = self.EAA2(lq)
        x = self.downsample2(x)

        x = self.EAA3(x)
        x = self.upsample2(x)
        x = self.down(torch.cat([x, lq], dim=1))

        hq = self.EAA4(x)
        x = self.upsample1(hq)
        
        x = self.EAA5(x) + res - self.transpose1(lq-hq)
        x = self.Conv_tail(x) + img - self.transpose2(lq-hq)
        return x 


if __name__ == '__main__':
    from thop import profile
    model = LIR(training=False)
    input = torch.randn(1, 3, 192, 192)
    flops, _ = profile(model, inputs=(input,))
    print('Total params: %.2f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total gflops: %.2f' % (flops / 1e9))

# Total params: 9.38 M
# Total gflops: 107.36