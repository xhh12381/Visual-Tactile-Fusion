from math import sqrt

import einops
import torch
import torch.nn.functional as F
from thop import profile, clever_format
from torch import nn

from deeplabv3plus import DeepLabV3Plus


class CAM(nn.Module):
    """Cross-Attention Module"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # visual qkv
        self.v_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # tactile qkv
        self.t_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.t_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.t_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self._norm_fact = 1 / sqrt(in_channels)

    def forward(self, visual, tactile):
        # visual qkv
        v_q = einops.rearrange(self.v_q(visual), 'b c h w -> b (h w) c')
        v_k_trans = einops.rearrange(self.v_k(visual), 'b c h w -> b c (h w)')
        v_v = einops.rearrange(self.v_v(visual), 'b c h w -> b (h w) c')
        # tactile qkv
        t_q = einops.rearrange(self.t_q(tactile), 'b c h w -> b (h w) c')
        t_k_trans = einops.rearrange(self.t_k(tactile), 'b c h w -> b c (h w)')
        t_v = einops.rearrange(self.t_v(tactile), 'b c h w -> b (h w) c')

        attention_vt = F.softmax(torch.bmm(v_q, t_k_trans), dim=-1) * self._norm_fact  # batch_size * seq_len * seq_len
        out_vt = torch.bmm(attention_vt, t_v)
        out_vt = einops.rearrange(out_vt, 'b (h w) c -> b c h w', h=visual.shape[2], w=visual.shape[3])

        attention_tv = F.softmax(torch.bmm(t_q, v_k_trans), dim=-1) * self._norm_fact  # batch_size * seq_len * seq_len
        out_tv = torch.bmm(attention_tv, v_v)
        out_tv = einops.rearrange(out_tv, 'b (h w) c -> b c h w', h=tactile.shape[2], w=tactile.shape[3])

        return out_vt, out_tv


class VTFNet(nn.Module):
    """Visual-Tactile Fusion Module"""

    def __init__(self, visual_channels=3, tactile_channels=1, num_classes=2):
        super().__init__()
        self.visual_channels = visual_channels
        self.tactile_channels = tactile_channels
        self.num_classes = num_classes

        self.visual_model = DeepLabV3Plus(self.visual_channels)
        self.tactile_model = DeepLabV3Plus(self.tactile_channels)

        self.fusion_module = CAM(in_channels=256)

        self.cat_conv = nn.Conv2d(4 * 256, 256, kernel_size=1, stride=1, padding=0)
        self.cls_conv = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, visual, tactile):
        v_out = self.visual_model(visual)
        t_out = self.tactile_model(tactile)
        out_vt, out_tv = self.fusion_module(v_out, t_out)
        out = self.cat_conv(torch.cat((v_out, t_out, out_vt, out_tv), dim=1))
        out = self.cls_conv(out)
        out = F.interpolate(out, size=visual.size()[2:], mode='nearest')
        return out


if __name__ == '__main__':
    v = torch.randn((8, 3, 256, 256))
    t = torch.randn((8, 1, 256, 256))
    net = VTFNet()
    flops, params = profile(net, inputs=(v, t))
    print(flops / (1000 ** 3))  # G
    print(params / (1000 ** 2))  # M
    print(clever_format([flops, params], '%3.f'))
