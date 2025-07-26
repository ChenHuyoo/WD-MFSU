"""
the final implement of Multi-Scale Fusion and Decomposition Network for Single Image Deraining
"""
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
# from restormer_block import *
# from Deraining_our import TransformerBlock
import torch.nn.functional as F
from einops.layers.torch import Rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):  # 不改变size的conv
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def st_conv(in_channels, out_channels, kernel_size, bias=False, stride=2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


##########################################################################
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
# S2FB
class S2FB_2(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_2, self).__init__()
        self.DSC = depthwise_separable_conv(n_feat * 2, n_feat)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        # self.hag_fea = HAG(n_feat)

    def forward(self, x1, x2):
        FEA_1 = self.DSC(torch.cat((x1, x2), 1))
        res = self.CA_fea(FEA_1) + x1
        return res

    ##########################################################################


# Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias),
                        act,
                        conv(n_feat, n_feat, kernel_size, bias=bias)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


# Enhanced Channel Attention Block with DSC (ECAB with dsc)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = [depthwise_separable_conv(n_feat, n_feat),
                        act,
                        depthwise_separable_conv(n_feat, n_feat)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        # self.hag = HAG(n_feat)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        # res = self.hag(res)
        res = self.S2FB2(res, x)
        # res += x
        return res


##########################################################################
# ASF-Net: Adaptive Screening Feature Network for Building Footprint Extraction From Remote-Sensing Images
class cSE(nn.Module):
    def __init__(self, input_channels):
        super(cSE, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // 4)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(input_channels // 4, input_channels)
        self.hard_swish = nn.Hardswish()

    def forward(self, inputs):
        x = self.global_avg_pool(inputs).view(inputs.size(0), -1)  # Global Average Pooling
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.hard_swish(x)
        x = x.view(inputs.size(0), inputs.size(1), 1, 1)  # Reshape for broadcasting
        x = inputs * x  # Channel-wise multiplication
        return x


class sSE(nn.Module):
    def __init__(self, input_channels):
        super(sSE, self).__init__()
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.relu = nn.PReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.relu(x)
        return x


class csSE(nn.Module):
    def __init__(self, input_channels):
        super(csSE, self).__init__()
        self.cse = cSE(input_channels)
        self.sse = sSE(input_channels)

    def forward(self, inputs):
        c = self.cse(inputs)
        s = self.sse(inputs)
        return c + s  # Element-wise addition


##########################################################################
# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),  # // : 整除,向下取整
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y


##########################################################################
# Long Feature Selection and Fusion Block (LFSFB)
class LFSFB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(LFSFB, self).__init__()
        self.FS = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=False)
        self.act1 = act
        self.FFU = nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.act2 = act

    def forward(self, x1, x2):
        res = self.act1(self.FS(x1))
        res_out = self.act2(self.FFU(x2 + res))
        return res_out


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class PatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(PatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.shallow_fea_B = depthwise_separable_conv(embed_dim, embed_dim)
        self.shallow_fea_R = depthwise_separable_conv(embed_dim, embed_dim)

    def forward(self, x):
        x_fea = self.proj(x)
        b_fea = self.shallow_fea_B(x_fea)
        r_fea = self.shallow_fea_R(x_fea)

        return [b_fea, r_fea]


##########################################################################
# Resizing modules
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor):
        super(DownSample, self).__init__()
        self.down_B = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                    nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        self.down_R = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                    nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x_B, x_R):
        return [self.down_B(x_B), self.down_R(x_R)]


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor):
        super(UpSample, self).__init__()
        self.up_B = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.up_R = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x_B, x_R):
        return [self.up_B(x_B), self.up_R(x_R)]


##########################################################################
# Reconstruction and Reproduction Block (RRB)
class RRB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(RRB, self).__init__()
        self.recon_B = conv(n_feat, 3, kernel_size, bias=bias)
        self.recon_R = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x):
        xB = x[0]
        xR = x[1]
        recon_B = self.recon_B(xB)
        recon_R = self.recon_R(xR)
        re_rain = recon_B + recon_R
        return [recon_B, re_rain, recon_R]


# Coupled Representation Block (CRB)
# class CRB(nn.Module):
#     def __init__(self, n_feat):
#         super(CRB, self).__init__()
#
#         # 设置可学习参数
#         self.fuse_weight_BTOR = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#         self.fuse_weight_RTOB = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#         # 初始化
#         self.fuse_weight_BTOR.data.fill_(0.2)
#         self.fuse_weight_RTOB.data.fill_(0.2)
#
#         self.conv_fuse_BTOR = nn.Sequential(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=False), nn.Sigmoid())
#         self.conv_fuse_RTOB = nn.Sequential(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=False), nn.Sigmoid())
#
#     def forward(self, xB_res, xR_res):
#         res_BTOR = xB_res * self.conv_fuse_BTOR(xR_res) * self.fuse_weight_BTOR
#         res_RTOB = xR_res * self.conv_fuse_RTOB(xB_res) * self.fuse_weight_RTOB
#
#         xb = xB_res - res_BTOR + res_RTOB
#         xr = xR_res - res_RTOB + res_BTOR
#
#         return [xb, xr]
# return [xb, xr, res_BTOR, res_RTOB]
#######################################################################
class CondConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_experts):
        super(CondConvLayer, self).__init__()
        self.num_experts = num_experts
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define expert weights
        self.weight = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels, kernel_size, kernel_size)
        )

        # Condition weight generation network
        self.routing = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        batch_size = x.size(0)
        routing_weights = self.routing(x.mean(dim=(2, 3)))  # Generate routing weights based on input
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Compute the weighted sum of expert kernels for each sample in the batch
        combined_weights = torch.einsum('be, eocij -> bocij', routing_weights, self.weight)

        # Reshape for group convolution
        combined_weights = combined_weights.view(batch_size * self.out_channels, self.in_channels, self.kernel_size,
                                                 self.kernel_size)

        # Expand input to match the group convolution
        x = x.view(1, batch_size * self.in_channels, x.size(2), x.size(3))

        # Perform the grouped convolution
        out = F.conv2d(x, combined_weights, groups=batch_size)

        # Reshape the output back to (batch_size, out_channels, height, width)
        out = out.view(batch_size, self.out_channels, out.size(2), out.size(3))

        return out


class EnhancedCRBWithCondConv(nn.Module):
    def __init__(self, n_feat):
        super(EnhancedCRBWithCondConv, self).__init__()

        self.cond_conv_BTOR = CondConvLayer(n_feat, n_feat, kernel_size=1, num_experts=4)
        self.cond_conv_RTOB = CondConvLayer(n_feat, n_feat, kernel_size=1, num_experts=4)

    def forward(self, xB_res, xR_res):
        # Dynamic convolution based on input
        res_BTOR = xB_res * self.cond_conv_BTOR(xR_res)
        res_RTOB = xR_res * self.cond_conv_RTOB(xB_res)

        xb = xB_res - res_BTOR + res_RTOB
        xr = xR_res - res_RTOB + res_BTOR

        return [xb, xr]


#################################################

def get_kernel_gussian(kernel_size, Sigma=1, in_channels=64):
    kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=Sigma)
    kernel_weights = kernel_weights * kernel_weights.T
    kernel_weights = np.repeat(kernel_weights[None, ...], in_channels, axis=0)[:, None, ...]

    return kernel_weights


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class eca_layer(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MaskPredictor(nn.Module):
    def __init__(self, in_channels, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(MaskPredictor, self).__init__()
        self.spatial_mask = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, bias=False)

    def forward(self, x):
        spa_mask = self.spatial_mask(x)
        spa_mask = F.gumbel_softmax(spa_mask, tau=1, hard=True, dim=1)
        return spa_mask


class HAG(nn.Module):
    def __init__(self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(HAG, self).__init__()
        self.CA = eca_layer(n_feats, k_size=3)
        self.MaskPredictor = MaskPredictor(n_feats * 8 // 8)

        self.k = nn.Sequential(
            wn(nn.Conv2d(n_feats * 8 // 8, n_feats * 8 // 8, kernel_size=3, padding=1, stride=1, groups=1)),
            nn.LeakyReLU(0.05),
        )

        self.k1 = nn.Sequential(
            wn(nn.Conv2d(n_feats * 8 // 8, n_feats * 8 // 8, kernel_size=3, padding=1, stride=1, groups=1)),
            nn.LeakyReLU(0.05),
        )

        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        res = x
        x = self.k(x)

        MaskPredictor = self.MaskPredictor(x)
        mask = (MaskPredictor[:, 1, ...]).unsqueeze(1)
        x = x * (mask.expand_as(x))

        x1 = self.k1(x)
        x2 = self.CA(x1)
        out = self.x_scale(x2) + self.res_scale(res)

        return out


###############3DCONV

class HighFreqEnhancement(nn.Module):
    def __init__(self, channels, L_num):
        super(HighFreqEnhancement, self).__init__()

        self.hf_agg = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), bias=False,
                                groups=channels
                                )
        self.L_num = L_num
        self.channels = channels
        # Gaussian Kernel
        ### parameters
        kernet_shapes = [3, 5]
        s_value = np.power(2, 1 / 3)
        sigma = 1.6

        ### Kernel weights for Laplacian pyramid
        if self.L_num > 1:
            self.sigma1_kernel = get_kernel_gussian(kernel_size=kernet_shapes[0], Sigma=sigma * np.power(s_value, 1),
                                                    in_channels=channels)
            self.sigma1_kernel = torch.from_numpy(self.sigma1_kernel).float().to(device)

        if self.L_num > 2:
            self.sigma2_kernel = get_kernel_gussian(kernel_size=kernet_shapes[1], Sigma=sigma * np.power(s_value, 2),
                                                    in_channels=channels)
            self.sigma2_kernel = torch.from_numpy(self.sigma2_kernel).float().to(device)

        self.detail_agg = nn.Conv3d(channels, channels, kernel_size=(self.L_num, 1, 1), bias=False,
                                    groups=channels)
        self.feature_agg = nn.Conv2d(channels, channels * 3, kernel_size=1)

    def forward(self, high_freq):
        high_ex = Rearrange('b (p c) h w -> b c p h w', p=3)(high_freq)

        # high_ex = Rearrange('b (p c) h w -> b c p h w', p=3)(high_freq)

        hf_agg = self.hf_agg(high_ex)[:, :, 0, ...]

        G0 = hf_agg
        L0 = G0[:, :, None, ...]  # Level 1

        L_layers = [L0, ]

        # 获取卷积核大小
        # kernel_height, kernel_width = self.sigma1_kernel.size(2), self.sigma1_kernel.size(3)
        # 计算对称填充以实现 same 效果
        padding_size1 = (self.sigma1_kernel.size(2) // 2, self.sigma1_kernel.size(3) // 2)
        padding_size2 = (self.sigma2_kernel.size(2) // 2, self.sigma2_kernel.size(3) // 2)

        if self.L_num > 1:
            G1 = F.conv2d(input=hf_agg, weight=self.sigma1_kernel, bias=None, padding=padding_size1,
                          groups=self.channels)
            L1 = torch.sub(G0, G1)[:, :, None, ...]  # Level 2
            L_layers += [L1]
        if self.L_num > 2:
            G2 = F.conv2d(input=hf_agg, weight=self.sigma2_kernel, bias=None, padding=padding_size2,
                          groups=self.channels)
            L2 = torch.sub(G1, G2)[:, :, None, ...]  # Level 3
            L_layers += [L2]

        lvl_cat = torch.cat(L_layers, dim=2)

        detail_agg = self.detail_agg(lvl_cat)[:, :, 0, ...]

        aggregated_features = self.feature_agg(detail_agg)

        # print(high_freq.shape,aggregated_features.shape)

        return aggregated_features


class LowFreqEnhancement(nn.Module):
    def __init__(self, channels):
        super(LowFreqEnhancement, self).__init__()
        self.eca = eca_layer(channels, 3)

    def forward(self, low_freq):
        # Enhance low-frequency features using ECA
        enhanced_low_freq = self.eca(low_freq)
        return enhanced_low_freq


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    # return [x_LL, x_HL, x_LH, x_HH]#torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1), [x_LL, x_HL, x_LH, x_HH]


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width]) #[1, 12, 56, 56]
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    # print(out_batch, out_channel, out_height, out_width) #1 3 112 112
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    # print(x1.shape) #torch.Size([1, 3, 56, 56])
    # print(x2.shape) #torch.Size([1, 3, 56, 56])
    # print(x3.shape) #torch.Size([1, 3, 56, 56])
    # print(x4.shape) #torch.Size([1, 3, 56, 56])
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


#
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  #

    def forward(self, x):
        return dwt_init(x)


#
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


#
# class w_att(nn.Module):
#     def __init__(self, channels, num_heads):
#         super(w_att, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
#
#         self.qkv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
#         # self.query = SSL(channels)
#         # self.conv_query = nn.Conv2d(in_channels=channels, out_channels=channels//2,
#         #                             kernel_size=1)
#         self.dwt = DWT()
#         self.iwt = IWT()
#         self.qkv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
#         self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.low_freq_module = LowFreqEnhancement(channels)
#         self.high_freq_module = HighFreqEnhancement(channels, 3)
#
#     def forward(self, x):
#         b, c, h, w = x.shape # 16 128 128
#
#         # q = (self.iwt(self.dwt(x)[0]))
#
#         low_freq, high_freq = self.dwt(x)[1][0], torch.cat(self.dwt(x)[1][1:], 1)  # 16 64 64 , 48 64 64
#         processed_low_freq = self.low_freq_module(low_freq)
#         processed_high_freq = self.high_freq_module(high_freq)
#
#         # 逆小波变换，合并处理后的低频和高频信息
#         restored = self.iwt(torch.cat([processed_low_freq, processed_high_freq], 1)) # 16 128 128
#
#
#         k, v = self.qkv_conv(self.qkv(x)).chunk(2, dim=1) # 16 128 128
#
#         q = restored  # 使用处理后的恢复图像计算 Q
#
#         q = q.reshape(b, self.num_heads, -1, h * w)
#         k = k.reshape(b, self.num_heads, -1, h * w)
#         v = v.reshape(b, self.num_heads, -1, h * w)
#         q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
#
#         attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
#         out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
#
#         return out

class w_att(nn.Module):
    def __init__(self, channels, num_heads):
        super(w_att, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        # self.query = SSL(channels)
        # self.conv_query = nn.Conv2d(in_channels=channels, out_channels=channels//2,
        #                             kernel_size=1)
        self.dwt = DWT()
        self.iwt = IWT()
        self.qkv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.low_freq_module = LowFreqEnhancement(channels)
        self.high_freq_module = HighFreqEnhancement(channels, 3)
        self.conv1 = nn.Sequential(nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1, groups=channels*4, bias=False)
                                   ,nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1, groups=channels*4, bias=False),
                                   # nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1, groups=channels * 4,
                                   #           bias=False)
                                   )
    def forward(self, x):
        b, c, h, w = x.shape  # 16 128 128

        # q = (self.iwt(self.conv1(self.dwt(x)[0])))
        q = (self.iwt(self.dwt(x)[0]))

        low_freq, high_freq = self.dwt(x)[1][0], torch.cat(self.dwt(x)[1][1:], 1)  # 16 64 64 , 48 64 64
        processed_low_freq = self.low_freq_module(low_freq)
        processed_high_freq = self.high_freq_module(high_freq)
        #
        # # 逆小波变换，合并处理后的低频和高频信息
        restored = self.iwt(torch.cat([processed_low_freq, processed_high_freq], 1)) # 16 128 128

        k, v = self.qkv_conv(self.qkv(x)).chunk(2, dim=1)  # 16 128 128

        q = restored  # 使用处理后的恢复图像计算 Q

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        return out


class FD(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(FD, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        #
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)

        return x


# class TransformerBlock(nn.Module):
#     def __init__(self, channels, num_heads, expansion_factor):
#         super(TransformerBlock, self).__init__()
#
#         self.norm1 = nn.LayerNorm(channels)
#         # self.attn = MDTA(channels, num_heads)
#         self.norm2 = nn.LayerNorm(channels)
#         self.ffn = FD(channels, expansion_factor)
#         # self.ffn = FeedForward(channels,expansion_factor,False)
#         #
#         # self.watt = WindowEfficientSelfAttention(in_channels=channels, key_channels_redution=2, num_heads=num_heads,
#         #                                          window_size=16, with_pos=True)
#         self.watt = w_att(channels, num_heads)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
#         #                   .contiguous().reshape(b, c, h, w))
#         x = x + self.watt(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
#                           .contiguous().reshape(b, c, h, w))
#
#         x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
#                          .contiguous().reshape(b, c, h, w))
#
#         return x
from restormer_block import RestormerBlock
class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        # self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = FD(channels, expansion_factor)
        # self.ffn = FeedForward(channels,expansion_factor,False)
        #
        # self.watt = WindowEfficientSelfAttention(in_channels=channels, key_channels_redution=2, num_heads=num_heads,
        #                                          window_size=16, with_pos=True)
        self.watt = w_att(channels, num_heads)

        self.dwt = DWT()
        self.iwt = IWT()
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.low_freq_module = LowFreqEnhancement(channels)
        self.high_freq_module = HighFreqEnhancement(channels, 3)
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        # self.fuse = Fuse(channels)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x= x[0]
        b, c, h, w = x.shape

        x_ = x
        # x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
        #                   .contiguous().reshape(b, c, h, w))
        x = x + self.watt(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))

        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))

        low_freq, high_freq = self.dwt(x_)[1][0], torch.cat(self.dwt(x_)[1][1:], 1)  # 16 64 64 , 48 64 64
        processed_low_freq = self.low_freq_module(low_freq)
        processed_high_freq = self.high_freq_module(high_freq)

        # 逆小波变换，合并处理后的低频和高频信息
        restored = self.iwt(torch.cat([processed_low_freq, processed_high_freq], 1))  # 16 128 128

        # fuse_fea = self.fuse(x,restored)

        fuse_fea = self.conv(torch.cat([restored, x], 1))

        return fuse_fea

class Fuse(nn.Module):  ### CWFA -> Fusion Block ###    #  guide  &&  wavelet subband

    def __init__(self, channels=32):
        super(Fuse, self).__init__()

        self.rcg_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.wave_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.rcg1_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.wave1_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.se = SELayer(channel=channels, reduction=1)
        self.image_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=True))  # 64->32

    def forward(self, x, sub):
        a = self.rcg_conv(x)
        b = self.wave_conv(sub)
        energy = a * b
        se = self.se(energy)

        rcg_gamma = self.rcg1_conv(se)
        rcg_out = x + rcg_gamma

        wave_gamma = self.wave1_conv(se)
        wave_out = sub + wave_gamma
        fuse_cat = torch.cat((rcg_out, wave_out), dim=1)
        fuse = self.image_conv(fuse_cat)

        return fuse


# FEB + CRB
class TAD(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_rb):
        super(TAD, self).__init__()

        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  # Other option 'BiasFree'
        num_RB = num_rb  # number of Restormer Blocks
        # self.CAB_r = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        # self.CAB_b = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.CAB_r = csSE(n_feat)
        self.CAB_b = csSE(n_feat)
        self.down_R = st_conv(n_feat, n_feat, kernel_size, bias=bias)
        self.act1 = act
        modules_body = [
            TransformerBlock(n_feat, heads, ffn_expansion_factor,
                             ) for _ in range(num_RB)]
        self.body = nn.Sequential(*modules_body)

        self.lfsfb = LFSFB(n_feat, kernel_size, act, bias)
        self.CRB = EnhancedCRBWithCondConv(n_feat)

    def forward(self, x):
        xB = x[0]
        xR = x[1]
        res_down_R = self.act1(self.down_R(xR))
        res_R= self.body(res_down_R)
        # CAB CRB改
        xR_res =  self.lfsfb(res_down_R, res_R)

        # xB_res = self.CAB_b(xB)

        x = [xB, xR_res]


        return x


##########################################################################
class MODEL(nn.Module):
    def  __init__(self, in_c, out_c, n_feat, kernel_size, reduction, act, bias, num_tb):
        super(MODEL, self).__init__()

        # embedding
        self.patch_embed = PatchEmbed(in_c, n_feat)

        self.down0_1 = DownSample(n_feat, n_feat * 2, 0.5)  # channel: 48, 96，192，384 # 2C H/2 W/2
        self.down0_2 = DownSample(n_feat, n_feat * 4, 0.25)  # 4C H/4 W/4

        self.crb_0 = TAD(n_feat, kernel_size, reduction, act, bias, 8)
        self.crb_1 = TAD(n_feat * 2, kernel_size, reduction, act, bias, 4)
        self.crb_2 = TAD(n_feat * 4, kernel_size, reduction, act, bias, 4)
        self.crb_3 = nn.Sequential(*[TAD(n_feat, kernel_size, reduction, act, bias, 4) for _ in range(3)])

        self.up1_0 = UpSample(n_feat * 2, n_feat, 2)  # From Level 2 to Level 1
        self.up2_0 = UpSample(n_feat * 4, n_feat, 4)  # C H W

        self.point_conv_B = nn.Conv2d(n_feat * 3, n_feat, kernel_size=1)  # 调整通道数 C H W
        self.point_conv_R = nn.Conv2d(n_feat * 3, n_feat, kernel_size=1)

        self.rrb = RRB(n_feat, kernel_size, act, bias=bias)

    def forward(self, x):
        [B_fea, R_fea] = self.patch_embed(x)  # B C H W

        [out_B_0, out_R_0] = self.crb_0([B_fea, R_fea])

        [B_down_1, R_down_1] = self.down0_1(B_fea, R_fea)  # 2C H/2 W/2
        [out_B_1, out_R_1]= self.crb_1([B_down_1, R_down_1])
        [B_up1_0, R_up1_0] = self.up1_0(out_B_1, out_R_1)  # B C H W

        [B_down_2, R_down_2] = self.down0_2(B_fea, R_fea)  # B 4C H/4 W/4
        [out_B_2, out_R_2] = self.crb_2([B_down_2, R_down_2])
        [B_up2_0, R_up2_0] = self.up2_0(out_B_2, out_R_2)  # B C H W

        B_cat = torch.cat([out_B_0, B_up1_0, B_up2_0], 1)  # C + C + C
        R_cat = torch.cat([out_R_0, R_up1_0, R_up2_0], 1)

        B_fuse = self.point_conv_B(B_cat)  # 调整通道数 C H W
        R_fuse = self.point_conv_B(R_cat)

        [out_B_3, out_R_3] = self.crb_3([B_fuse, R_fuse])  # 进一步修正 C H W

        [img_B, img_R, streak] = self.rrb([out_B_3, out_R_3])

        return img_B, img_R, streak


##########################################################################
class Net(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=48, kernel_size=3, reduction=4, num_tb=4, bias=False):
        super(Net, self).__init__()

        act = nn.PReLU()
        self.model = MODEL(in_c, out_c, n_feat, kernel_size, reduction, act, bias, num_tb)

    def forward(self, x_img):  # b,c,h,w
        imitation, rain_out, streak = self.model(x_img)
        return [imitation, rain_out, streak]


# if __name__ == '__main__':
#     x = torch.rand((2, 3, 128, 128)).cuda()
#     model = Deraining().cuda()
#     out = model(x)
# if __name__ == '__main__':
#
#     input1 = torch.randn(1, 3, 128, 128).cuda()
#
#     # 记录开始时间
#     start_time = time.time()
#     i = 1
#     while (i > 0):
#         model = HPCNet().cuda(0)
#         i = i - 1
#         # misc1.print_module_summary(model, [input1, input2])
#         # flops, params = profile(model, inputs=(input1, input2))
#         # flops, params = clever_format([flops, params], "%.3f")
#         # print(flops)
#         # print(params)
#         out = model(input1)
#     # print(out[0].shape)
#     #
#     end_time = time.time()
#     print(f"程序执行了 {end_time - start_time} 秒。")
'''  + Number of FLOPs: 15.86G
  + Number of params: 5.88M
  '''
if __name__ == '__main__':
    from thop import profile
    from get_model_size import *

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    model = Net().cuda()
    input = torch.randn(1, 3, 512, 512).cuda()
    output = model(input)[0]
    flops, params = profile(model, inputs=(input,))
    print('flops(G): %.5f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))
    getModelSize(model)
