# PyNET model architecture, which is proposed by A.Ignatov, was used as a base model to construct proposed model architecture.

import torch.nn as nn
import torch
import math
import torch.nn.functional as F 

class raw_to_tonemappedHDR(nn.Module):

    def __init__(self, level, instance_norm=True, instance_norm_level_1=False):
        super(raw_to_tonemappedHDR, self).__init__()

        self.level = level

        self.conv_l1_d1 = ConvMultiBlock(4, 32, 3, instance_norm=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv_l2_d1 = ConvMultiBlock(32, 64, 3, instance_norm=instance_norm) 
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv_l3_d1 = ConvMultiBlock(64, 128, 3, instance_norm=instance_norm) 
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # -------------------------------------

        self.conv_l4_d1 = ConvMultiBlock(128, 256, 3, instance_norm=instance_norm)
        self.conv_l4_d2 = ConvMultiBlock(256, 256, 3, instance_norm=instance_norm)
        self.out_eca1_l4 = ECALayer(channels=256)
        self.conv_l4_d3 = ConvMultiBlock(256, 256, 3, instance_norm=instance_norm)
        self.conv_l4_d4 = ConvMultiBlock(256, 256, 3, instance_norm=instance_norm)
        self.out_eca2_l4 = ECALayer(channels=256)
        self.out_cbam_l4 = CBAM(channels=256, r=16)
        self.conv_t3b = UpsampleConvLayer(256, 128, 3)

        self.conv_l4_out = ConvLayer(256, 3, kernel_size=3, stride=1, relu=False)
        self.output_l4 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l3_d3 = ConvMultiBlock(256, 128, 3, instance_norm=instance_norm)
        self.out_eca_l3 = ECALayer(channels=128)
        self.conv_t2b = UpsampleConvLayer(128, 64, 3)
        self.out_cbam_l3 = CBAM(channels=128, r=8)
        self.conv_l3_out = ConvLayer(128, 3, kernel_size=3, stride=1, relu=False)
        self.output_l3 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l2_d2 = ConvMultiBlock(128, 64, 3, instance_norm=instance_norm)
        self.conv_l2_IEM1 = ConvLayer(64, 32, kernel_size=3, stride=1, relu=True)
        self.IEM_l2_1 = IEM_module(in_channels = 32)
        self.conv_l2_IEM2 = ConvLayer(64, 32, kernel_size=5, stride=1, relu=True)
        self.IEM_l2_2 = IEM_module(in_channels = 32)
        self.conv_l2_IEM3 = ConvLayer(64, 32, kernel_size=7, stride=1, relu=True)
        self.IEM_l2_3 = IEM_module(in_channels = 32)
        self.conv_l2_IEM4 = ConvLayer(64, 32, kernel_size=9, stride=1, relu=True)
        self.IEM_l2_4 = IEM_module(in_channels = 32)
        self.simam_l2 = simam_module()
        self.conv_l2_d3 = ConvMultiBlock(128, 64, 3, instance_norm=instance_norm)
        self.conv_t1b = UpsampleConvLayer(64, 32, 3)
       
        self.conv_l2_out = ConvLayer(64, 3, kernel_size=3, stride=1, relu=False)
        self.output_l2 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l1_d2 = ConvMultiBlock(64, 32, 3, instance_norm=instance_norm)
        self.conv_l1_IEM1 = ConvLayer(32, 32, kernel_size=3, stride=1, relu=True)
        self.IEM_l1_1 = IEM_module(in_channels = 32)
        self.conv_l1_IEM2 = ConvLayer(32, 32, kernel_size=5, stride=1, relu=True)
        self.IEM_l1_2 = IEM_module(in_channels = 32)
        self.conv_l1_IEM3 = ConvLayer(32, 32, kernel_size=7, stride=1, relu=True)
        self.IEM_l1_3 = IEM_module(in_channels = 32)
        self.conv_l1_IEM4 = ConvLayer(32, 32, kernel_size=9, stride=1, relu=True)
        self.IEM_l1_4 = IEM_module(in_channels = 32)

        self.simam_l1 = simam_module()
        self.conv_l1_d3 =ConvLayer(128, 32, kernel_size=3, stride=1, relu=True)

        self.conv_t0b = UpsampleConvLayer(32, 16, 3)
        self.conv_l1_out = ConvLayer(32, 3, kernel_size=3, stride=1, relu=False)
        self.output_l1 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l0_d1 = ConvLayer(16, 3, kernel_size=3, stride=1, relu=False)
        self.output_l0 = nn.Sigmoid()

    def level_4(self, pool3):
        
        conv_l4_d1 = self.conv_l4_d1(pool3)
        conv_l4_d2 = self.conv_l4_d2(conv_l4_d1)
        att_eca1_l4 = self.out_eca1_l4(conv_l4_d2)
        conv_l4_d3 = self.conv_l4_d3(att_eca1_l4)       
        conv_l4_d4 = self.conv_l4_d4(conv_l4_d3)
        att_eca2_l4 = self.out_eca2_l4(conv_l4_d4)
        out_cbam_l4 = self.out_cbam_l4(att_eca2_l4) 
        conv_t3b = self.conv_t3b(out_cbam_l4)
        conv_l4_out = self.conv_l4_out(out_cbam_l4)
        output_l4 = self.output_l4(conv_l4_out)

        return output_l4, conv_t3b

    def level_3(self, conv_l3_d1, conv_t3b):

        conv_l3_d2 = torch.cat([conv_l3_d1, conv_t3b], 1)
        conv_l3_d3 = self.conv_l3_d3(conv_l3_d2)
        att_eca_l3 = self.out_eca_l3(conv_l3_d3)
        out_cbam_l3 = self.out_cbam_l3(att_eca_l3) 
        conv_t2b = self.conv_t2b(out_cbam_l3)
        conv_l3_out = self.conv_l3_out(out_cbam_l3)
        output_l3 = self.output_l3(conv_l3_out)
        
        return output_l3, conv_t2b

    def level_2(self, conv_l2_d1, conv_t2b):

        z1 = torch.cat([conv_l2_d1, conv_t2b], 1)
        conv_l2_d2 = self.conv_l2_d2(z1)

        conv_l2_IEM1 = self.conv_l2_IEM1(conv_l2_d2)
        IEM_l2_1 = self.IEM_l2_1(conv_l2_IEM1)
        conv_l2_IEM2 = self.conv_l2_IEM2(conv_l2_d2)
        IEM_l2_2 = self.IEM_l2_2(conv_l2_IEM2)
        conv_l2_IEM3 = self.conv_l2_IEM3(conv_l2_d2)
        IEM_l2_3 = self.IEM_l2_3(conv_l2_IEM3) 
        conv_l2_IEM4 = self.conv_l2_IEM4(conv_l2_d2)
        IEM_l2_4 = self.IEM_l2_4(conv_l2_IEM4)
        z2 = torch.cat([IEM_l2_1, IEM_l2_2, IEM_l2_3, IEM_l2_4], 1)
        
        simam_l2 = self.simam_l2(z2)
        conv_l2_d3 = self.conv_l2_d3(simam_l2)     
        conv_t1b = self.conv_t1b(conv_l2_d3)
        conv_l2_out = self.conv_l2_out(conv_l2_d3)
        output_l2 = self.output_l2(conv_l2_out)

        return output_l2, conv_t1b

    def level_1(self, conv_l1_d1, conv_t1b):

        z1 = torch.cat([conv_l1_d1, conv_t1b], 1)
        conv_l1_d2 = self.conv_l1_d2(z1)
        conv_l1_IEM1 = self.conv_l1_IEM1(conv_l1_d2)
        IEM_l1_1 = self.IEM_l1_1(conv_l1_IEM1)
        conv_l1_IEM2 = self.conv_l1_IEM2(conv_l1_d2)
        IEM_l1_2 = self.IEM_l1_2(conv_l1_IEM2)
        conv_l1_IEM3 = self.conv_l1_IEM3(conv_l1_d2)
        IEM_l1_3 = self.IEM_l1_3(conv_l1_IEM3) 
        conv_l1_IEM4 = self.conv_l1_IEM4(conv_l1_d2)
        IEM_l1_4 = self.IEM_l1_4(conv_l1_IEM4)
        z2 = torch.cat([IEM_l1_1, IEM_l1_2, IEM_l1_3, IEM_l1_4], 1)

        simam_l1 = self.simam_l1(z2)
        conv_l1_d3 = self.conv_l1_d3(simam_l1)
        conv_t0b = self.conv_t0b(conv_l1_d3)
        conv_l1_out = self.conv_l1_out(conv_l1_d3)
        output_l1 = self.output_l1(conv_l1_out)

        return output_l1, conv_t0b

    def level_0(self, conv_t0b):

        conv_l0_d1 = self.conv_l0_d1(conv_t0b)
        output_l0 = self.output_l0(conv_l0_d1)

        return output_l0

    def forward(self, x):

        conv_l1_d1 = self.conv_l1_d1(x)
        pool1 = self.pool1(conv_l1_d1)

        conv_l2_d1 = self.conv_l2_d1(pool1)
        pool2 = self.pool2(conv_l2_d1)

        conv_l3_d1 = self.conv_l3_d1(pool2)
        pool3 = self.pool3(conv_l3_d1)

        output_l4, conv_t3b =self.level_4(pool3)

        if self.level < 4:
            output_l3, conv_t2b = self.level_3(conv_l3_d1, conv_t3b)
        if self.level < 3:
            output_l2, conv_t1b = self.level_2(conv_l2_d1, conv_t2b)
        if self.level < 2:
            output_l1, conv_t0b = self.level_1(conv_l1_d1, conv_t1b)
        if self.level < 1:
            output_l0 = self.level_0(conv_t0b)

        if self.level == 0:
            enhanced = output_l0
        if self.level == 1:
            enhanced = output_l1
        if self.level == 2:
            enhanced = output_l2
        if self.level == 3:
            enhanced = output_l3
        if self.level == 4:
            enhanced = output_l4

        return enhanced


class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, max_conv_size, instance_norm):

        super(ConvMultiBlock, self).__init__()
        self.max_conv_size = max_conv_size

        self.conv_3a = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)
        self.conv_3b = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 5:
            self.conv_5a = ConvLayer(in_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)
            self.conv_5b = ConvLayer(out_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 7:
            self.conv_7a = ConvLayer(in_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)
            self.conv_7b = ConvLayer(out_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 9:
            self.conv_9a = ConvLayer(in_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)
            self.conv_9b = ConvLayer(out_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)

    def forward(self, x):

        out_3 = self.conv_3a(x)
        output_tensor = self.conv_3b(out_3)

        if self.max_conv_size >= 5:
            out_5 = self.conv_5a(x)
            out_5 = self.conv_5b(out_5)
            output_tensor = torch.cat([output_tensor, out_5], 1)

        if self.max_conv_size >= 7:
            out_7 = self.conv_7a(x)
            out_7 = self.conv_7b(out_7)
            output_tensor = torch.cat([output_tensor, out_7], 1)

        if self.max_conv_size >= 9:
            out_9 = self.conv_9a(x)
            out_9 = self.conv_9b(out_9)
            output_tensor = torch.cat([output_tensor, out_9], 1)

        return output_tensor


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=2, stride=1, relu=True):

        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.upsample(x)
        out = self.reflection_pad(out)
        out = self.conv2d(out)

        if self.relu:
            out = self.relu(out)

        return out
    
class ECALayer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding='same', dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear_max = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))
        self.linear_avg = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear_max(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear_avg(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x    

class IEM_module(nn.Module):
    def __init__(self, in_channels):
    
        super(IEM_module, self).__init__()
        
        reflection_padding = 3//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False)
        self.act1 = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(in_channels, in_channels); #print('in_channels: ', in_channels)
        self.act2 = nn.ReLU()
        self.fc2 = nn.Linear(in_channels, int(in_channels/2))
        self.act3 = nn.ReLU()
        self.fc3 = nn.Linear (int(in_channels/2), in_channels)
        self.act4 = nn.Sigmoid()
        
    def forward(self, x):
       ref_pad = self.reflection_pad(x)
       conv1 = self.act1(self.conv1(ref_pad))
             
       avg_pool_out = self.avg_pool(x)
       avg_pool_out = torch.squeeze(avg_pool_out)
       z1 = self.act2(self.fc1(avg_pool_out))
       z2 = self.act3(self.fc2(z1))
       z3 = self.act4(self.fc3(z2))
       
       out1 = conv1 * torch.unsqueeze(torch.unsqueeze(z3, dim=-1),dim=-1)
       out = x * out1
       return out

class simam_module(nn.Module):
    def __init__(self, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):

        b, c, h, w = x.size()
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)