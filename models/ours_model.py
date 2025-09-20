import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math


class Conv2d_SN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_sn=False, **kwargs):
        super(Conv2d_SN, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        if use_sn:
            conv = spectral_norm(conv)
        self.conv = conv

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2d_SN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, use_sn=False, **kwargs):
        super(ConvTranspose2d_SN, self).__init__()
        conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, **kwargs)
        if use_sn:
            conv_transpose = spectral_norm(conv_transpose)
        self.conv_transpose = conv_transpose

    def forward(self, x):
        return self.conv_transpose(x)


class Linear_SN(nn.Module):
    def __init__(self, in_features, out_features, use_sn=False, **kwargs):
        super(Linear_SN, self).__init__()
        linear = nn.Linear(in_features, out_features, **kwargs)
        if use_sn:
            linear = spectral_norm(linear)
        self.linear = linear

    def forward(self, x, label=None):
        return self.linear(x)


class Embedding_SN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, use_sn=False, **kwargs):
        super(Embedding_SN, self).__init__()
        embedding = nn.Embedding(num_embeddings, embedding_dim, **kwargs)
        if use_sn:
            embedding = spectral_norm(embedding)
        self.embedding = embedding

    def forward(self, x):
        return self.embedding(x)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=False)
        self.gain = nn.Linear(cond_dim, num_features, bias=False)
        self.bias = nn.Linear(cond_dim, num_features, bias=False)

    def forward(self, x, y):
        gamma = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        beta = self.bias(y).view(y.size(0), -1, 1, 1)
        return gamma * self.bn(x) + beta


class DoubleConvCBN(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.cbn1 = ConditionalBatchNorm2d(out_ch, cond_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.cbn2 = ConditionalBatchNorm2d(out_ch, cond_dim)

    def forward(self, x, y=None):
        x = F.relu(self.cbn1(self.conv1(x), y), inplace=True)
        x = F.relu(self.cbn2(self.conv2(x), y), inplace=True)
        return x


class DownCBN(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConvCBN(in_ch, out_ch, cond_dim)

    def forward(self, x, y=None):
        return self.double_conv(self.pool(x), y)


class UpCBN(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.double_conv = DoubleConvCBN(in_ch, out_ch, cond_dim)

    def forward(self, x1, x2, y=None):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x, y)


class Generator(nn.Module):
    def __init__(self, z_dim=None, n_class=5, img_size=64, out_channels=3):
        super().__init__()
        self.num_classes = n_class

        self.inc = DoubleConvCBN(out_channels, img_size, n_class)
        self.down1 = DownCBN(img_size, img_size * 2, n_class)
        self.down2 = DownCBN(img_size * 2, img_size * 2, n_class)

        self.up1 = UpCBN(img_size * 2 + img_size * 2, img_size * 2, n_class)
        self.up2 = UpCBN(img_size * 2 + img_size, img_size, n_class)

        self.outc = nn.Conv2d(img_size, out_channels, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, noise, labels):
        y = F.one_hot(labels, self.num_classes).float()

        x1 = self.inc(noise, y)
        x2 = self.down1(x1, y)
        x3 = self.down2(x2, y)

        x = self.up1(x3, x2, y)
        x = self.up2(x, x1, y)
        x = self.outc(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, d_cond_mtd=None, num_classes=5, img_size=64, in_channels=3, use_sn=True):
        super(Discriminator, self).__init__()
        self.d_cond_mtd = d_cond_mtd
        self.num_classes = num_classes

        def discriminator_block(in_filters, out_filters):
            block = [Conv2d_SN(in_filters, out_filters, 3, 2, 1, use_sn=use_sn),
                     nn.LeakyReLU(0.25, inplace=True)
                     ]
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(in_channels, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),)

        self.feature_dim = 128 * (img_size // 2 ** 4) ** 2
        self.adv = Linear_SN(in_features=self.feature_dim, out_features=1, use_sn=use_sn)
        self.ac = Linear_SN(in_features=self.feature_dim, out_features=num_classes*2, use_sn=use_sn)

    def forward(self, x, label, adc_fake=False):
        h = self.conv_blocks(x)
        h = h.view(h.shape[0], -1)
        adv_output = self.adv(h)

        if adc_fake:
            label = label * 2 + 1
        else:
            label = label * 2

        cls_output = self.ac(h, label)
        return {"h": h, "adv_output": adv_output, "cls_output": cls_output,  "label": label}