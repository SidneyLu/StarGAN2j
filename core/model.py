import math
import numpy as np
import jittor as jt
from jittor import nn
from munch import Munch

"""This is the module for network defining and building"""
"""模型"""

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample=False):
        super(ResBlk, self).__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = (dim_in != dim_out)
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = nn.AvgPool2d(2, stride=2)(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = nn.AvgPool2d(2, stride=2)(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def execute(self, x):
        out = self._shortcut(x) + self._residual(x)
        return out / math.sqrt(2)


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def execute(self, x, s):
        h = self.fc(s)
        b, c = h.shape[0], h.shape[1]
        h = h.reshape(b, c, 1, 1)

        gamma = h[:, :c//2, :, :]
        beta  = h[:, c//2:, :, :]
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                     actv=nn.LeakyReLU(0.2), upsample=False):
        super(AdainResBlk, self).__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = (dim_in != dim_out)
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)

        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)

        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def execute(self, x, s):
        out = self._residual(x, s)
        # 仅在未使用高通滤波时添加跳跃连接
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.filter = jt.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]]) / w_hpf

    def execute(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(0).expand(x.shape[1], 1, 3, 3)
        return nn.conv2d(x, filter, padding=1, groups=x.shape[1])


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()


        encoder_dims = [dim_in]

        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1

        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            encoder_dims.append(dim_out)
            dim_in = dim_out

        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))

        decoder_dims = encoder_dims[::-1]

        for _ in range(2):
            self.decode.append(AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        for i in range(len(decoder_dims) - 1):
            self.decode.append(AdainResBlk(decoder_dims[i], decoder_dims[i + 1],
                                           style_dim, w_hpf=w_hpf, upsample=True))

        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(decoder_dims[-1], affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(decoder_dims[-1], 3, 1, 1, 0))


    def execute(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}

        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)

        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = nn.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])

        return self.to_rgb(x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super(MappingNetwork, self).__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512), nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, style_dim)
            ))

    def execute(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out = jt.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # 按标签选择对应域的风格
        idx = jt.arange(y.shape[0])
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super(StyleEncoder, self).__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(math.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, style_dim))

    def execute(self, x, y):
        h = self.shared(x)
        h = h.reshape(h.shape[0], -1)
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out = jt.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = jt.arange(y.shape[0])
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super(Discriminator, self).__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(math.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def execute(self, x, y):
        out = self.main(x)
        out = out.reshape(out.shape[0], -1)  # (batch, num_domains)
        idx = jt.arange(y.shape[0])
        out = out[idx, y]
        return out


def build_model(args):
    G = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    F = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    E = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    D = Discriminator(args.img_size, args.num_domains)
    nets = Munch(generator=G, mapping_network=F, style_encoder=E, discriminator=D)
    G_ema = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    F_ema = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    E_ema = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    nets_ema = Munch(generator=G_ema, mapping_network=F_ema, style_encoder=E_ema)
    return nets, nets_ema