import torch
import torch.nn as nn
import torch.nn.functional as F


def round_up(x, m):
    return (x + m - 1) // m * m


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, expansion=4):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=False, stride=stride, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class ResNet(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], in_channels=6, stem_width=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(stem_width * 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(256, layers[2], stride=2, dilation=1)
        self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=2)

    def _make_layer(self, planes, num_blocks, stride, dilation, expansion=4):
        downsample = nn.Sequential(
            nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 4),
        )

        layers = [Bottleneck(planes * 2, planes, stride, downsample=downsample, dilation=1)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(planes * expansion, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        conv_out = [x]
        c0 = self.conv1(x)
        x = self.bn1(c0)
        x = F.relu(x, inplace=True)
        conv_out.append(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        conv_out.append(c1)
        c2 = self.layer2(c1)
        conv_out.append(c2)
        c3 = self.layer3(c2)
        conv_out.append(c3)
        c4 = self.layer4(c3)
        conv_out.append(c4)
        return conv_out


class tripool(nn.Module):
    def __init__(self, channels, pool_size, stride=1):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=pool_size, stride=stride, padding=(pool_size - 1) // 2, count_include_pad=False)
        self.conv = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=1, bias=True), nn.BatchNorm2d(256), nn.LeakyReLU(inplace=True))

    def forward(self, supp_feat_c):
        supp_feat = supp_feat_c[:, :-1, :, :]
        mask = supp_feat_c[:, -1:, :, :]
        supp_feat = self.conv(supp_feat)
        supp_feat_m = supp_feat * mask
        out = self.avgpool(supp_feat_m) / (self.avgpool(mask) + 1e-6)
        return out


class global_local_fusion(nn.Module):
    def __init__(self, channels, inter_channels, out_channels, kernel_size=3, upscale_factor=2, reduction_ratio=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.inter_channels = inter_channels
        self.group_channels = 16
        self.groups = self.inter_channels // self.group_channels
        self.reduce = nn.Conv2d(in_channels=channels, out_channels=inter_channels, kernel_size=1)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=inter_channels, out_channels=out_channels, kernel_size=1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True)
        )
        out_channels = inter_channels // reduction_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=inter_channels, out_channels=out_channels, kernel_size=1))
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=kernel_size**2 * self.groups, kernel_size=3, padding=1)
        self.embconv = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1)
        self.upscale_factor = upscale_factor

    def forward(self, high_level, low_level, emb):
        cat = torch.cat([F.pixel_shuffle(high_level, upscale_factor=self.upscale_factor), low_level], dim=1)
        cat = self.reduce(cat)
        weight = self.conv2(self.relu(self.conv1(cat) + self.embconv(emb)))
        b, _, h, w = weight.shape
        weight = weight.view(b, self.groups, 1, self.kernel_size**2, h, w)
        if 1:
            # Using loops to decrease memory usage. Performance is not too bad because range is small.
            cat = cat.view(b, self.groups, self.group_channels, h, w)
            out = torch.zeros((b, self.groups, self.group_channels, h, w), device=weight.device)
            for i_batch in range(b):
                for i_group in range(self.groups):
                    for y in range(self.kernel_size):
                        for x in range(self.kernel_size):
                            k = x + y * self.kernel_size
                            # Padded cat is warm and soft
                            padded_cat = F.pad(cat[i_batch, i_group], [self.kernel_size // 2] * 4, mode="constant")[:, y : y + h, x : x + w]
                            out[i_batch, i_group] += weight[i_batch, i_group, :, k, :, :] * padded_cat
        else:
            # Same as above, but requires more memory
            unfold = nn.Unfold(self.kernel_size, padding=1)
            out = unfold(cat).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
            out = (weight * out).sum(dim=3)
        out = out.view(b, self.inter_channels, h, w)
        return self.out(out)


class TMFdecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ppm = nn.ModuleList([tripool(2048, size) for size in (31, 17, 11, 5)])
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(2048 + len(self.ppm) * 256, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.globalpool = nn.AdaptiveAvgPool2d(1)

        self.conv_up2 = global_local_fusion(channels=64 + 256, inter_channels=256, out_channels=256)
        self.conv_up3 = global_local_fusion(channels=64 + 128, inter_channels=256, out_channels=64)
        self.conv_up4_1 = global_local_fusion(channels=16 + 3 + 3, inter_channels=32, out_channels=32)
        self.conv_up4_2 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.LeakyReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=1))

    def forward(self, conv_out):
        conv0, conv1, conv2, _, _, conv5 = conv_out
        mask = conv0[:, -2:, :, :].sum(1).unsqueeze(1)
        mask = F.interpolate(mask, conv5.shape[2:], mode="bilinear", align_corners=False)
        conv5_c = torch.cat([conv5, mask], dim=1)
        ppm_out = [conv5] + [pool_scale(conv5_c) for pool_scale in self.ppm]
        x = self.conv_up1(torch.cat(ppm_out, 1))
        emb = self.globalpool(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        x = self.conv_up2(x, conv2, emb)
        x = self.conv_up3(x, conv1, emb)
        x = self.conv_up4_1(x, conv0, emb)
        x = self.conv_up4_2(x)

        return x


# Dummy classes to make model.load_state_dict work without renaming
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = ResNet()


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = TMFdecoder()


class TMFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()

    def forward(self, image, trimap):
        _, _, h, w = trimap.shape

        # Pad image size to multiples of 16, which is required by network
        factor = 16
        h_pad = round_up(h, factor) - h
        w_pad = round_up(w, factor) - w

        pad = (0, w_pad, 0, h_pad)

        image = F.pad(image, pad, mode="replicate")
        trimap = F.pad(trimap, pad, mode="replicate")

        assert trimap.shape[2:] == (h + h_pad, w + w_pad)

        # Decompose trimap into foreground, background and unknown
        is_fg = (trimap == 1).float()
        is_bg = (trimap == 0).float()
        is_unknown = 1 - is_fg - is_bg

        inputs = torch.cat([image, is_bg, is_unknown, is_fg], dim=1)

        # Release memory
        del image, trimap, is_bg, is_unknown, is_fg

        conv_out = self.backbone.encoder.pretrained(inputs)

        outputs = self.backbone.decoder(conv_out)

        # Remove padding
        outputs = outputs[:, :, :h, :w]

        return outputs
