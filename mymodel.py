#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


# He重みの初期化
def weight_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class normMSE(nn.Module):
    def __init__(self):
        super(normMSE, self).__init__()

    def forward(self, inp, tar):
        pad = torch.where(tar != -1)
        inp = inp[pad]
        tar = tar[pad]
        mse = F.mse_loss(inp, tar)
        return mse / torch.mean(tar)


class conv1DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super(conv1DBatchNorm, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        output = self.batchnorm(x)

        return output


class conv1DBatchNormReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super(conv1DBatchNormReLU, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.relu(x)

        return output


class conv1DBatchNormMish(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super(conv1DBatchNormMish, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.mish = nn.Mish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.mish(x)

        return output


class Tconv1DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super(Tconv1DBatchNorm, self).__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)

        return x


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.embedding = nn.Embedding(6, 120)
        self.convs = nn.ModuleList()
        self.convs.append(
            conv1DBatchNormMish(
                in_channels=120, out_channels=120, kernel_size=9, padding=4
            )
        )
        for i in range(40):
            self.convs.append(
                conv1DBatchNormReLU(
                    in_channels=120,
                    out_channels=120,
                    kernel_size=5,
                    dilation=3,
                    padding=6,
                )
            )
        self.convs.append(
            conv1DBatchNormMish(
                in_channels=120, out_channels=120, kernel_size=9, padding=4
            )
        )
        self.convs.append(
            conv1DBatchNormReLU(
                in_channels=120, out_channels=1, kernel_size=9, padding=4
            )
        )

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        attention_show_flg=False,
    ):
        x = self.embedding(input_ids)
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)

        return x.view(x.shape[0], -1)


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.embedding = nn.Embedding(6, 120)
        self.encoder = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.conv2 = nn.ModuleList()

        self.encoder.append(
            conv1DBatchNorm(
                in_channels=120, out_channels=240, kernel_size=5, stride=2, padding=2
            )
        )
        self.encoder.append(
            conv1DBatchNorm(
                in_channels=240, out_channels=360, kernel_size=5, stride=2, padding=2
            )
        )
        self.encoder.append(
            conv1DBatchNorm(
                in_channels=360, out_channels=480, kernel_size=5, stride=2, padding=2
            )
        )

        for i in range(35):
            self.conv1.append(
                conv1DBatchNormReLU(
                    in_channels=480,
                    out_channels=480,
                    kernel_size=5,
                    padding=6,
                    dilation=3,
                )
            )

        self.decoder.append(
            Tconv1DBatchNorm(
                in_channels=480, out_channels=360, kernel_size=5, stride=2, padding=2
            )
        )
        self.decoder.append(
            Tconv1DBatchNorm(
                in_channels=360, out_channels=240, kernel_size=5, stride=2, padding=2
            )
        )
        self.decoder.append(
            Tconv1DBatchNorm(
                in_channels=240, out_channels=120, kernel_size=5, stride=2, padding=2
            )
        )

        self.conv2.append(
            conv1DBatchNormMish(
                in_channels=120, out_channels=120, kernel_size=9, padding=4
            )
        )
        self.conv2.append(
            conv1DBatchNormReLU(
                in_channels=120, out_channels=1, kernel_size=9, padding=4
            )
        )

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        attention_show_flg=False,
    ):
        x = self.embedding(input_ids)
        x = torch.transpose(x, 1, 2)
        u_pre = []
        for i, l in enumerate(self.encoder):
            u_pre.append(x)
            x = l(x)
            x = F.relu(x)
        for i, l in enumerate(self.conv1):
            x = l(x)
        for i, l in enumerate(self.decoder):
            x = l(x)
            x = F.relu(x + u_pre[-(i + 1)])
        for i, l in enumerate(self.conv2):
            x = l(x)

        return x.view(x.shape[0], -1)


class RBERT(nn.Module):
    def __init__(self, net_bert):
        super(RBERT, self).__init__()
        self.bert = net_bert
        self.conv1 = conv1DBatchNormMish(
            in_channels=120, out_channels=120, kernel_size=9, padding=4
        )
        self.conv2 = conv1DBatchNormReLU(
            in_channels=120, out_channels=1, kernel_size=9, padding=4
        )

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        attention_show_flg=False,
    ):
        encoded_layers, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            attention_show_flg=False,
        )
        x = encoded_layers
        x = torch.transpose(x, 1, 2)
        x = self.conv2(self.conv1((x)))

        return x.view(x.shape[0], -1)
