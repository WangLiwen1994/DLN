import torch
import torch.nn as nn

class LightenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(LightenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code+offset
        out = self.conv_Decoder(code_lighten)
        return out

class DarkenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DarkenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code-offset
        out = self.conv_Decoder(code_lighten)
        return out

class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.outlayer(y)
        return y


class LBP(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(LBP, self).__init__()
        self.fusion = FusionLayer(input_size,output_size)
        self.conv1_1 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DarkenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x=self.fusion(x)
        hr = self.conv1_1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1_1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2_1(hr)
        return hr_weight + h_residue


class DLN(nn.Module):
    def __init__(self, input_dim=3, dim=64):
        super(DLN, self).__init__()
        inNet_dim = input_dim + 1
        # 1:brightness
        self.feat1 = ConvBlock(inNet_dim, 2 * dim, 3, 1, 1)
        self.feat2 = ConvBlock(2 * dim, dim, 3, 1, 1)

        self.feat_out_1 = LBP(input_size=dim, output_size=dim, kernel_size=3, stride=1, padding=1)
        self.feat_out_2 = LBP(input_size=2 * dim, output_size=dim, kernel_size=3, stride=1,
                              padding=1)
        self.feat_out_3 = LBP(input_size=3 * dim, output_size=dim, kernel_size=3, stride=1,
                              padding=1)

        self.feature = ConvBlock(input_size=4 * dim, output_size=dim, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_ori, tar=None):
        # data gate
        x = (x_ori - 0.5) * 2
        x_bright, _ = torch.max(x_ori, dim=1, keepdim=True)

        x_in = torch.cat((x, x_bright), 1)

        # feature extraction
        feature = self.feat1(x_in)
        feature_1_in = self.feat2(feature)
        feature_1_out = self.feat_out_1(feature_1_in)

        feature_2_in = torch.cat([feature_1_in, feature_1_out], dim=1)
        feature_2_out = self.feat_out_2(feature_2_in)

        feature_3_in = torch.cat([feature_1_in, feature_1_out, feature_2_out], dim=1)
        feature_3_out = self.feat_out_3(feature_3_in)

        feature_in = torch.cat([feature_1_in, feature_1_out, feature_2_out, feature_3_out], dim=1)
        feature_out = self.feature(feature_in)
        pred = self.out(feature_out) + x_ori

        return pred


############################################################################################
# Base models
############################################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filter)

        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.act2(out)

        return out
