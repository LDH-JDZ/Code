import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        # 注意力门控向量
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 同层编码器特征图向量
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # ReLU激活函数
        self.hswish = nn.Handswish()
        # 卷积+BN+sigmoid激活函数
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )



    ###  Attention门控的前向计算流程
    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.hswish(g1 + x1)
        psi = self.psi(psi)
        return x * psi




class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.Att5 = Attention_block(F_g=out_size, F_l=out_size, F_int=int(out_size / 2))

    def forward(self, inputs1, inputs2,inputs3):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)

        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = torch.cat([outputs, self.up(inputs3)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        x = self.relu(outputs)

        outputs = self.Att5(g=x, x=outputs)
        return outputs
class Unet(nn.Module):
    def __init__(self, num_classes=3111111111, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        # print(feat1.shape,feat2.shape,feat3.shape,feat4.shape,feat5.shape)
        up4 = self.up_concat4(feat4, feat5, feat5)

        up3 = self.up_concat3(feat3, up4, feat4)
        up2 = self.up_concat2(feat2, up3, feat3)
        up1 = self.up_concat1(feat1, up2, feat2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
