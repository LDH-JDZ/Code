import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .utils import *
import torch

import torch.nn.functional as F








class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)

        feat1 = self.features[:4](x)

        feat2 = self.features[4:9](feat1)

        feat3 = self.features[9:16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:-1](feat4)
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:

                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




class con1v2d(nn.Module):
    def __init__(self, in_channels, v, ):
        super().__init__()

        self.out11 = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)

        self.bn1 = nn.BatchNorm2d(v)
        self.out22 = nn.Conv2d(in_channels, v, kernel_size=3,stride=1,padding=1)
        self.relu = torch.nn.ReLU()



    def forward(self, inp):

        out_f = self.out22(inp)

        # 例如，使用残差连接
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(ResidualBlock, self).__init()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            def forward(self, x):
                out = self.conv1(x)
                out = F.relu(out)
                out = self.conv2(out)
                out += x  # 残差连接
                out = F.relu(out)
                return out


class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        return self.gamma * out + input

class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        y = self.avgpool(x)
       #print(y.shape)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        z = self.upsample(x)
        #print(z.shape)
        z = self.max(z)
        z = self.sigmoid(z)
        out = self.sigmoid(y)

        out = out * x + z * x
        out = self.relu(out)
        return out


#nn.Sequential

def make_layers(cfg, batch_norm=False, in_channels = 3,groups=1, kernel_size=56,stride=1,base_width=64,nf=32):
    layers = []
    for v in cfg:
        if v == 'M':

            layers += [nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),SE_Block(in_channels))]

        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)





            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    return nn.Sequential(*layers)
# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels = 3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

    del model.avgpool
    del model.classifier
    return model

