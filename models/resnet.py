import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import weights_init
import torchvision.models as models

class resnetShallow(nn.Module):
    def __init__(self, nlayers=18, n_classes=2, learned_billinear=False):
        super(resnetShallow, self).__init__()

        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        if nlayers == 18:
            self.trunk = models.resnet18(pretrained=True)
        elif nlayers == 34:
            self.trunk = models.resnet34(pretrained=True)
        else:
            assert False

        self.mode = 'learned_stack'
        if self.mode == 'stack':
            nchannels = 128 + 256 + 512
        else:
            self.upscale3 = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),)
            self.upscale3.apply(weights_init)

            self.upscale4 = nn.Sequential(
                    nn.ConvTranspose2d(512, 128, 7, stride=4, padding=3, output_padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),)
            self.upscale4.apply(weights_init)
            nchannels = 128 + 128 + 128

        self.classifier = nn.Conv2d(nchannels, self.n_classes, 1, padding=0)
        self.classifier.apply(weights_init)

        if self.learned_billinear:
            assert False, "padding should be properly set"
            self.upscale = nn.ConvTranspose2d(self.n_classes, self.n_classes, 63, stride=32, padding=31, output_padding=7)
            self.upscale.apply(weights_init)

    def cuda(self, gpuid):
        super(resnetShallow, self).cuda(gpuid)

    def forward(self, input):
        x = self.trunk.conv1(input)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x = self.trunk.layer1(x)
        x2 = self.trunk.layer2(x)
        x3 = self.trunk.layer3(x2)
        x4 = self.trunk.layer4(x3)

        # stack all together
        if self.mode == 'stack':
            x3u = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=True)
            x4u = F.interpolate(x4, scale_factor=4, mode="bilinear", align_corners=True)
        else:
            x3u = self.upscale3(x3)
            x4u = self.upscale4(x4)

        xall = torch.cat((x2, x3u, x4u), dim=1)

        out = {}
        score = self.classifier(xall)
        if self.learned_billinear:
            out = self.upscale(score)
        else:
            out = F.interpolate(score, scale_factor=8, mode="bilinear", align_corners=True)
        return out

    def load_pretrained(self):
        pass

class resnetDeep(nn.Module):
    def __init__(self, nlayers=50, n_classes=2, learned_billinear=False):
        super(resnetDeep, self).__init__()

        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        if nlayers == 50:
            self.trunk = models.resnet50(pretrained=True)
        elif nlayers == 101:
            self.trunk = models.resnet101(pretrained=True)
        elif nlayers == 152:
            self.trunk = models.resnet152(pretrained=True)
        else:
            assert False

        self.mode = 'learned_stack'
        if self.mode == 'stack':
            nchannels = 512 + 1024 + 2048
        else:
            self.upscale3 = nn.Sequential(
                    nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),)
            self.upscale3.apply(weights_init)

            self.upscale4 = nn.Sequential(
                    nn.ConvTranspose2d(2048, 512, 7, stride=4, padding=3, output_padding=3),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),)
            self.upscale4.apply(weights_init)
            nchannels = 512 + 512 + 512


        self.classifier = nn.Conv2d(nchannels, self.n_classes, 1, padding=0)
        self.classifier.apply(weights_init)

        if self.learned_billinear:
            assert False, "padding should be properly set"
            self.upscale = nn.ConvTranspose2d(self.n_classes, self.n_classes, 63, stride=32, padding=31, output_padding=7)
            self.upscale.apply(weights_init)

    def cuda(self, gpuid):
        super(resnetDeep, self).cuda(gpuid)

    def forward(self, input, feature_map=False):
        x = self.trunk.conv1(input)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x = self.trunk.layer1(x)
        x2 = self.trunk.layer2(x)
        x3 = self.trunk.layer3(x2)
        x4 = self.trunk.layer4(x3)

        # stack all together
        if self.mode == 'stack':
            x3u = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=True)
            x4u = F.interpolate(x4, scale_factor=4, mode="bilinear", align_corners=True)
        else:
            x3u = self.upscale3(x3)
            x4u = self.upscale4(x4)

        xall = torch.cat((x2, x3u, x4u), dim=1)

        out = {}
    
        score = self.classifier(xall)
        if self.learned_billinear:
            out = self.upscale(score)
        else:
            out = F.interpolate(score, scale_factor=8, mode="bilinear", align_corners=True)

        return out

    def load_pretrained(self):
        pass


class resnet18(resnetShallow):
    def __init__(self, n_classes=2, learned_billinear=False):
        super(resnet18, self).__init__(18, n_classes, learned_billinear)

class resnet34(resnetShallow):
    def __init__(self, n_classes=2, learned_billinear=False):
        super(resnet34, self).__init__(34, n_classes, learned_billinear)

class resnet50(resnetDeep):
    def __init__(self, n_classes=2, learned_billinear=False):
        super(resnet50, self).__init__(50, n_classes, learned_billinear)

class resnet101(resnetDeep):
    def __init__(self, n_classes=2, learned_billinear=False):
        super(resnet101, self).__init__(101, n_classes, learned_billinear)

class resnet152(resnetDeep):
    def __init__(self, n_classes=2, learned_billinear=False):
        super(resnet152, self).__init__(152, n_classes, learned_billinear)
