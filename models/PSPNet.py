import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_unit import *


class PspNet(nn.Module):
    def __init__(self, args, pretrained=True, use_aux=False, use_lstm=False):
        super(PspNet, self).__init__()
        self.use_aux = use_aux
        self.use_lstm = use_lstm
        base_model = BackNet("resnet101", pretrained=pretrained).back()
        self.layer0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4
        self.ppm = PyramidPoolingModule(args, 2048, 512, (1, 2, 3, 6), lstm=use_lstm)
        self.final1 = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(512, args.num_classes, kernel_size=1)
        )

        if self.use_aux:
            self.aux_logits = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256, momentum=.95),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, args.num_classes, kernel_size=1)
            )

    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final1(x)
        x = self.final2(x)
        if self.training and self.use_aux:
            return F.interpolate(x, x_size[2:], mode="bilinear"), F.interpolate(aux, x_size[2:], mode="bilinear")
        return F.interpolate(x, x_size[2:], mode="bilinear")


# input = torch.rand((4, 3, 512, 512))
# init_model = PspNet(12, use_aux=False, use_lstm=True)
# input = input.cuda().float()
# init_model.cuda()
# init_model.eval()
# out1 = init_model(input)
# print(out1.size())