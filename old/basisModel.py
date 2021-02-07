import torch
import torch.nn as nn
import math

class trt_basis_VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, lf=512):
        super(trt_basis_VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(lf * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, num_basis_filters, batch_norm=False):
    layers = []
    in_channels = 3
    count = -1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            count += 1
            conv2d = nn.Conv2d(in_channels, num_basis_filters[count], kernel_size=3, padding=1)
            conv2d1 = nn.Conv2d(num_basis_filters[count], v, kernel_size=1, padding=0)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, conv2d1, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def get_basis_model(conf, ratio):

    basis_f = []
    for v in conf:
        if not v == 'M':
            basis_f.append( math.floor(v*ratio) )
    layers = make_layers(conf, basis_f)
    return trt_basis_VGG(features=layers, lf=conf[-2])
