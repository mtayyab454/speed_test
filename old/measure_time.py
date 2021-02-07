import sys
sys.path.insert(1, '../base_code')

import os
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
from torch2trt import torch2trt
from helpers.helpers import get_basis_model
import math

from main import options, validate

class trt_basis_VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(trt_basis_VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
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

def construct_trt_basis_mode(basis_model):
    conf =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    layers = make_layers(conf, basis_model.num_basis_filters.tolist())
    return trt_basis_VGG(layers)

parser = options()
parser.set_defaults(jobid='measure_time', gpu=0, pretrained=True, batch_size=2048)
args = parser.parse_args()

model = models.vgg16(pretrained=True).eval().cuda()

# model, _ = get_basis_model('vgg16.3x', args)
# model = construct_trt_basis_mode(model)
# model = model.eval().cuda()


# create example data
x = torch.ones((1, 3, 224, 224)).cuda()
model_trt = torch2trt(model, [x])

# Data loading code
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

criterion = nn.CrossEntropyLoss().cuda(args.gpu)
t1 = time.time()
val_acc1, val_acc5, val_time = validate(val_loader, model_trt, criterion, args)
t2 = time.time()

print(t2-t1)