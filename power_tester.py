import torch
import numpy as np
from basisModel import basisModel, display_stats
import torchvision.models as models

def get_basis_model(model, mul_factor):
    basis_model = basisModel(model, use_weights=True, add_bn=False, trainable_basis=True)
    f_list = [27]
    f_list.extend((basis_model.num_original_filters.cpu().numpy() * mul_factor).tolist()[1:])
    f_list = torch.IntTensor(f_list).tolist()
    basis_model.update_channels(f_list)

    return basis_model

def run_model(model, repetitions):
    device = torch.device("cuda")
    model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
    # GPU-WARM-UP
    print('warming up')
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    print('measuring time')
    with torch.no_grad():
        for rep in range(repetitions):
            _ = model(dummy_input)

        torch.cuda.synchronize()

    # print(mean_syn)

mul_factor = 0.441
repetitions = 100

################################
model = models.vgg16(pretrained=False)
basis_model = get_basis_model(model, mul_factor)

run_model(model, repetitions)
run_model(basis_model, repetitions)

info = display_stats(basis_model, model, 'test', [224, 224])
