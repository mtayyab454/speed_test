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

def measure_time(model, repetitions):
    device = torch.device("cuda")
    model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    print('warming up')
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    print('measuring time')
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    return mean_syn, timings
    # print(mean_syn)

mul_factor = 0.441
repetitions = 100

################################
model = models.vgg16(pretrained=False)
basis_model = get_basis_model(model, mul_factor)

model_mu_time, model_times = measure_time(model, repetitions)
print(model_mu_time)

bmodel_mu_time, bmodel_times = measure_time(basis_model, repetitions)
print(bmodel_mu_time)

info = display_stats(basis_model, model, 'test', [224, 224])
