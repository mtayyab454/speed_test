import torch
import numpy as np
from basisModel import basisModel, display_stats
import torchvision.models as models
import time

def get_basis_model(model, mul_factor):
    basis_model = basisModel(model, use_weights=True, add_bn=False, trainable_basis=True)
    if not isinstance(mul_factor, list):
        f_list = [27]
        f_list.extend((basis_model.num_original_filters.cpu().numpy() * mul_factor).tolist()[1:])
        f_list = torch.IntTensor(f_list).tolist()
    else:
        f_list = mul_factor
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

flop_reduction = {'2x':0.441, '3x':[9, 13, 32, 64, 64, 64, 90, 128, 128, 154, 154, 154, 154], '4x':[8, 10, 20, 45, 52, 52, 64, 103, 103, 103, 103, 103, 103],
                  '5x':[8, 10, 20, 39, 39, 39, 52, 77, 77, 77, 77, 77, 77]}
repetitions = 1000

################################
model = models.vgg16(pretrained=False)

############ Original Model ############
print('Original Model')
model_mu_time, model_times = measure_time(model, repetitions)
print(model_mu_time)

############ 2x Model ############
time.sleep(10)
print('2x Model')
basis_model = get_basis_model(model, flop_reduction['2x'])
# info = display_stats(basis_model, model, '2x', [224, 224])
bmodel_mu_time, bmodel_times = measure_time(basis_model, repetitions)
print(bmodel_mu_time)

############ 3x Model ############
print('3x Model')
basis_model = get_basis_model(model, flop_reduction['3x'])
# info = display_stats(basis_model, model, '3x', [224, 224])
bmodel_mu_time, bmodel_times = measure_time(basis_model, repetitions)
print(bmodel_mu_time)

############ 4x Model ############
time.sleep(10)
print('4x Model')
basis_model = get_basis_model(model, flop_reduction['4x'])
# info = display_stats(basis_model, model, '4x', [224, 224])
bmodel_mu_time, bmodel_times = measure_time(basis_model, repetitions)
print(bmodel_mu_time)

############ 5x Model ############
time.sleep(10)
print('5x Model')
basis_model = get_basis_model(model, flop_reduction['5x'])
# info = display_stats(basis_model, model, '5x', [224, 224])
bmodel_mu_time, bmodel_times = measure_time(basis_model, repetitions)
print(bmodel_mu_time)
