import torch
from basisModel import get_basis_model
import time
from dataset import get_dataloader

model = get_basis_model()
print(model)
print('Moving to cuda...')
model.cuda()

print('Starting...')

dl = get_dataloader(8, 4)
total_t = 0

for i, (images, target) in enumerate(dl):

	images = images.cuda(0, non_blocking=True)
	# compute output
	
	torch.cuda.synchronize()
	t0 = time.time()
	output = model(images)
	torch.cuda.synchronize()
	t1 = time.time()
	
	t = t1-t0
	total_t += t
	print(i, t)


print("Done")
print(total_t)
