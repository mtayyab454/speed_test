import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Data loading code
valdir = os.path.join('../../data/ImageNet/val10')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

def get_dataloader(batch_size, workers):
	val_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(valdir, transforms.Compose([
	    transforms.Resize(256),
	    transforms.CenterCrop(224),
	    transforms.ToTensor(),
	    normalize,
	])),
	batch_size=batch_size, shuffle=False,
	num_workers=workers, pin_memory=True)

	return val_loader

# d = get_dataloader(1, 1)
