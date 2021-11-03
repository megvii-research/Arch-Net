import os
import torch
from tqdm import tqdm
import cv2
import numpy as np
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from datasets.imagenet_dataset import CustomImagenetDataset as imagenet_dataset


def imagenet_validation(model, val_data_path, num_threads=4, model_name=None, batch_size=32):
    ds = imagenet_dataset(val_data_path, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]))
    val_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_threads, pin_memory=True)

    # for calculating accuracy
    class_err = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for sample in tqdm(val_loader):
            inputs, targets = sample
            inputs = inputs.cuda()
            output = model(inputs)
            class_err.add(output.data, targets)

    if not model_name:
        print('Validation top1/top5 accuracy:', class_err.value())
    else:
        print(model_name + ' Validation top1/top5 accuracy:', class_err.value())

    return class_err.value()
