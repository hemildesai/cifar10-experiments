from functools import lru_cache as cache

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.utils import data

import matplotlib.pyplot as plt

cifar10_classes= 'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck'.split(', ')
mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

normalise = lambda data, mean=mean, std=std: (data - mean)/std
unnormalise = lambda data, mean=mean, std=std: data*torch.Tensor(std).unsqueeze(1).unsqueeze(2) + torch.Tensor(mean).unsqueeze(1).unsqueeze(2)
pad = lambda data, border: nn.ReflectionPad2d(border)(data)
transpose = lambda x, source='NHWC', target='NCHW': x.permute([source.index(d) for d in target]) 
to = lambda *args, **kwargs: (lambda x: x.to(*args, **kwargs))

transforms_train = torchvision.transforms.Compose([
  torchvision.transforms.RandomCrop(32, padding=4),
  torchvision.transforms.RandomHorizontalFlip(),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(mean, std),
])

transforms_test = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(mean, std),
])

@cache(None)
def cifar10(root='./storage/cifar10', batch_size=128):
    train =torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transforms_train)
    test =torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transforms_test)
    train_ds = data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_ds = data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_ds, test_ds
  
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    
    img = unnormalise(img)
    npimg = img.numpy()
    
    fig = plt.figure(figsize=(28,28))
    plt.axis('off')
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
def plot_history(history, figname=None):
  lrs = [h[2] for h in history]
  train_loss = [h[0]['train_loss'] for h in history]
  test_loss = [h[1]['test_loss'] for h in history]
  train_accuracy = [h[0]['train_accuracy'] for h in history]
  test_accuracy = [h[1]['test_accuracy'] for h in history]
  
  fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,16))
  ax1.plot(lrs)
  ax1.set(yscale='log', ylim=(0.0001, 0.5), title='Learning Rate Schedule', xlabel='Epoch', ylabel='Learning Rate')

  ax2.plot(train_loss, label='train_loss')
  ax2.plot(test_loss, label='test_loss')
  ax2.legend()
  ax2.set(title='Loss over epochs', xlabel='Epoch', ylabel='Loss')

  ax3.plot(train_accuracy, label='train_accuracy')
  ax3.plot(test_accuracy, label='test_accuracy')
  ax3.legend()
  ax3.set(title='Accuracy over epochs', xlabel='Epoch', ylabel='Accuracy')

  plt.show()
  if figname:
    fig.savefig(figname)