import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import json
from tqdm import tqdm

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

def data_loader(config, mode):
    transform = transforms.Compose([transforms.Resize((60, 60)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    if mode == 'train':
        dataset = torchvision.datasets.MNIST(root=config['data_dir'], train=True, download=True, transform=transform)
    elif mode == 'eval':
        dataset = torchvision.datasets.MNIST(root=config['data_dir'], train=False, download=True, transform=transform)
    elif mode == 'test':
        dataset = torchvision.datasets.MNIST(root=config['data_dir'], train=False, download=True, transform=transform)
    else:
        raise NotImplementedError('No such mode.')

    dataloader = DataLoader(dataset=dataset,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=config['num_workers'])

    return dataloader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_image(image_tensor, image_path):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def generate_reconstruction(model, test_loader, config):
    model.eval()

    with torch.no_grad():
        idx = 0
        for (input, target) in tqdm(test_loader):
            if config['cuda']:
                input = input.cuda()

            reconstruction, mu, logvar = model(input)
            reconstruction = reconstruction[0].cpu().detach()
            input = input[0].cpu().detach()
            save_image(reconstruction, os.path.join(config['reconstructions_save_path'], 'reconstruction_{}.png'.format(idx)))
            save_image(input, os.path.join(config['reconstructions_save_path'], 'input_{}.png'.format(idx)))
            
            idx += 1

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.pth.tar'))

def load_checkpoint(save_path, filename='checkpoint.pth.tar'):
    return torch.load(os.path.join(save_path, filename))

def load_config(config_path):
    with open(config_path, 'r') as handle:
        config = json.loads(handle.read())
    return config

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
