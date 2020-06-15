from typing import Tuple, Dict

import random
import numpy as np

import torch
from torchvision import datasets, transforms

from sklearn.metrics.pairwise import cosine_distances
from matplotlib import pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

CIFAR10_ANNOTATION = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def plot_cifar_image(image, label=""):
    plt.title(label)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()


class AccumulateStats:

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def __call__(self):
        pass


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


class MeterLogger:

    def __init__(self, meters: Tuple[str], writer: SummaryWriter):
        self.average_meters: Dict[str, AverageMeter] = {k: AverageMeter() for k in meters}
        self._writer = writer

    def update(self, name: str, val, n=1):
        self.average_meters[name].update(val, n)

    def reset(self):
        for meter in self.average_meters.values():
            meter.reset()

    def write(self, step, prefix):
        for name, meter in self.average_meters.items():
            tag = prefix + '/' + name
            self._writer.add_scalar(tag, meter.avg, step)


class ImageLogger:

    def __init__(self, writer: SummaryWriter, mean=None, std=None):
        self._writer = writer
        self.mean = mean
        self.std = std

        if self.mean is not None:
            self.mean = torch.tensor(self.mean).reshape(1, 3, 1, 1)

        if self.std is not None:
            self.std = torch.tensor(self.std).reshape(1, 3, 1, 1)

    def write(self, images, reconstruction, step, prefix):
        images = images.cpu()
        reconstruction = reconstruction.cpu()

        if self.mean is not None and self.std is not None:
            images = images * self.std + self.mean
            reconstruction = reconstruction * self.std + self.mean

        image_tag = prefix + '/' + 'image'
        self._writer.add_images(image_tag, images, step)

        reconstruction_tag = prefix + '/' + 'reconstruction'
        self._writer.add_images(reconstruction_tag, reconstruction, step)


class VQEmbeddingLogger:

    def __init__(self, writer: SummaryWriter):
        self._writer = writer

    def write(self, embeddings, step):
        embeddings = embeddings.detach().cpu().numpy()
        sim = cosine_distances(embeddings)

        self._writer.add_image('cos_sim_vq_embeddings', sim, step, dataformats='HW')


def double_soft_orthogonality(weights: torch.Tensor):
    a = torch.norm(weights @ weights.t() - torch.eye(weights.shape[0]).to(weights.device)) ** 2
    b = torch.norm(weights.t() @ weights - torch.eye(weights.shape[1]).to(weights.device)) ** 2

    return a + b


def set_random_seed(seed: int, cuda: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
