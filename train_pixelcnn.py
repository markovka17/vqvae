import pathlib
import argparse
import json
from pprint import pprint
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from torchvision import datasets, transforms

from models.pixelcnn import Model
from utils import MeterLogger, ImageLogger


def main(args):
    writer = SummaryWriter(args.experiment_log_path)
    writer.add_hparams(vars(args), {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    train_dataloader = DataLoader(train_dataset, args.batch_size,
                                  shuffle=True, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, args.batch_size // 4,
                                 pin_memory=True, num_workers=4)

    model = Model(args.in_channels, args.hidden_channels, args.num_layers, args.num_classes) \
        .to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)

    # Initialize Loggers
    train_metric_logger = MeterLogger(("nll",), writer)
    val_metric_logger = MeterLogger(("nll",), writer)

    print(model)

    for epoch in tqdm(range(args.num_epoch)):

        train_metric_logger.reset()
        model.train()
        for train_batch in tqdm(train_dataloader):
            images, labels = train_batch
            images = images.to(device)
            labels = labels.to(device)

            images = model.discretize(images)
            logits = model(images, labels)
            loss = criterion(logits, images.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metric_logger.update('nll', loss.item(), train_dataloader.batch_size)

        # Save train metrics
        train_metric_logger.write(epoch, 'train')

        val_metric_logger.reset()
        model.eval()
        for test_batch in tqdm(test_dataloader):
            images, labels = test_batch
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                images = model.discretize(images[:, :1])
                logits = model(images, labels)
                loss = criterion(logits, images.squeeze())

            val_metric_logger.update('nll', loss.item(), test_dataloader.batch_size)

        # Save val metrics
        val_metric_logger.write(epoch, 'val')

        # Generate
        resolution = 32 if args.dataset == 'cifar10' else 28
        generated = model.generate(torch.arange(10).to(device), resolution).cpu()
        writer.add_images('generated', generated, epoch)

        # Save checkpoint
        checkpoint_path = pathlib.Path(experiment_model_path) / f"{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of VQVAE')

    # Common
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--experiment-name', type=str)
    parser.add_argument('--use-cuda', action='store_true')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)

    # Model
    parser.add_argument('--in-channels', type=int, default=256)
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--num-classes', type=int, default=10)

    args = parser.parse_args()

    experiment_root = pathlib.Path('experiments') / args.experiment_name
    args.experiment_root = str(experiment_root)
    if not experiment_root.exists():
        experiment_root.mkdir()

    with open(experiment_root / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    experiment_log_path = experiment_root / 'logs'
    args.experiment_log_path = str(experiment_log_path)
    if not experiment_log_path.exists():
        experiment_log_path.mkdir()

    experiment_model_path = experiment_root / 'models'
    args.experiment_model_path = str(experiment_model_path)
    if not experiment_model_path.exists():
        experiment_model_path.mkdir()

    pprint(vars(args))
    main(args)
