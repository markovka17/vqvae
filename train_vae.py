import pathlib
import argparse
import json
from pprint import pprint
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from torchvision import datasets, transforms

from models.vae import Model, Criterion, ModelV2
from utils import MeterLogger, ImageLogger, set_random_seed


def main(args):
    writer = SummaryWriter(args.experiment_log_path)
    writer.add_hparams(vars(args), {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((32, 32), 3),
        transforms.ToTensor(),
    ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
        args.in_channels = 3
    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
        args.in_channels = 1
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    train_dataloader = DataLoader(train_dataset, args.batch_size,
                                  shuffle=True, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, args.batch_size // 4,
                                 pin_memory=True, num_workers=4)

    if args.model_type == 'basic':
        model = Model(
            args.in_channels,
            args.hidden_channels,
            args.latent_dim,
            args.num_latents
        )
    elif args.model_type == 'resnet':
        model = ModelV2(
            args.in_channels,
            args.hidden_channels,
            args.latent_dim
        )
    else:
        raise ValueError(f"Invalid type model: {args.model_type}")

    model = model.to(device)
    criterion = Criterion(beta=args.beta, likelihood=args.likelihood_type)
    optimizer = optim.Adam(model.parameters(), args.lr)

    # Initialize Loggers
    train_metric_logger = MeterLogger(("total_loss", "reconstruction_loss", "kl"), writer)
    val_metric_logger = MeterLogger(("total_loss", "reconstruction_loss", "kl"), writer)
    image_logger = ImageLogger(writer)

    print(model)

    for epoch in tqdm(range(args.num_epoch)):

        train_metric_logger.reset()
        model.train()
        for train_batch in tqdm(train_dataloader):
            images, labels = train_batch
            images = images.to(device)

            reconstruction, proposal_distribution = model(images)
            total_loss, reconstruction_loss, kl = \
                criterion(images, proposal_distribution, reconstruction)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_metric_logger.update('total_loss', total_loss.item(), train_dataloader.batch_size)
            train_metric_logger.update('reconstruction_loss', reconstruction_loss.item(), train_dataloader.batch_size)
            train_metric_logger.update('kl', kl.item(), train_dataloader.batch_size)

        # Save train metrics
        train_metric_logger.write(epoch, 'train')
        image_logger.write(images, reconstruction, epoch, 'train')

        val_metric_logger.reset()
        model.eval()
        for test_batch in tqdm(test_dataloader):
            images, labels = test_batch
            images = images.to(device)

            with torch.no_grad():
                reconstruction, proposal_distribution = model(images)
                total_loss, reconstruction_loss, kl = \
                    criterion(images, proposal_distribution, reconstruction)

            val_metric_logger.update('total_loss', total_loss.item(), test_dataloader.batch_size)
            val_metric_logger.update('reconstruction_loss', reconstruction_loss.item(), test_dataloader.batch_size)
            val_metric_logger.update('kl', kl.item(), test_dataloader.batch_size)

        # Save val metrics
        val_metric_logger.write(epoch, 'val')
        image_logger.write(images, reconstruction, epoch, 'val')

        # Generate
        if args.model_type == 'basic':
            generated = model.generate(16, device).cpu()
        else:
            generated = model.generate(16, 8, device).cpu()
        writer.add_images('generated', generated, epoch)

        # Save checkpoint
        checkpoint_path = pathlib.Path(experiment_model_path) / f"{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of VQVAE')

    # Common
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--experiment-name', type=str)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=987)

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)

    # Model
    parser.add_argument('--model-type', type=str, default='basic')
    parser.add_argument('--likelihood-type', type=str, default='bernoulli')
    parser.add_argument('--hidden-channels', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=10)
    parser.add_argument('--num-latents', type=int, default=10)
    parser.add_argument('--beta', type=float, default=1.0)

    args = parser.parse_args()

    set_random_seed(args.seed)

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
