import enum

import torch
from torch import nn
from torch import autograd


class Flatten(nn.Module):

    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return torch.flatten(input, self.start_dim, self.end_dim)


class Clamper(nn.Module):

    def __init__(self, min=None, max=None):
        super().__init__()

        self.min = min
        self.max = max

    def forward(self, input):
        return torch.clamp(input, self.min, self.max)


class ResidualBlock(nn.Module):
    """
    https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels // 4,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // 4, num_channels,
                      kernel_size=1, bias=False)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.net(input)


class VectorQuantization(autograd.Function):
    """
    https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input, codebook):
        pass

    @staticmethod
    def backward(ctx, grad_outputs):
        pass


class CondGatedMaskedConv2d(nn.Module):

    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        num_classes: int,
        mask_type: str,
        skip_connection: bool = True,
    ) -> None:
        super().__init__()

        assert mask_type in ['A', 'B']
        assert kernel_size % 2 == 1

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.mask_type = mask_type
        self.skip_connection = skip_connection

        self.cond = nn.Embedding(num_classes, num_channels * 2)

        vertical_kernel_size = (kernel_size // 2 + 1, kernel_size)
        vertical_padding = kernel_size // 2
        self.vertical_conv = nn.Conv2d(num_channels, num_channels * 2, vertical_kernel_size,
                                       padding=vertical_padding)

        self.vertical2horizontal = nn.Conv2d(num_channels * 2, num_channels * 2, 1)

        horizontal_kernel_size = (1, kernel_size // 2 + 1)
        horizontal_padding = (0, kernel_size // 2)
        self.horizontal_conv = nn.Conv2d(num_channels, num_channels * 2, horizontal_kernel_size,
                                         padding=horizontal_padding)

        self.horizontal_residual = nn.Conv2d(num_channels, num_channels, 1)

    def enable_mask_a(self):
        self.horizontal_conv.weight.data[:, :, :, -1].zero_()
        self.vertical_conv.weight.data[:, :, -1].zero_()

    @staticmethod
    def gate(input: torch.Tensor):
        # split over channel
        left, right = input.chunk(2, dim=1)
        return torch.tanh(left) * torch.sigmoid(right)

    def forward(
        self,
        vertical_input: torch.Tensor,
        horizontal_input: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        if self.mask_type == 'A':
            self.enable_mask_a()

        condition = self.cond(condition)[:, :, None, None]

        vertical_logit = self.vertical_conv(vertical_input)[:, :, :vertical_input.size(-1), :]
        vertical_output = self.gate(vertical_logit + condition)

        horizontal_logit = self.horizontal_conv(horizontal_input)[:, :, :, :horizontal_input.size(-2)]
        v2h = self.vertical2horizontal(vertical_logit)
        horizontal_output = self.gate(v2h + horizontal_logit + condition)
        horizontal_output = self.horizontal_residual(horizontal_output)

        if self.skip_connection:
            horizontal_output = horizontal_output + horizontal_input

        return vertical_output, horizontal_output
