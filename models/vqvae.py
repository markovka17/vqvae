import torch
import torch.nn.functional as F
from torch import distributions
from torch import nn

from .modules import ResidualBlock, Clamper


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class Decoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=4, stride=2, padding=1),
            Clamper(-10, 10),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class VectorQuantizer(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

        self.scale = 1. / self.num_embeddings
        torch.nn.init.uniform_(self.embeddings.weight, -self.scale, self.scale)

    def proposal_distribution(self, input: torch.Tensor):
        # input.shape == [B, C, H, W]

        input = input.permute(0, 2, 3, 1)
        input_shape = input.shape

        # .shape == [B * H * W, C]
        # worth noting that each image has (H * W) codes
        # (each code match some pixel) of size C
        flatten_input = input.flatten(end_dim=-2).contiguous()

        distances = (flatten_input ** 2).sum(dim=1, keepdim=True)
        distances = distances + (self.embeddings.weight ** 2).sum(dim=1)
        distances -= 2 * flatten_input @ self.embeddings.weight.t()

        categorical_posterior = torch.argmin(distances, dim=-1, keepdim=True) \
            .view(input_shape[:-1])

        return categorical_posterior

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        proposal = self.proposal_distribution(input)

        quantized = self.embeddings(proposal) \
            .permute(0, 3, 1, 2) \
            .contiguous()

        return quantized


class Criterion(nn.Module):
    """
    vq_loss: \| \text{sg}[I(x, e)] * e  - \text{sg}[z_e(x)] \|_2^2
    """

    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        input: torch.Tensor,
        encoder_output: torch.Tensor,
        quantized: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        flatten_quantized = quantized.permute(0, 2, 3, 1).flatten(end_dim=-2)
        flatten_encoder_output = encoder_output.permute(0, 2, 3, 1).flatten(end_dim=-2)

        reconstruction_loss = F.mse_loss(input, reconstruction)
        vq_loss = F.mse_loss(flatten_encoder_output.detach(), flatten_quantized)
        commitment_loss = F.mse_loss(flatten_encoder_output, flatten_quantized.detach())

        total_loss = reconstruction_loss + vq_loss + self.beta * commitment_loss

        return total_loss, reconstruction_loss, vq_loss, commitment_loss


class Model(nn.Module):
    """
    https://arxiv.org/abs/1711.00937
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_channels)
        self.decoder = Decoder(hidden_channels, in_channels)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        self.prenet = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1)
        self.postnet = nn.Conv2d(embedding_dim, hidden_channels,
                                 kernel_size=3, padding=1)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(input)
        encoder_output = self.prenet(encoder_output)
        quantized = self.vector_quantizer(encoder_output)

        return quantized

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(input)
        encoder_output = self.prenet(encoder_output)

        quantized = self.vector_quantizer(encoder_output)

        # Straight Through Estimator (Some Magic)
        st_quantized = encoder_output + (quantized - encoder_output).detach()
        post_quantized = self.postnet(st_quantized)

        reconstruction = self.decoder(post_quantized)

        return encoder_output, quantized, reconstruction

    @torch.no_grad()
    def generate(self, prior):
        raise NotImplementedError()

