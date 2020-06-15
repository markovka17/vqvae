import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions

from .vqvae import Encoder as VQVAEEncoder, Decoder as VQVAEDecoder
from .modules import Clamper, Flatten


class Criterion(nn.Module):

    def __init__(
        self,
        beta: float = 1.0,
        likelihood: str = 'bernoulli'
    ):
        super().__init__()

        self.beta = beta

        if likelihood not in ['bernoulli', 'normal']:
            raise ValueError(f"Invalid likelihood: {likelihood}")
        self.likelihood = likelihood

    @staticmethod
    def kl(q_mu, q_sigma, p_mu, p_sigma):
        """
        Compute KL-divergence KL(q || p) between n pairs of Gaussians
        with diagonal covariance matrices (MultivariateNormal)

        Shape of all inputs is (B x D)
        """

        diff_mu = p_mu - q_mu

        q_sigma = q_sigma ** 2
        p_sigma = p_sigma ** 2

        kl = (torch.log(p_sigma) - torch.log(q_sigma)).sum(dim=-1, keepdim=True) - p_sigma.shape[-1]
        kl += (torch.reciprocal(p_sigma) * q_sigma).sum(dim=-1, keepdim=True)
        kl += (diff_mu * torch.reciprocal(p_sigma) * diff_mu).sum(dim=-1, keepdim=True)
        kl *= 0.5

        return kl

    def forward(
        self,
        input: torch.Tensor,
        proposal: distributions.Normal,
        reconstruction: torch.Tensor
    ) -> torch.Tensor:

        if self.likelihood == 'bernoulli':
            likelihood = distributions.Bernoulli(probs=reconstruction)
        else:
            likelihood = distributions.Normal(reconstruction, torch.ones_like(reconstruction))

        likelihood = distributions.Independent(likelihood, reinterpreted_batch_ndims=-1)
        reconstruction_loss = likelihood.log_prob(input).mean()

        assert proposal.loc.dim() == 2, "proposal.shape == [*, D], D is shape of isotopic gaussian"

        prior = distributions.Normal(torch.zeros_like(proposal.loc), torch.ones_like(proposal.scale))
        regularization = distributions.kl_divergence(
            proposal, prior
        ).sum(dim=-1).mean()

        # evidence lower bound (maximize)
        total_loss = reconstruction_loss - self.beta * regularization

        return -total_loss, -reconstruction_loss, regularization


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=3, stride=1),
        )

    def forward(self, input):
        return self.net(input)


class Decoder(nn.Module):

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim, hidden_channels,
                               kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels,
                               kernel_size=4, stride=2, padding=1),
            Clamper(-10, 10),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.net(input)


class Model(nn.Module):
    """
    https://arxiv.org/pdf/1906.02691.pdf

    latent_dim(VAE) == embedding_dim(VQVAE) for fairness
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        num_latents: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        self.proposal_network = Encoder(in_channels, hidden_channels)
        self.prenet = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, latent_dim * num_latents, kernel_size=1),
            Flatten(1)
        )

        self.proposal_mu_head = nn.Linear(latent_dim * num_latents, latent_dim * num_latents)
        self.proposal_sigma_head = nn.Sequential(
            nn.Linear(latent_dim * num_latents, latent_dim * num_latents),
            nn.Softplus()
        )

        self.generative_network = Decoder(in_channels, hidden_channels, latent_dim * num_latents)

    def forward(self, input: torch.Tensor):
        encoder_output = self.proposal_network(input)
        bottleneck_resolution = encoder_output.size()[-2:]
        encoder_output = self.prenet(encoder_output)

        assert encoder_output.size(-1) == self.latent_dim * self.num_latents

        proposal_mu = self.proposal_mu_head(encoder_output) \
            .reshape(-1, self.num_latents, self.latent_dim) \
            .flatten(end_dim=-2)
        proposal_sigma = self.proposal_sigma_head(encoder_output) \
            .reshape(-1, self.num_latents, self.latent_dim) \
            .flatten(end_dim=-2)

        proposal_distribution = distributions.Normal(proposal_mu, proposal_sigma)
        proposal_sample = proposal_distribution.rsample() \
            .reshape(-1, self.num_latents * self.latent_dim, 1, 1)
        proposal_sample = F.interpolate(proposal_sample, bottleneck_resolution)

        reconstruction = self.generative_network(proposal_sample)

        return reconstruction, proposal_distribution

    def generate(self, batch_size: int, device: torch.device = torch.device('cpu')):
        prior = torch.randn(batch_size * self.num_latents, self.latent_dim) \
            .reshape(-1, self.num_latents * self.latent_dim, 1, 1) \
            .to(device)

        generated = self.generative_network(prior)
        return generated


class ModelV2(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        self.proposal_network = nn.Sequential(
            VQVAEEncoder(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, latent_dim, kernel_size=1),
        )

        self.proposal_mu_head = nn.Linear(latent_dim, latent_dim)
        self.proposal_sigma_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softplus()
        )

        self.generative_network = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, kernel_size=3, padding=1),
            VQVAEDecoder(hidden_channels, in_channels),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        encoder_output = self.proposal_network(input)

        encoded_shape = encoder_output.shape
        encoded_resolution = encoded_shape[-1] * encoded_shape[-2]

        # rearrange feature maps into "latent space"
        rearranged_encoder_output = encoder_output \
            .flatten(start_dim=-2) \
            .transpose(-1, -2) \
            .flatten(end_dim=1)

        proposal_mu = self.proposal_mu_head(rearranged_encoder_output)
        proposal_sigma = self.proposal_sigma_head(rearranged_encoder_output)

        proposal_distribution = distributions.Normal(proposal_mu, proposal_sigma)
        proposal_sample = proposal_distribution.rsample() \
            .reshape(-1, encoded_resolution, self.latent_dim) \
            .transpose(-1, -2) \
            .reshape(encoded_shape)

        reconstruction = self.generative_network(proposal_sample)

        return reconstruction, proposal_distribution

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        latent_resolution: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        prior = torch.randn(batch_size * (latent_resolution ** 2), self.latent_dim) \
            .reshape(-1, latent_resolution ** 2, self.latent_dim) \
            .transpose(-1, -2) \
            .reshape(-1, self.latent_dim, latent_resolution, latent_resolution) \
            .to(device)

        generated = self.generative_network(prior)

        return generated