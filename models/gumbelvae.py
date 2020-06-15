import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions

from .modules import Flatten
from .vqvae import Encoder as VQVAEEncoder, Decoder as VQVAEDecoder
from .vae import Criterion as VAECriterion, Encoder as VAEEncoder, \
    Decoder as VAEDecoder


class Criterion(VAECriterion):

    def __init__(
        self,
        beta: float = 1.0,
        likelihood: str = 'bernoulli'
    ):
        super().__init__(beta, likelihood)

    def forward(
        self,
        input: torch.Tensor,
        proposal: distributions.RelaxedOneHotCategorical,
        proposal_sample: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> torch.Tensor:

        if self.likelihood == 'bernoulli':
            likelihood = distributions.Bernoulli(probs=reconstruction)
        else:
            likelihood = distributions.Normal(reconstruction, torch.ones_like(reconstruction))

        likelihood = distributions.Independent(likelihood, reinterpreted_batch_ndims=-1)
        reconstruction_loss = likelihood.log_prob(input).mean()

        assert proposal.logits.dim() == 2, "proposal.shape == [*, D], D is shape of isotopic gaussian"

        prior = distributions.RelaxedOneHotCategorical(
            proposal.temperature,
            logits=torch.ones_like(proposal.logits)
        )
        regularization = (proposal.log_prob(proposal_sample) - prior.log_prob(proposal_sample)) \
            .mean()

        # evidence lower bound (maximize)
        total_loss = reconstruction_loss - self.beta * regularization

        return -total_loss, -reconstruction_loss, regularization


class Model(nn.Module):
    """
    https://arxiv.org/pdf/1611.01144.pdf
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        num_latents: int,
        temperature: torch.Tensor,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.temperature = temperature

        self.proposal_network = VAEEncoder(in_channels, hidden_channels)
        self.prenet = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, latent_dim * num_latents, kernel_size=1),
            Flatten(1)
        )

        # predict parameters for Categorical distributions
        self.proposal_logits_head = nn.Linear(latent_dim * num_latents, latent_dim * num_latents)

        self.generative_network = VAEDecoder(in_channels, hidden_channels, latent_dim * num_latents)

    def forward(self, input: torch.Tensor):
        encoder_output = self.proposal_network(input)
        bottleneck_resolution = encoder_output.size()[-2:]
        encoder_output = self.prenet(encoder_output)

        assert encoder_output.size(-1) == self.latent_dim * self.num_latents

        proposal_logits = self.proposal_logits_head(encoder_output) \
            .reshape(-1, self.num_latents, self.latent_dim) \
            .flatten(end_dim=-2)

        proposal_distribution = distributions.RelaxedOneHotCategorical(self.temperature, logits=proposal_logits)
        proposal_sample = proposal_distribution.rsample()
        proposal_sample_copy = proposal_sample
        proposal_sample = proposal_sample.reshape(-1, self.num_latents * self.latent_dim, 1, 1)
        proposal_sample = F.interpolate(proposal_sample, bottleneck_resolution)

        reconstruction = self.generative_network(proposal_sample)

        return reconstruction, proposal_distribution, proposal_sample_copy

    def generate(self, batch_size: int, device: torch.device = torch.device('cpu')):
        prior_distribution = torch.distributions.RelaxedOneHotCategorical(
            temperature=self.temperature,
            logits=torch.ones(batch_size * self.num_latents, self.latent_dim).to(device)
        )

        prior = prior_distribution.sample() \
            .reshape(-1, self.num_latents * self.latent_dim, 1, 1)

        generated = self.generative_network(prior)
        return generated


class ModelV2(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        temperature: torch.Tensor,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.temperature = temperature

        self.proposal_network = nn.Sequential(
            VQVAEEncoder(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, latent_dim, kernel_size=1),
        )

        self.proposal_logits_head = nn.Linear(latent_dim, latent_dim)

        self.generative_network = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, kernel_size=3, padding=1),
            VQVAEDecoder(hidden_channels, in_channels),
        )

    def forward(self, input: torch.Tensor):
        encoder_output = self.proposal_network(input)

        encoded_shape = encoder_output.shape
        encoded_resolution = encoded_shape[-1] * encoded_shape[-2]

        # rearrange feature maps into "latent space"
        rearranged_encoder_output = encoder_output \
            .flatten(start_dim=-2) \
            .transpose(-1, -2) \
            .flatten(end_dim=1)

        proposal_logits = self.proposal_logits_head(rearranged_encoder_output)

        proposal_distribution = distributions.RelaxedOneHotCategorical(self.temperature, logits=proposal_logits)
        proposal_sample = proposal_distribution.rsample()
        proposal_sample_copy = proposal_sample
        proposal_sample = proposal_sample.reshape(-1, encoded_resolution, self.latent_dim) \
            .transpose(-1, -2) \
            .reshape(encoded_shape)

        reconstruction = self.generative_network(proposal_sample)

        return reconstruction, proposal_distribution, proposal_sample_copy

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        latent_resolution: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        prior_distribution = torch.distributions.RelaxedOneHotCategorical(
            temperature=self.temperature,
            logits=torch.ones(batch_size * (latent_resolution ** 2), self.latent_dim).to(device)
        )

        prior = prior_distribution.sample() \
            .reshape(-1, latent_resolution ** 2, self.latent_dim) \
            .transpose(-1, -2) \
            .reshape(-1, self.latent_dim, latent_resolution, latent_resolution)

        generated = self.generative_network(prior)

        return generated