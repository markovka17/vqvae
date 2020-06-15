import torch
from torch import nn

from .modules import CondGatedMaskedConv2d


class Model(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.embeddings = nn.Embedding(in_channels, hidden_channels)

        self.blocks = nn.ModuleList([
            CondGatedMaskedConv2d(hidden_channels, 7, num_classes, 'A', False)
        ])

        for _ in range(num_layers - 1):
            self.blocks.append(
                CondGatedMaskedConv2d(hidden_channels, 3, num_classes, 'B', True)
            )

        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 8, in_channels, 1)
        )

        self.init_weight()

    def init_weight(self):
        for n, p in self.named_parameters():
            if n.endswith('bias'):
                p.data.fill_(0)

    def forward(self, input, condition) -> torch.Tensor:
        # input.shape == [B, C == 1, H, W]
        input = self.embeddings(input) \
            .squeeze(dim=1) \
            .permute(0, 3, 1, 2)

        vertical_input, horizontal_input = input, input
        for i, block in enumerate(self.blocks):
            vertical_input, horizontal_input = block(vertical_input, horizontal_input, condition)

        output = self.head(horizontal_input)
        return output

    @staticmethod
    def discretize(input: torch.Tensor, mu: int = 256):
        return (input * (mu - 1)).long()

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        resolution: int
    ) -> torch.Tensor:
        batch_size = condition.size(0)
        generated = torch.zeros(batch_size, 1, resolution, resolution) \
            .to(condition.device) \
            .long()

        for i in range(resolution):
            for j in range(resolution):
                logits = self.forward(generated, condition)
                probs = torch.softmax(logits[:, :, i, j], dim=-1)
                generated.data[:, :, i, j].copy_(probs.multinomial(1).data)

        return generated