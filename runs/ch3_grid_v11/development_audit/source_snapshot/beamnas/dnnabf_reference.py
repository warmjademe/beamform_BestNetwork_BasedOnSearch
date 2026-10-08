"""Original DNNABF hidden widths, adapted to the frozen v2 feature/loss pipeline.

This is a fixed-backbone control, not an exact reconstruction of the original
AOA-to-tanh DNNABF method. See BASELINE_ALIGNMENT.md for the differences.
"""
from torch import nn

from .refined import constrained_canonical


class MatchedDNNABF(nn.Module):
    hidden_widths = (2048, 1024, 1024, 1024)

    def __init__(self):
        super().__init__()
        layers = []
        previous = 9 * 27
        for width in self.hidden_widths:
            layers.extend((nn.Linear(previous, width), nn.PReLU(width)))
            previous = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(previous, 24)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, mask):
        # Same padding and physical output handling as the v2 searched models.
        h = self.hidden((features * mask[..., None]).flatten(1))
        return constrained_canonical(self.output(h))

    @staticmethod
    def specification():
        return {
            'kind': 'fixed_dnnabf_backbone_matched_v2',
            'hidden_widths': list(MatchedDNNABF.hidden_widths),
            'activation': 'PReLU_per_hidden_unit',
            'input': 'relative_spatial_phase_27_features',
            'output': 'linear_then_centrohermitian_distortionless_projection',
            'not_original_full_method': True,
        }
