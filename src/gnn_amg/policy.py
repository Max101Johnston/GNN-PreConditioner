"""Small shared GNN and two Bernoulli selection heads.

The second call must receive features and edges from the actually constructed
K1. A rollout contains two calls, regardless of the number of selected DOFs.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Bernoulli


class TwoStagePolicy(nn.Module):
    def __init__(self, feature_count, width=32):
        super().__init__()
        self.encoder = nn.Linear(feature_count, width)
        self.self_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(2))
        self.neighbour_layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(2))
        self.heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(2))

    def forward(self, features, normalized_strength, connection_prior, stage):
        if stage not in (0, 1):
            raise ValueError("stage must be 0 or 1")
        n = len(features)
        if normalized_strength.shape != (n, n) or connection_prior.shape != (n,):
            raise ValueError("Graph, features and prior have different DOF mappings")
        h = torch.nn.functional.gelu(self.encoder(features))
        for local, neighbour, norm in zip(self.self_layers, self.neighbour_layers, self.norms):
            aggregate = torch.sparse.mm(normalized_strength, h)
            h = torch.nn.functional.gelu(norm(local(h) + neighbour(aggregate)))
        return connection_prior + self.heads[stage](h).squeeze(-1)


@dataclass(frozen=True)
class Sample:
    raw: torch.Tensor
    explored: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor


def sample_local(logits, frozen_selection, exploration_mask, eligible_mask):
    """Archive the raw draw. Numerical repair must operate on another copy."""
    if not (logits.shape == frozen_selection.shape == exploration_mask.shape == eligible_mask.shape):
        raise ValueError("Action masks have different shapes")
    active = exploration_mask.bool() & eligible_mask.bool()
    if not torch.any(active):
        raise ValueError("At least one eligible DOF must be explored")
    distribution = Bernoulli(logits=logits[active])
    draws = distribution.sample()
    action = frozen_selection.detach().bool().clone() & eligible_mask.bool()
    action[active] = draws.bool()
    return Sample(raw=action.detach().clone(), explored=active.detach().clone(),
                  log_probability=distribution.log_prob(draws).sum(),
                  entropy=distribution.entropy().sum())


def supervised_loss(logits, labels, eligible):
    """Class-balanced binary loss; true solver convergence is a separate gate."""
    values, target = logits[eligible], labels[eligible].to(logits.dtype)
    positive = target.sum()
    if positive <= 0 or positive >= len(target):
        raise ValueError("Teacher must contain both C and F free DOFs")
    weight = (len(target)-positive)/positive
    return nn.functional.binary_cross_entropy_with_logits(values, target, pos_weight=weight)
