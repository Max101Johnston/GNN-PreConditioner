"""Seconds-based on-policy objective for a shared multi-frame policy."""
from dataclasses import dataclass, field
import math


def time_reward(reference_seconds, candidate_seconds, common_scale, timeout_seconds, converged):
    if not all(math.isfinite(x) and x > 0 for x in (reference_seconds, common_scale, timeout_seconds)):
        raise ValueError("Reference, common scale and time limit must be positive")
    successful = converged and math.isfinite(candidate_seconds) and 0 <= candidate_seconds <= timeout_seconds
    if not successful:
        # Strictly below every valid (Tref-T)/T* with T <= the fixed timeout.
        return -timeout_seconds/common_scale-1.0
    return (reference_seconds-candidate_seconds)/common_scale


def importance_weight(event_mass, probability, total_event_mass):
    if not all(math.isfinite(x) and x > 0 for x in (event_mass, probability, total_event_mass)):
        raise ValueError("Invalid event weights or sampling probability")
    if probability > 1 or event_mass > total_event_mass:
        raise ValueError("Weights do not describe a probability distribution")
    return event_mass/(total_event_mass*probability)


@dataclass
class RewardBaseline:
    """Past-only per-frame EMA, updated after constructing the current loss."""
    decay: float = 0.9
    values: dict = field(default_factory=dict)

    def get(self, key):
        return self.values.get(key, 0.0)

    def update(self, observations):
        grouped = {}
        for key, reward in observations:
            grouped.setdefault(key, []).append(float(reward))
        for key, rewards in grouped.items():
            mean = sum(rewards)/len(rewards)
            self.values[key] = self.decay*self.values.get(key, mean) + (1-self.decay)*mean


def reinforce_loss(rollouts, baseline):
    """One candidate per frame; no same-frame four-candidate requirement.

    Call once per frozen-policy batch. Do not replay stale trajectories or
    derive action likelihoods from the repaired C/F labels.
    """
    import torch
    if not rollouts:
        raise ValueError("Empty on-policy batch")
    terms = []
    for r in rollouts:
        if not r["samples"]:
            raise ValueError("A rollout must contain at least its first decision")
        log_probability = torch.stack([s.log_probability for s in r["samples"]]).sum()
        advantage = float(r["reward"])-float(baseline.get(r["key"]))
        terms.append(-float(r["importance"])*advantage*log_probability)
    return torch.stack(terms).mean()
