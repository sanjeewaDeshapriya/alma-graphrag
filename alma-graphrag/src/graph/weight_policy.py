"""
Weight policies — how the composite-score weight vector is decided.

Background
----------
`src/graph/retriever.py` combines five sub-scores into one ranking score:

    score = w_spatial·spatial + w_access·accessibility + w_facility·facility
          + w_economic·economic + w_disruption·disruption

Until now `w` came from one of two places, and both are weak claims:

  * `weights_for_intent()` — a ladder of hand-written `+= 0.20` rules. Every
    constant in it was chosen by hand. There are eleven of them.
  * `WEIGHT_PROFILES` — three fixed vectors, one of which (`elicited`) is
    estimated from the discrete-choice study. Better founded, but static: the
    same vector is applied to "cheapest hotel near the station" and to
    "quiet 5-star away from traffic", which plainly want different trade-offs.

This module replaces both with a *model* that maps query context to a weight
vector, and provides the two static policies as named baselines so the
comparison is like-for-like.

The model
---------
A small MLP reads a context vector (query intent + live pool conditions) and
emits the concentration parameters of a Dirichlet distribution:

    h = MLP(x)              (context -> 64 -> 64 -> 5)
    alpha = softplus(h) + 1 (strictly > 1, so the density is unimodal)
    w ~ Dirichlet(alpha)    (training)      w = alpha / sum(alpha)  (inference)

Why a Dirichlet rather than a softmax head or five sigmoids: the weight vector
must lie on the simplex (non-negative, sums to one) or the composite score is
not a convex combination and the per-component explanation bars stop being
comparable. The Dirichlet has the simplex as its support by construction, so
the clamp-and-renormalise step that `weights_for_intent` ends with — which
silently distorts any vector it touches — disappears. It also gives a
distribution rather than a point, which is what makes policy-gradient training
possible.

Why reinforcement learning rather than supervised regression: there are no
ground-truth weight vectors to regress onto. What exists is a ranking quality
measure, nDCG@10, and nDCG is computed *after* a sort, so it is piecewise
constant in the weights — gradient is zero almost everywhere and undefined at
the swaps. REINFORCE sidesteps this by differentiating the log-probability of
the sampled weights rather than the metric:

    grad J = E[ (R - b) * grad log p(w | x) ]

with R = nDCG@10 of the ranking that `w` produced and `b` a moving-average
baseline for variance reduction. The retriever is treated as a black box, which
it should be — nothing about `_score` needs to become differentiable.

This is a one-step contextual bandit, not a sequential MDP: one query in, one
weight vector out, one reward. Framing it as a multi-step MDP would add a
discount factor and credit-assignment machinery with nothing to assign credit
across.

Guardrails (the reason this can be believed)
--------------------------------------------
A 5-output net trained on 60 queries will memorise them given the chance. So:

  * `scripts/train_weight_policy.py` trains under k-fold cross-validation over
    *queries* and reports held-out nDCG only. Training-set nDCG is printed but
    is not a result.
  * The net is deliberately small (two hidden layers of 64) with weight decay,
    entropy regularisation and early stopping on a validation fold.
  * The policy must beat BOTH `handset` and `elicited` on held-out folds to be
    worth reporting. If it does not, that is the finding — say so.

Usage
-----
    from src.graph.weight_policy import get_policy
    policy = get_policy("learned")            # or "handtuned" / "elicited"
    retriever = WeightedRetriever(weight_model=policy)

Torch is imported lazily, so importing this module costs nothing in the API
process and the hand-tuned path has no torch dependency at all.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.crag.query_parser import QueryIntent
from src.graph.retriever import (
    ScoringWeights,
    WEIGHT_PROFILES,
    base_weights,
    weights_for_intent,
)

logger = logging.getLogger("alma.graph.weight_policy")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "weight_policy.pt"

# The five components the policy allocates over, in a FIXED order. Everything —
# feature vector, network output, checkpoint — depends on this order, so it is
# defined once here and never re-spelled.
COMPONENTS: Sequence[str] = ("spatial", "accessibility", "facility",
                             "economic", "disruption")


# ---------------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------------

# Feature names in fixed order. Kept explicit (rather than derived from a dict)
# so a checkpoint trained today stays readable and a reordering is a visible
# diff rather than a silent accuracy loss.
FEATURE_NAMES: Sequence[str] = (
    # -- query intent: what the user asked for -------------------------------
    "sort_best_overall",
    "sort_cheapest",
    "sort_highest_rated",
    "sort_most_accessible",
    "accessibility_high",
    "avoid_traffic",
    "proximity_close",
    "proximity_far",
    "proximity_any",
    "has_max_price",
    "has_min_price",
    "has_min_rating",
    "has_min_star",
    "n_required_amenities",
    "n_near_attractions",
    # -- live pool conditions: what the graph currently looks like -----------
    # Without these the policy cannot be condition-aware, which is the whole
    # point of a *learned* policy over a fixed vector: up-weighting disruption
    # is only sensible when there is disruption to avoid.
    "pool_disruption_mean",
    "pool_disruption_spread",
    "pool_price_missing_rate",
    "pool_event_active",
    "bias",
)

N_FEATURES = len(FEATURE_NAMES)


@dataclass
class PoolConditions:
    """Live state of the candidate pool, as seen at query time.

    Supplied by the caller because the retriever has already paid for this
    information; recomputing it inside the policy would double the graph round
    trip. All fields default to a neutral value so a caller that has not
    measured the pool still gets a usable weight vector.
    """
    disruption_mean: float = 0.0      # mean exposure in [0,1]; 0 = all calm
    disruption_spread: float = 0.0    # max - min exposure; 0 = no discrimination
    price_missing_rate: float = 0.0   # fraction of pool with no price
    event_active: float = 0.0         # 1.0 when any candidate is event-linked

    @classmethod
    def from_candidates(cls, cands: List[Dict[str, Any]]) -> "PoolConditions":
        if not cands:
            return cls()
        exposures: List[float] = []
        for c in cands:
            etas = [float(e) for e in (c.get("signal_etas") or []) if e]
            own = min(max(etas) / 20.0, 1.0) if etas else 0.0
            nbr = min(float(c.get("nbr_eta") or 0.0) / 20.0, 1.0)
            exposures.append(max(own, nbr))
        missing = sum(1 for c in cands if not c.get("price")) / len(cands)
        events = any(float(c.get("event_impact") or 0.0) > 0 for c in cands)
        return cls(
            disruption_mean=sum(exposures) / len(exposures),
            disruption_spread=max(exposures) - min(exposures),
            price_missing_rate=missing,
            event_active=1.0 if events else 0.0,
        )


def featurise(intent: QueryIntent,
              conditions: Optional[PoolConditions] = None) -> List[float]:
    """Context vector x for a query. Order matches FEATURE_NAMES exactly."""
    c = conditions or PoolConditions()
    sort_intent = intent.sort_intent or "best_overall"
    prox = intent.proximity_preference or "any"
    return [
        1.0 if sort_intent == "best_overall" else 0.0,
        1.0 if sort_intent == "cheapest" else 0.0,
        1.0 if sort_intent == "highest_rated" else 0.0,
        1.0 if sort_intent == "most_accessible" else 0.0,
        1.0 if intent.accessibility_priority == "high" else 0.0,
        1.0 if intent.avoid_traffic else 0.0,
        1.0 if prox == "close" else 0.0,
        1.0 if prox == "far" else 0.0,
        1.0 if prox == "any" else 0.0,
        1.0 if intent.max_price_lkr is not None else 0.0,
        1.0 if intent.min_price_lkr is not None else 0.0,
        1.0 if intent.min_rating is not None else 0.0,
        1.0 if intent.min_star is not None else 0.0,
        # Counts are squashed to [0,1] at 3 items: "some" vs "none" is the
        # signal, the exact count is noise.
        min(len(intent.required_amenities) / 3.0, 1.0),
        min(len(intent.near_attractions) / 3.0, 1.0),
        c.disruption_mean,
        c.disruption_spread,
        c.price_missing_rate,
        c.event_active,
        1.0,  # bias
    ]


# ---------------------------------------------------------------------------
# Policy interface + static policies
# ---------------------------------------------------------------------------

class WeightPolicy:
    """Maps query context to a weight vector.

    Implementations must be deterministic at inference time so an evaluation
    run is reproducible.
    """

    name = "base"

    def predict(self, intent: QueryIntent,
                conditions: Optional[PoolConditions] = None) -> ScoringWeights:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"policy": self.name}


class HandTunedPolicy(WeightPolicy):
    """The existing `weights_for_intent` ladder, wrapped as a policy.

    This is the incumbent the learned policy has to beat.
    """

    name = "handtuned"

    def __init__(self, profile: Optional[str] = None) -> None:
        self.profile = profile

    def predict(self, intent: QueryIntent,
                conditions: Optional[PoolConditions] = None) -> ScoringWeights:
        return weights_for_intent(intent, self.profile)

    def describe(self) -> Dict[str, Any]:
        return {"policy": self.name, "profile": self.profile or "config default",
                "learned": False}


class StaticProfilePolicy(WeightPolicy):
    """One fixed vector for every query — no intent adaptation at all.

    Included because it is the honest floor: if the learned policy cannot beat
    a constant, the context features are not carrying information.
    """

    def __init__(self, profile: str) -> None:
        if profile not in WEIGHT_PROFILES:
            raise KeyError(f"unknown weight profile {profile!r}")
        self.profile = profile
        self.name = f"static[{profile}]"

    def predict(self, intent: QueryIntent,
                conditions: Optional[PoolConditions] = None) -> ScoringWeights:
        return base_weights(self.profile).normalised()

    def describe(self) -> Dict[str, Any]:
        return {"policy": self.name, "profile": self.profile, "learned": False,
                "weights": base_weights(self.profile).normalised().to_dict()}


# ---------------------------------------------------------------------------
# Dirichlet policy network
# ---------------------------------------------------------------------------

def _torch():
    """Import torch on demand and fail with an actionable message."""
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The learned weight policy needs PyTorch. Install it with\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "or use WEIGHT_POLICY=handtuned."
        ) from exc


def build_network(n_features: int = N_FEATURES, n_components: int = len(COMPONENTS),
                  hidden: int = 64):
    """MLP producing Dirichlet concentration parameters.

    Two hidden layers of 64 is already generous for ~20 inputs and a few
    hundred training queries; anything larger memorises. Kept as a function so
    the trainer and the loader build byte-identical architectures.
    """
    torch = _torch()
    from torch import nn

    class DirichletPolicyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(n_features, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, n_components),
            )

        def alpha(self, x):
            # softplus + 1 keeps every concentration > 1, which makes the
            # Dirichlet unimodal. Without the +1 the density can spike at the
            # simplex corners and the policy collapses to "one component gets
            # everything" early in training, before it has learned anything.
            return torch.nn.functional.softplus(self.body(x)) + 1.0

        def forward(self, x):
            return self.alpha(x)

        def distribution(self, x):
            return torch.distributions.Dirichlet(self.alpha(x))

        def mean_weights(self, x):
            a = self.alpha(x)
            return a / a.sum(dim=-1, keepdim=True)

    return DirichletPolicyNet()


class LearnedPolicy(WeightPolicy):
    """Trained Dirichlet policy, evaluated at its distribution mean.

    Inference is the mean alpha/sum(alpha), not a sample: an evaluation run must
    be reproducible, and a sampled ranking would differ between runs.
    """

    name = "learned"

    def __init__(self, checkpoint: Path | str = DEFAULT_CHECKPOINT) -> None:
        torch = _torch()
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No weight-policy checkpoint at {self.checkpoint_path}. "
                "Train one with:\n"
                "    python scripts/train_weight_policy.py --folds 5"
            )
        blob = torch.load(self.checkpoint_path, map_location="cpu",
                          weights_only=False)
        self.meta: Dict[str, Any] = blob.get("meta", {})

        saved_features = self.meta.get("feature_names")
        if saved_features and list(saved_features) != list(FEATURE_NAMES):
            raise ValueError(
                "Checkpoint feature layout does not match the current "
                "FEATURE_NAMES. The context vector changed since training; "
                "retrain rather than silently mismatching columns.\n"
                f"  checkpoint: {list(saved_features)}\n"
                f"  current:    {list(FEATURE_NAMES)}"
            )

        self.net = build_network(
            n_features=self.meta.get("n_features", N_FEATURES),
            hidden=self.meta.get("hidden", 64),
        )
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        self._torch = torch

    def predict(self, intent: QueryIntent,
                conditions: Optional[PoolConditions] = None) -> ScoringWeights:
        torch = self._torch
        x = torch.tensor([featurise(intent, conditions)], dtype=torch.float32)
        with torch.no_grad():
            w = self.net.mean_weights(x)[0].tolist()
        return ScoringWeights(**dict(zip(COMPONENTS, w)))

    def describe(self) -> Dict[str, Any]:
        return {
            "policy": self.name,
            "learned": True,
            "checkpoint": str(self.checkpoint_path),
            **{k: v for k, v in self.meta.items() if k != "feature_names"},
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_policy(name: str, checkpoint: Path | str = DEFAULT_CHECKPOINT) -> WeightPolicy:
    """Resolve a policy by name.

    Names: "handtuned", "learned", or any key in WEIGHT_PROFILES
    ("handset" / "elicited" / "blended") for the fixed-vector policies.
    """
    key = (name or "handtuned").lower()
    if key == "handtuned":
        return HandTunedPolicy()
    if key == "learned":
        return LearnedPolicy(checkpoint)
    if key in WEIGHT_PROFILES:
        return StaticProfilePolicy(key)
    raise KeyError(
        f"unknown weight policy {name!r}; expected 'handtuned', 'learned', "
        f"or one of {sorted(WEIGHT_PROFILES)}"
    )


def save_checkpoint(net, path: Path | str, meta: Dict[str, Any]) -> None:
    """Persist a trained policy together with the metadata needed to trust it."""
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "feature_names": list(FEATURE_NAMES),
            "components": list(COMPONENTS), "n_features": N_FEATURES}
    torch.save({"state_dict": net.state_dict(), "meta": meta}, path)
    # A JSON sidecar so the training provenance is readable without torch.
    Path(str(path) + ".json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved weight policy to %s", path)
