"""Episode records. Observations are a dict of tensors, one key per modality.

Each observation tensor has T rows. Row t is the observation before action t.
The observation after the last action is info["final_observation"], with the
same keys and one frame each. obs["image"] is packed as bytes on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass
class Episode:
    """One trajectory. obs keys are the environment's observation space."""

    obs: dict[str, Tensor]
    act: Tensor
    rew: Tensor
    end: Tensor
    trunc: Tensor
    info: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.obs or any(type(key) is not str or not key for key in self.obs):
            raise ValueError("Observation keys must be nonempty strings")
        if any(not isinstance(value, Tensor) or value.ndim < 1 for value in self.obs.values()):
            raise ValueError("Each observation must be a tensor with a time dimension")
        if len({len(value) for value in self.obs.values()}) != 1:
            raise ValueError("Observation keys must share one length")
        if any(len(x) != len(self) for x in (self.act, self.rew, self.end, self.trunc)):
            raise ValueError("Episode fields must have equal lengths")
        if any(x.ndim != 1 for x in (self.rew, self.end, self.trunc)):
            raise ValueError("Rewards and outcome flags must be one-dimensional")
        if len(self) > 1 and bool(self.dead[:-1].any()):
            raise ValueError("An episode cannot contain transitions after an end")
        final = self.info.get("final_observation")
        if not isinstance(final, dict) or set(final) != set(self.obs):
            raise ValueError("final_observation must use the same observation keys")
        for key, rows in self.obs.items():
            frame = final[key]
            if not isinstance(frame, Tensor) or tuple(frame.shape) != tuple(rows.shape[1:]):
                raise ValueError("final_observation shape must match one observation frame")

    def __len__(self) -> int:
        return len(next(iter(self.obs.values())))

    @property
    def dead(self) -> Tensor:
        """Whether each transition ends the episode, including time limits."""
        return self.end.bool() | self.trunc.bool()

    def __add__(self, other: Episode) -> Episode:
        if bool(self.dead.any()):
            raise ValueError("Cannot extend an ended episode")
        if set(self.obs) != set(other.obs):
            raise ValueError("Observation keys must match")
        obs = {key: torch.cat((self.obs[key], other.obs[key]), dim=0) for key in self.obs}
        return Episode(
            obs,
            torch.cat((self.act, other.act), dim=0),
            torch.cat((self.rew, other.rew), dim=0),
            torch.cat((self.end, other.end), dim=0),
            torch.cat((self.trunc, other.trunc), dim=0),
            merge_info(self.info, other.info),
        )

    def to(self, device: torch.device | str) -> Episode:
        """Move trajectory tensors; info stays as supplied."""
        return Episode(
            {key: value.to(device) for key, value in self.obs.items()},
            self.act.to(device),
            self.rew.to(device),
            self.end.to(device),
            self.trunc.to(device),
            self.info,
        )

    def compute_metrics(self) -> dict[str, int | float]:
        """Return episode length and undiscounted return."""
        return {"length": len(self), "return": float(self.rew.sum().item())}

    @classmethod
    def load(cls, path: Path, map_location: torch.device | str = "cpu") -> Episode:
        """Load tensor-only episode data without allowing arbitrary pickle objects."""
        data = torch.load(path, map_location=map_location, weights_only=True)
        data["obs"] = _unpack_observations(data["obs"])
        return cls(**data)

    def save(self, path: Path) -> None:
        """Atomically save an episode, packing obs['image'] as bytes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["obs"] = _pack_observations(self.obs)
        temporary = path.with_suffix(".tmp")
        torch.save(data, temporary)
        temporary.replace(path)


def _pack_observations(obs: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: _pack_image(value) if key == "image" else value for key, value in obs.items()}


def _unpack_observations(obs: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        key: value.float().div(255).mul(2).sub(1) if key == "image" else value
        for key, value in obs.items()
    }


def _pack_image(value: Tensor) -> Tensor:
    return value.add(1).div(2).mul(255).round().clamp(0, 255).byte()


def merge_info(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge episode metadata, concatenating shared tensor entries."""
    # A collected prefix may already carry its endpoint. Extending that prefix
    # replaces the endpoint; it is not a time-series field to concatenate.
    left = {key: value for key, value in left.items() if key != "final_observation"}
    return {
        **left,
        **right,
        **{key: torch.cat((left[key], right[key]), dim=0) for key in left.keys() & right.keys()},
    }
