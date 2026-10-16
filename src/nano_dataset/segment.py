"""DIAMOND sequence addressing and padded segments."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from nano_dataset.episode import Episode


@dataclass(frozen=True)
class SegmentId:
    """Identify an episode slice; stop is exclusive and padding may exceed its bounds."""

    episode_id: int
    start: int
    stop: int


@dataclass
class Segment:
    """A sequence with a validity mask; padded rows never count as evidence."""

    obs: dict[str, Tensor]
    act: Tensor
    rew: Tensor
    end: Tensor
    trunc: Tensor
    mask_padding: Tensor
    info: dict[str, Any]
    id: SegmentId

    @property
    def effective_size(self) -> int:
        """Number of real rows, excluding left and right padding."""
        return int(self.mask_padding.sum().item())


def make_segment(episode: Episode, segment_id: SegmentId, should_pad: bool = True) -> Segment:
    """Slice an episode with DIAMOND's zero-padding and clamped segment identity."""
    start, stop = segment_id.start, segment_id.stop
    if not start < len(episode) or not stop > 0 or start >= stop:
        raise ValueError("Segment must overlap the episode")
    left, right = max(0, -start), max(0, stop - len(episode))
    if not should_pad and (left or right):
        raise ValueError("Segment exceeds episode bounds")
    start, stop = max(0, start), min(len(episode), stop)

    def slice_tensor(value: Tensor) -> Tensor:
        selected = value[start:stop]
        return torch.cat(
            (
                value.new_zeros((left, *value.shape[1:])),
                selected,
                value.new_zeros((right, *value.shape[1:])),
            )
        )

    device = next(iter(episode.obs.values())).device
    mask = torch.zeros(left + stop - start + right, dtype=torch.bool, device=device)
    mask[left : left + stop - start] = True
    return Segment(
        obs={key: slice_tensor(value) for key, value in episode.obs.items()},
        act=slice_tensor(episode.act),
        rew=slice_tensor(episode.rew),
        end=slice_tensor(episode.end),
        trunc=slice_tensor(episode.trunc),
        mask_padding=mask,
        info=episode.info,
        id=SegmentId(segment_id.episode_id, start, stop),
    )
