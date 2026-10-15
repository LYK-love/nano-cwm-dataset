"""Storage, sequence selection and online collection across the DIAMOND boundary."""

from pathlib import Path

import pytest
import torch

from nano_cwm_dataset.collected import EpisodeCollector
from nano_cwm_dataset.dataset import Dataset
from nano_cwm_dataset.episode import Episode
from nano_cwm_dataset.observations import observation_at
from nano_cwm_dataset.segment import SegmentId
from nano_cwm_dataset.view import DatasetView


def episode(length: int = 3) -> Episode:
    """Create identifiable input frames and a distinct last observation."""
    frames = torch.arange(length + 1).float().reshape(-1, 1, 1, 1).expand(-1, 3, 2, 2) / 8
    return Episode(
        {"image": frames[:-1]},
        torch.arange(length),
        torch.arange(length).float(),
        torch.zeros(length, dtype=torch.bool),
        torch.zeros(length, dtype=torch.bool),
        {
            "final_observation": {"image": frames[-1]},
            "private_note": "never expose metadata through tools",
        },
    )


def test_persistence_growth_and_segment_padding(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path)
    first = episode()
    dataset.add_episode(first)
    dataset.add_episode(episode(2))
    loaded = Dataset(tmp_path)
    assert loaded.lengths == [3, 2] and len(loaded) == 5
    assert loaded.load_episode(0).act.tolist() == [0, 1, 2]
    assert torch.equal(
        observation_at(loaded.load_episode(0), 3)["image"],
        first.info["final_observation"]["image"],
    )
    segment = loaded[SegmentId(0, -1, 5)]
    assert segment.mask_padding.tolist() == [False, True, True, True, False, False]
    assert segment.effective_size == 3
    assert segment.act.tolist() == [0, 0, 1, 2, 0, 0]
    loaded.add_episode(episode(4), episode_id=0)
    assert Dataset(tmp_path).lengths == [4, 2]
    loaded.is_static = True
    with pytest.raises(ValueError, match="static"):
        loaded.add_episode(first)


def test_rgb_serialization_preserves_all_byte_values(tmp_path: Path) -> None:
    frame = torch.arange(256, dtype=torch.uint8).reshape(1, 16, 16).expand(3, -1, -1)
    normalized = frame.float().div(255).mul(2).sub(1)
    depth = torch.tensor([0.1])
    ep = Episode(
        {"image": normalized[None], "depth": depth},
        torch.tensor([0]),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        {"final_observation": {"image": normalized, "depth": torch.tensor(1.5)}},
    )
    ep.save(tmp_path / "episode.pt")
    loaded = Episode.load(tmp_path / "episode.pt")
    assert torch.equal(loaded.obs["image"], ep.obs["image"])
    assert torch.equal(loaded.obs["depth"], depth)


def test_view_preserves_prefix_boundary_and_source(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path, save_on_disk=False)
    dataset.add_episode(episode(4))
    dataset.add_episode(episode(3))
    view = DatasetView(dataset, (1, 0), step_limit=2)
    prefix = view.load_episode(0)
    assert view.lengths == [2, 2]
    assert torch.equal(observation_at(prefix, 2)["image"], dataset.load_episode(1).obs["image"][2])
    assert not prefix.dead.any()
    assert dataset.lengths == [4, 3]
    with pytest.raises(ValueError):
        view.load_episode(2)
    with pytest.raises(ValueError):
        observation_at(prefix, 3)


def test_collector_keeps_reset_and_final_frame_across_reopen(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path)
    collector = EpisodeCollector(dataset)
    reset = {"image": torch.zeros(3, 2, 2)}
    assert collector.record(None, reset) == (0, 0)
    assert torch.equal(observation_at(dataset.load_episode(0), 0)["image"], reset["image"])
    after = {"image": torch.ones(3, 2, 2)}
    collector.record(2, after, 3.0)
    reopened = Dataset(tmp_path)
    resumed = EpisodeCollector(reopened)
    resumed.record(1, {key: -frame for key, frame in after.items()}, 4.0, truncated=True)
    result = reopened.load_episode(0)
    assert result.act.tolist() == [2, 1] and result.rew.tolist() == [3, 4]
    assert torch.equal(result.obs["image"][1], after["image"])
    assert torch.equal(observation_at(result, 2)["image"], -after["image"])
    assert result.trunc.tolist() == [False, True]
    with pytest.raises(ValueError, match="Reset"):
        resumed.prepare(0)
    assert resumed.record(None, reset) == (1, 0)


def test_observation_keys_follow_the_environment() -> None:
    state = torch.zeros(2, 4)
    recorded = Episode(
        {"state": state},
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        {"final_observation": {"state": torch.ones(4)}},
    )
    assert "image" not in recorded.obs
    assert torch.equal(observation_at(recorded, 0)["state"], state[0])
    assert torch.equal(observation_at(recorded, 2)["state"], torch.ones(4))


def test_concatenating_prefixes_replaces_the_final_observation() -> None:
    first, second = episode(2), episode(3)
    first.info = {"final_observation": first.info["final_observation"]}
    second.info = {"final_observation": second.info["final_observation"]}
    combined = first + second
    assert len(combined) == 5
    assert torch.equal(
        observation_at(combined, 5)["image"], second.info["final_observation"]["image"]
    )
    assert combined.info["final_observation"]["image"].shape == (3, 2, 2)
    first.end[-1] = True
    with pytest.raises(ValueError, match="ended"):
        first + second
