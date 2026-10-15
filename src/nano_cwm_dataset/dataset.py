"""General-purpose DIAMOND episode storage and segment lookup.

No tool, environment, split or scoring policy belongs here. Each directory is an
independent dataset. This typed adaptation keeps DIAMOND's episode paths and .pt
payloads, with an automatically saved index instead of trainer-managed state.
"""

from pathlib import Path
from typing import Any

import torch

from nano_cwm_dataset.episode import Episode
from nano_cwm_dataset.segment import Segment, SegmentId, make_segment


class Dataset(torch.utils.data.Dataset[Segment]):
    """Store episodes, update growing trajectories, and retrieve selected segments."""

    def __init__(
        self,
        directory: Path,
        name: str | None = None,
        *,
        cache_in_ram: bool = False,
        save_on_disk: bool = True,
    ) -> None:
        self._directory = Path(directory)
        self.name = name or self._directory.name
        self._cache_in_ram = cache_in_ram or not save_on_disk
        self._save_on_disk = save_on_disk
        self._cache: dict[int, Episode] = {}
        self.is_static = False
        self.lengths: list[int] = []
        self.revision = 0
        self.load_from_default_path()

    @property
    def num_episodes(self) -> int:
        """Number of addressable episodes, with IDs in range(num_episodes)."""
        return len(self.lengths)

    @property
    def num_steps(self) -> int:
        """Number of stored transitions."""
        return sum(self.lengths)

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        return make_segment(self.load_episode(segment_id.episode_id), segment_id)

    def load_episode(self, episode_id: int) -> Episode:
        """Load one episode by ID, optionally reusing an in-memory cache."""
        if type(episode_id) is not int or not 0 <= episode_id < self.num_episodes:
            raise ValueError("Episode ID out of bounds")
        if episode_id not in self._cache:
            episode = Episode.load(self._get_episode_path(episode_id))
            if len(episode) != self.lengths[episode_id]:
                raise ValueError("Episode length differs from the dataset index")
            if self._cache_in_ram:
                self._cache[episode_id] = episode
            return episode
        return self._cache[episode_id]

    def add_episode(self, episode: Episode, *, episode_id: int | None = None) -> int:
        """Append a trajectory or replace an existing growing episode, then save its index."""
        self.assert_not_static()
        if episode_id is None:
            episode_id = self.num_episodes
        elif type(episode_id) is not int or not 0 <= episode_id < self.num_episodes:
            raise ValueError("Episode ID out of bounds")
        episode = episode.to("cpu")
        if self._save_on_disk:
            episode.save(self._get_episode_path(episode_id))
        if episode_id == self.num_episodes:
            self.lengths.append(len(episode))
        else:
            self.lengths[episode_id] = len(episode)
        if self._cache_in_ram:
            self._cache[episode_id] = episode
        self.revision += 1
        if self._save_on_disk:
            self.save_to_default_path()
        return episode_id

    def _get_episode_path(self, episode_id: int) -> Path:
        digits = [episode_id % 1000 // 100 * 100, episode_id % 100 // 10 * 10, episode_id % 10]
        return self._directory / f"{digits[0]:03d}/{digits[1]:02d}/{digits[2]:01d}/{episode_id}.pt"

    def assert_not_static(self) -> None:
        """Reject writes to a researcher-supplied static dataset."""
        if self.is_static:
            raise ValueError("Trying to modify a static dataset")

    def state_dict(self) -> dict[str, Any]:
        """Return portable dataset bookkeeping, with no cached episode tensors."""
        return {"lengths": self.lengths.copy(), "revision": self.revision}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore dataset bookkeeping and invalidate cached episodes."""
        lengths = state["lengths"]
        if any(type(n) is not int or n < 0 for n in lengths):
            raise ValueError("Episode lengths must be nonnegative integers")
        self.lengths = list(lengths)
        self.revision = int(state["revision"])
        self._cache.clear()

    def save_to_default_path(self) -> None:
        """Atomically save dataset metadata to info.pt."""
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = self._directory / "info.tmp"
        torch.save(self.state_dict(), temporary)
        temporary.replace(self._directory / "info.pt")

    def load_from_default_path(self) -> None:
        """Load info.pt when reopening an existing dataset."""
        path = self._directory / "info.pt"
        if path.is_file():
            self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
