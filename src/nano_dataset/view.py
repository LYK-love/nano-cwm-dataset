"""Read-only episode selection using the same standard Dataset API."""

from nano_dataset.dataset import Dataset
from nano_dataset.episode import Episode
from nano_dataset.observations import observation_at


class DatasetView(Dataset):
    """Expose selected episodes and optional prefixes, with local contiguous IDs."""

    def __init__(
        self, source: Dataset, episodes: tuple[int, ...], *, step_limit: int | None = None
    ) -> None:
        if len(set(episodes)) != len(episodes) or any(
            type(i) is not int or not 0 <= i < source.num_episodes for i in episodes
        ):
            raise ValueError("Select distinct valid episode IDs")
        if step_limit is not None and step_limit < 1:
            raise ValueError("step_limit must be positive")
        # A view is lazy and never opens another backing store.
        self._source = source
        self._episodes = episodes
        self._step_limit = step_limit
        self.is_static = True
        self.name = source.name
        self.revision = source.revision
        self.lengths = [
            min(source.lengths[i], step_limit) if step_limit is not None else source.lengths[i]
            for i in episodes
        ]

    def load_episode(self, episode_id: int) -> Episode:
        """Return the selected episode or prefix without changing environment end flags."""
        if type(episode_id) is not int or not 0 <= episode_id < self.num_episodes:
            raise ValueError("Episode ID out of bounds")
        episode = self._source.load_episode(self._episodes[episode_id])
        length = self.lengths[episode_id]
        if length == len(episode):
            return episode
        return Episode(
            {key: value[:length] for key, value in episode.obs.items()},
            episode.act[:length],
            episode.rew[:length],
            episode.end[:length],
            episode.trunc[:length],
            {"final_observation": observation_at(episode, length)},
        )

    def state_dict(self) -> dict[str, object]:
        """Identify the source and selection for reproducible evaluation suites."""
        return {
            "source": self._source.state_dict(),
            "episodes": self._episodes,
            "step_limit": self._step_limit,
        }

    def save_to_default_path(self) -> None:
        """Views have no independent persistent state."""
        raise ValueError("Cannot save a dataset view")

    def load_from_default_path(self) -> None:
        """Views obtain metadata from their source."""
        raise ValueError("Cannot reload a dataset view")

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Construct a new view to change its selection."""
        raise ValueError("Cannot replace dataset view state")
