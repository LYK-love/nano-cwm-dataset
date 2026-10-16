"""Collect public online transitions into ordinary DIAMOND episodes."""

import torch
from torch import Tensor

from nano_dataset.dataset import Dataset
from nano_dataset.episode import Episode


class EpisodeCollector:
    """Own the current episode and its latest observation; the dataset owns storage."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self._episode_id: int | None = None
        self._episode: Episode | None = None
        self._observation: dict[str, Tensor] | None = None
        if dataset.num_episodes:
            self._episode_id = dataset.num_episodes - 1
            self._episode = dataset.load_episode(self._episode_id)
            self._observation = self._episode.info["final_observation"]

    def prepare(self, action: int | None) -> None:
        """Reject stepping before reset or after an episode has ended."""
        if action is not None and (self._episode is None or bool(self._episode.dead.any())):
            raise ValueError("Reset before stepping a new or ended episode")

    def record(
        self,
        action: int | None,
        observation: dict[str, Tensor],
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
    ) -> tuple[int, int]:
        """Record a successful reset/step, retaining the last observation even for prefixes."""
        self.prepare(action)
        if action is None:
            episode = Episode(
                {key: frame.new_empty((0, *frame.shape)) for key, frame in observation.items()},
                torch.empty(0, dtype=torch.long),
                torch.empty(0),
                torch.empty(0, dtype=torch.bool),
                torch.empty(0, dtype=torch.bool),
                {"final_observation": observation},
            )
            episode_id = self.dataset.add_episode(episode)
        else:
            assert self._episode is not None and self._observation is not None
            episode = Episode(
                {
                    key: torch.cat((self._episode.obs[key], frame.unsqueeze(0)))
                    for key, frame in self._observation.items()
                },
                torch.cat((self._episode.act, torch.tensor([action]))),
                torch.cat((self._episode.rew, torch.tensor([reward]))),
                torch.cat((self._episode.end, torch.tensor([terminated]))),
                torch.cat((self._episode.trunc, torch.tensor([truncated]))),
                {"final_observation": observation},
            )
            assert self._episode_id is not None
            episode_id = self.dataset.add_episode(episode, episode_id=self._episode_id)
        self._episode, self._episode_id, self._observation = episode, episode_id, observation
        return episode_id, len(episode)
