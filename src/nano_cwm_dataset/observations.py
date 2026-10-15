"""Observation indexing, including the final observation of every key."""

from torch import Tensor

from nano_cwm_dataset.episode import Episode


def observation_at(episode: Episode, step: int) -> dict[str, Tensor]:
    """Read frame 0..T for each observation key. Frame T lives in metadata."""
    if type(step) is not int or not 0 <= step <= len(episode):
        raise ValueError("Observation index out of bounds")
    if step < len(episode):
        return {key: value[step] for key, value in episode.obs.items()}
    final = episode.info.get("final_observation")
    if not isinstance(final, dict):
        raise ValueError("Episode is missing its final_observation")
    return final
