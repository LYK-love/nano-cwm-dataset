# Workflows

## Save and reopen

`Dataset(path)` creates an empty dataset when `info.pt` is absent, and reopens the index when it exists. `add_episode` appends. `add_episode(episode, episode_id=id)` replaces that episode. `save_on_disk=False` keeps episodes in memory. `cache_in_ram=True` retains loaded episodes. `is_static = True` makes the next `add_episode` fail.

## Read one frame

```python
from nano_cwm_dataset.observations import observation_at

frame = observation_at(episode, step)["image"]
```

`step` may be `T`. That call returns `final_observation`. A missing key is absent from the dict. Callers that need an image check for the `image` key themselves.

## Slice actions

```python
from nano_cwm_dataset.segment import SegmentId

segment = dataset[SegmentId(0, 8, 12)]
```

For an episode of length 10 this returns four rows. `mask_padding` is `[True, True, False, False]`. `segment.id` is the clamped range `[8, 10)`.

`DatasetView(dataset, (7, 2, 9), step_limit=8)` exposes those source episodes as local ids `0, 1, 2`. A prefix's `final_observation` is the frame at the cutoff.

## Collect online steps

```python
from nano_cwm_dataset.collected import EpisodeCollector

collector = EpisodeCollector(dataset)
collector.record(None, {"state": reset_state})
collector.record(action, {"state": next_state}, reward, terminated, truncated)
```

`record(None, observation)` starts an empty episode whose endpoint is that observation. A later `record` appends the previous observation as a row and stores the new observation as the endpoint, under the same id. The next reset allocates a new id. A step before reset, or after `dead`, is rejected. Every record in one episode must use the same keys.
