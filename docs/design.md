# Design

## Problem

A rollout dataset has to store the observation that an action was applied to, and the observation that action produced. Different environments expose different observation spaces. One environment may store an image. Another may store only a state vector, or add a depth map beside the image.

## Approach

Each dataset is a directory of episode files plus an `info.pt` index. An observation is a dict of tensors that share one time length. The caller chooses the keys. The library checks that the keys are nonempty strings, that the tensors share a length, and that `final_observation` uses the same keys with one frame each.

## Components

```text
Dataset
  Episode 0 .. n-1
    obs[key][0 .. T)         frames before each action
    act, rew, end, trunc     T action rows
    final_observation[key]   frame T, after the last action
  Segment [start, stop)      one contiguous action range
DatasetView                  selected episodes, optional action prefix
EpisodeCollector             reset and step into the current episode
```

`Dataset` owns the directory. `Episode` is one trajectory. `Segment` is a slice, with zero padding outside the episode and `mask_padding` marking the real rows. `DatasetView` renumbers a selection and does not write. `EpisodeCollector` turns a reset and later steps into ordinary episodes.

## Interaction

A writer builds an `Episode` and calls `add_episode`. A reader opens the same directory, loads an episode, and calls `observation_at(episode, step)` for frame `0 .. T`. `dataset[SegmentId(episode_id, start, stop)]` returns that action range. Replacing an episode uses the same id and increments `revision`.

## Details

T counts actions after reset. Reset is not an action. Action `t` is applied to `obs[key][t]`. The next frame is `obs[key][t + 1]` when `t + 1 < T`, and `final_observation[key]` when `t + 1 = T`. A length-zero episode stores the reset frame only as `final_observation`.

On disk, episode `n` is `{hundreds:03d}/{tens:02d}/{ones}/{n}.pt`. Saves write a temporary file and then replace the target. Only `obs["image"]` is quantized to uint8. `final_observation` stays as supplied, so a prefix endpoint is not passed through the byte grid.

A segment requested past the episode is zero-padded on the overflow side. `should_pad=False` rejects that overflow. The returned segment id is the overlap clamped to the episode. `segment.info["final_observation"]` remains the whole episode's endpoint.

## Non-goals

No random sampler, no loss, no environment client, and no training-read accounting. Those belong to the program that uses this package.
