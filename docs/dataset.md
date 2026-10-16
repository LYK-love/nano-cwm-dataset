# Dataset

A dataset is a directory of RL rollouts. The layout follows DIAMOND (see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)). This package stores, slices, and collects episodes. It does not sample batches, score predictions, or talk to an environment.

## Data Format

Episodes are stored at episode level. An episode with T actions has **T rows and T+1 frames**. T counts only actions after `reset`.

`obs` is a dict of tensors, one key per modality. Every tensor has length T. Keys depend on the environment. One environment may use `obs["image"]`. Another may have only `obs["state"]`, or add keys such as `obs["depth"]`.

- `reset` produces frame 0 only: `obs[key][0]`. It has no action row.
- Action `t` (`t = 0..T-1`) is applied to `obs[key][t]` and produces the next frame.
- The frame after the last action is not in `obs`. It is `info["final_observation"][key]`, frame `T`. `final_observation` has the same keys.

Each row `t` stores the action and its outcome:

| Field   | Row t                        | Example                                           |
| ------- | ---------------------------- | ------------------------------------------------- |
| `obs`   | One tensor per key, length T | `obs["image"]`: `(T, 3, 64, 64)` float32 `[-1, 1]` |
| `act`   | Action                       | `(T,)` int64                                      |
| `rew`   | Reward                       | `(T,)` float                                      |
| `end`   | Termination                  | `(T,)` bool                                       |
| `trunc` | Truncation                   | `(T,)` bool                                       |
| `info`  | Metadata                     | `final_observation["image"]`: `(3, 64, 64)`       |

## Structure

```text
Dataset
  Episode 0 .. n-1
    frames 0 .. T              T+1 frames
    actions 0 .. T-1           T actions; reset is not one of them
    Segment [start, stop)      a contiguous run of those actions
```

A `Dataset` is one directory of episodes. Episode IDs are `0 .. n-1`.

An `Episode` is one trajectory. `reset` writes frame 0 of every observation key, `obs[key][0]`. The T actions are `act`, `rew`, `end`, and `trunc`. Action `t` is applied to `obs[key][t]`. The frame it produces is `obs[key][t+1]` when `t+1 < T`, and `info["final_observation"][key]` when `t+1 = T`.

A `Segment` is a contiguous action range inside one episode: `SegmentId(episode_id, start, stop)`, with `stop` excluded. It does not cross into another episode. `step` is an integer index, not a separate type. Action indices are `0 .. T-1`. Frame indices are `0 .. T`.

## Storage

```text
dataset/
  info.pt
  000/00/0/0.pt
  000/00/1/1.pt
  100/20/3/123.pt
```

`info.pt` stores `lengths` and `revision`. Episode `n` is `{hundreds:03d}/{tens:02d}/{ones}/{n}.pt`, using the decimal groups of `n`. Episode 123 is `100/20/3/123.pt`.

On save, `obs["image"]` is quantized to uint8 and mapped back to float in `[-1, 1]` on load. `final_observation` and every other key are stored as supplied. Each save writes a temporary file, then replaces the target. `revision` increments on every `add_episode`, including a replacement of the same ID.

`Dataset(path)` loads `info.pt` when that file exists. `save_on_disk=False` keeps episodes in memory. `cache_in_ram=True` retains episodes after load.

`is_static = True` rejects `add_episode`. Returned tensors are still ordinary tensors; callers treat a static episode as read-only.

## Sampling

Sampling returns a `Segment`. The caller names the episode and the action range. `Dataset` has no random sampler.

`dataset[SegmentId(episode_id, start, stop)]` is that range. The request must overlap the episode.

Rows requested outside `[0, T)` are zeros. `mask_padding` is true on stored rows and false on the zeros. `effective_size` counts the true rows. The returned `segment.id` is the overlap clamped to the episode. For an episode of length 10, `SegmentId(id, 8, 12)` has four rows, mask `[True, True, False, False]`, and `segment.id` of `[8, 10)`.

`make_segment(episode, segment_id, should_pad=False)` rejects a request that extends past the episode.

`segment.info["final_observation"]` is the whole episode's endpoint. The frame after an in-bounds slice ending at `stop` is `observation_at(episode, stop)`.

```python
from nano_cwm_dataset.view import DatasetView

selected = DatasetView(training, episodes=(7, 2, 9), step_limit=8)
first = selected.load_episode(0)  # source episode 7, at most 8 actions
```

View IDs `0, 1, 2` are source IDs `7, 2, 9`. Source IDs are distinct and in range. `step_limit`, when set, is a positive action count. The frame at the cutoff becomes that prefix's `final_observation`. `end` and `trunc` are the stored flags, sliced to the prefix. A view reads through its source and has no directory of its own.

## API

```python
from nano_cwm_dataset.dataset import Dataset
from nano_cwm_dataset.observations import observation_at
from nano_cwm_dataset.segment import SegmentId
```

| Call | Result |
| --- | --- |
| `Dataset(path)` | Open a directory. Missing `info.pt` means an empty dataset |
| `num_episodes` | Episode count. IDs are `0 .. n-1` |
| `lengths[id]` | Action count, without loading tensors |
| `num_steps`, `len(dataset)` | Total action rows |
| `load_episode(id)` | One `Episode` |
| `dataset[SegmentId(id, start, stop)]` | One segment |
| `add_episode(episode)` | Append, save, and return the new ID |
| `add_episode(episode, episode_id=id)` | Replace that episode and save |
| `observation_at(episode, step)` | Frame `0..T` of every observation key |
| `EpisodeCollector(dataset).record(…)` | Record one reset or step |

`Episode` requires equal row counts for every `obs` tensor, `act`, `rew`, `end`, and `trunc`. `episode_a + episode_b` concatenates them when `episode_a` has not ended and both episodes use the same keys. The endpoint in the result comes from `episode_b`.

`EpisodeCollector` owns the active episode. `record(None, observation)` starts an empty episode whose endpoint is that observation. A later `record` appends the previous observation and the new outcome, then stores the returned frame as the endpoint under the same ID. The next reset allocates a new ID. A step before reset, or after `dead`, is rejected.
