# nano-cwm-dataset
> You can use AI to translate or explain this document and the rest of the project's documentation in your preferred language.
>
> 你可以使用 AI 将本文档和本项目的其他文档翻译成你偏好的语言，或为你解读其中的内容。

Episode storage for reinforcement-learning rollouts. One directory is one dataset. An episode with T actions stores T rows and T+1 observation frames.

`obs` is a dict of tensors, one key per modality. Read frame `t` of a modality with `obs["image"][t]`. Keys depend on the environment. An environment may have only `obs["state"]`, or add keys such as `obs["depth"]`.

This package stores, slices, and collects episodes. It does not sample batches, score predictions, or talk to an environment.

## Install

Python 3.12 or newer.

```bash
uv add --editable /path/to/nano-cwm-dataset
```

## Use

```python
import torch
from pathlib import Path

from nano_cwm_dataset.dataset import Dataset
from nano_cwm_dataset.episode import Episode

frame = torch.zeros(3, 64, 64)
episode = Episode(
    {"image": frame.unsqueeze(0)},
    torch.zeros(1, dtype=torch.long),
    torch.zeros(1),
    torch.zeros(1, dtype=torch.bool),
    torch.zeros(1, dtype=torch.bool),
    {"final_observation": {"image": frame}},
)
dataset = Dataset(Path("rollouts"))
dataset.add_episode(episode)
```

Import concrete modules. The package initializer does not re-export them.

`obs["image"]`, when present, is stored as uint8 and loaded back to float in `[-1, 1]`. Every other key, and `info["final_observation"]`, is stored as supplied. Format, storage, sampling, and the package API are in [the dataset document](docs/dataset.md).

This project was written collaboratively by humans and AI.
