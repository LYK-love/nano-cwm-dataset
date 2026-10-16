# DIAMOND

The episode, dataset, and segment implementation in `src/nano_dataset/`
is a typed adaptation of [DIAMOND](https://github.com/eloialonso/diamond),
by Eloi Alonso. It retains DIAMOND's field alignment, per-episode tensor files,
segment addressing and padding conventions. Observations are a dict of
tensors, one key per modality. The index is saved automatically;
GPU training utilities and multiprocessing caches are not included.

Source files: `src/data/episode.py`, `dataset.py`, `segment.py`, `utils.py`.

```
MIT License

Copyright (c) 2024 Eloi Alonso

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
