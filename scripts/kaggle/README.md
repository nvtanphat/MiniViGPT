# Running MiniViGPT on Kaggle

The notebook in this folder is the supported way to train on Kaggle GPUs.
It replaced an earlier single-file "bundle" script that packed the whole
package into one base64 blob; that approach broke in three separate ways
(see Gotchas) and is no longer used.

## Layout

Two datasets, one notebook:

| Dataset | Contents | Built from |
|---|---|---|
| `<user>/minivigpt-src` | `minivigpt/*.py` + `config.yaml` | `src/minivigpt/`, `configs/minivigpt_20m.yaml` |
| `<user>/minivigpt-vi-tokens-16k` | `tokenizer.json`, `{train,validation,test}.bin` + sidecar `.json` | `artifacts/local/` |

The notebook copies the package into `/kaggle/working`, points
`dataset.prepared_dir` at the corpus, and calls `minivigpt.train.train()`.
Shipping the pre-tokenized corpus skips roughly 30 minutes of streaming and
tokenizing per run, and guarantees every run uses the same sha256-verified data.

## Publishing

```bash
# 1. source + config
mkdir -p /tmp/src && cp -r src/minivigpt /tmp/src/ && \
  cp configs/minivigpt_20m.yaml /tmp/src/config.yaml
rm -rf /tmp/src/minivigpt/__pycache__
# dataset-metadata.json: {"title": "MiniViGPT Source", "id": "<user>/minivigpt-src", ...}
kaggle datasets create  -p /tmp/src --dir-mode zip     # first time
kaggle datasets version -p /tmp/src --dir-mode zip -m "update"

# 2. corpus (once; ~239 MB)
kaggle datasets create -p artifacts/local

# 3. notebook
kaggle kernels push -p scripts/kaggle --accelerator NvidiaTeslaT4
kaggle kernels status <user>/minivigpt-train
```

`--dir-mode zip` matters: without it the CLI prints
`Skipping folder: minivigpt` and uploads nothing but the loose files.
Kaggle unzips the archive on its side, so the mounted layout still has
`minivigpt/*.py` as real files.

## Gotchas

**Ask for a specific GPU.** Without `"machine_shape": "NvidiaTeslaT4"` Kaggle
picks for you, and a P100 (`sm_60`) is no longer supported by the PyTorch build
in the Kaggle image — every CUDA op dies with `no kernel image is available for
execution on the device`. Cell 3 checks `get_device_capability()` against
`torch.cuda.get_arch_list()` and stops early with a readable message.

**Datasets are not mounted one directory per dataset.** The real layout is
`/kaggle/input/datasets/<user>/<slug>/...`, so `glob("*/train.bin")` finds
nothing. The notebook uses `rglob` and locates things by content
(`minivigpt/train.py`, `config.yaml`, `train.bin`, `checkpoint_latest.pt`)
rather than by path shape.

**A script kernel uploads only `code_file`.** Sibling files such as
`config.yaml` never reach the kernel. A notebook kernel plus a code dataset
sidesteps this entirely.

**`/kaggle/src` is read-only.** Anything the run writes must go under
`/kaggle/working`.

**Push files must be pure ASCII.** The CLI reads them as cp1252 and dies on
Vietnamese text with `'charmap' codec can't decode byte`. The notebook keeps
prompts as `\uXXXX` escapes.

## Resuming

Kaggle caps a session at 9 hours. To continue a run, save the previous
output as a dataset and attach it; the notebook picks up any
`checkpoint_latest.pt` it finds under `/kaggle/input` and resumes from it.
