"""PyTorch dataset over the labeller's export folder.

    ds = DocumentDataset("data/export", split="train", processor=DatasetProcessor(preset="structured"))
    ds[0]   # {"id", "task", "prompt": str, "images": [PIL.Image, ...], "label": str}

It reads the canonical `{split}.jsonl` (one record per document) and builds each example on the fly
with the processor, so the task, prompt wording, schema format and output format change every epoch
(call `ds.set_epoch(epoch)`). Validation and test splits are drawn once (epoch 0) so scores stay
comparable between runs. Images are opened lazily and are always a list, one per page.

An item is deliberately raw. Tokenising and label masking happen in the loader's collate step, via
`make_loader(..., transform=create_transform(...))`, which runs the transform on each item and pads the
batch with the transform's own collator (`transform.make_collator()`). Without a transform the loader just collects lists.

Distributed training (DDP and FSDP): `make_loader` detects an initialised process group (torchrun) or
takes `rank` / `world_size`, and gives every rank its own equal shard, so all ranks run the same number
of steps (a requirement for DDP gradient sync and FSDP collectives). `batch_size` is per rank. The
epoch travels with each index, so `set_epoch(loader, epoch)` reshuffles the shards and re-draws the
prompts correctly even with persistent or multi-process workers.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch.distributed as dist

from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, Sampler, SequentialSampler

from .processors import BaseCollator, BaseTransform, DatasetProcessor, NoFittingExample, SequenceTooLong


class DocumentDataset(Dataset):
    def __init__(
            self,
            root: str | Path, split: str = "train",
            processor: DatasetProcessor | None = None,
            load_images: bool = True,
            languages: set[str] | None = None,
            max_documents: int | None = None,
            samples_per_document: int = 1,
            seed: int = 0,
            redraw_each_epoch: bool | None = None,
            fits=None
        ):
        """fits(page_sizes, prompt, label) -> bool: when given (make_loader passes the transform's), a drawn
        example that would not fit the length limit is never returned. Another task / format is drawn for
        the same document, and when nothing of the document fits the next document is used. Nothing
        raises, batches keep their full static shape, and the long examples simply drop out of the
        mix. `dropped_examples` / `skipped_documents` count what was left out (per process)."""
        self.root = Path(root)
        self.processor = processor or DatasetProcessor()
        self.load_images = load_images
        self.samples_per_document = samples_per_document
        self.seed = seed
        self.epoch = 0
        self.redraw = (split == "train") if redraw_each_epoch is None else redraw_each_epoch
        self.fits = fits
        self.dropped_examples = 0
        self.skipped_documents = 0
        self._sizes: dict[str, tuple[int, int]] = {}
        path = self.root / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `labeller export` first "
                                    "(add --include-unverified for unverified licences).")
        with path.open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if languages:
            self.records = [r for r in self.records if r.get("language") in languages]
        if max_documents is not None:
            self.records = self.records[:max_documents]

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_document

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int | tuple[int, int]) -> dict:
        epoch, index = index if isinstance(index, tuple) else (self.epoch, index)   # samplers pass (epoch, index)
        example = self._draw(epoch if self.redraw else 0, index)
        prompt = next(part["text"] for part in reversed(example["messages"][0]["content"]) if part["type"] == "text")
        images = [self._open(rel) for rel in example["images"]] if self.load_images else example["images"]
        return {"id": example["id"], "task": example["task"], "prompt": prompt, "images": images,
                "label": example["messages"][-1]["content"]}

    def _draw(self, epoch: int, index: int) -> dict:
        for attempt in range(len(self.records)):
            record = self.records[(index + attempt) % len(self.records)]
            try:
                return self.processor.sample(record, epoch=epoch, index=index, seed=self.seed,
                                             accept=self._accept if self.fits else None)
            except NoFittingExample:
                self.skipped_documents += 1
        raise RuntimeError("No example of any document is usable: none fits the length limit (raise max_length "
                           "or lower the image resolution) or no enabled task applies to them.")

    def _accept(self, example: dict) -> bool:
        sizes = []
        for rel in example["images"]:
            if rel not in self._sizes:
                with Image.open(self.root / rel) as image:      # reads the header only, no decoding
                    self._sizes[rel] = image.size
            sizes.append(self._sizes[rel])
        prompt = next(p["text"] for p in reversed(example["messages"][0]["content"]) if p["type"] == "text")
        ok = self.fits(sizes, prompt, example["messages"][-1]["content"])
        self.dropped_examples += 0 if ok else 1
        return ok

    def _open(self, rel: str) -> Image.Image:
        with Image.open(self.root / rel) as image:
            return image.convert("RGB")


def collate(batch: list[dict]) -> dict[str, list]:
    """Keeps variable-sized images as lists, one entry per sample."""
    return {key: [sample.get(key) for sample in batch] for key in batch[0]}


class TransformCollator:
    """Runs the transform on every item of a batch, then pads the batch. Living in the collate step
    keeps `__getitem__` raw and moves tokenisation into DataLoader workers."""

    def __init__(self, transform: BaseTransform, collator: BaseCollator | None = None,
                 on_too_long: str = "raise"):
        """Last line of defence for an example that exceeds the transform's max_length. "raise" stops;
        "replace" repeats another example of the same batch so the batch keeps its static shape. With
        make_loader the dataset already leaves such examples out, so this should never trigger."""
        if on_too_long not in ("raise", "replace"):
            raise ValueError("on_too_long must be 'raise' or 'replace'")
        self.transform = transform
        self.collator = collator or transform.make_collator()
        self.on_too_long = on_too_long
        self.replaced = 0   # per process; with num_workers > 0 each worker counts its own

    def __call__(self, batch: list[dict]) -> dict:
        encoded: list[dict | None] = []
        for item in batch:
            try:
                encoded.append(self.transform(item["images"], item["prompt"], item["label"]))
            except SequenceTooLong:
                if self.on_too_long == "raise":
                    raise
                encoded.append(None)
        good = [e for e in encoded if e is not None]
        if not good:
            raise SequenceTooLong(f"Every example in the batch exceeds max_length={self.transform.max_length}.")
        self.replaced += len(encoded) - len(good)
        return self.collator([e if e is not None else good[i % len(good)] for i, e in enumerate(encoded)])


class EpochSampler(Sampler):
    """Wraps any sampler and yields (epoch, index), so the dataset knows the epoch even inside worker
    processes. `set_epoch` also reshuffles a DistributedSampler."""

    def __init__(self, base: Sampler):
        self.base = base
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __iter__(self):
        for index in self.base:
            yield self.epoch, index

    def __len__(self) -> int:
        return len(self.base)


def world_info(rank: int | None = None, world_size: int | None = None) -> tuple[int, int]:
    """(rank, world_size) from the arguments, else from an initialised process group, else (0, 1)."""
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if rank is None:
        rank = dist.get_rank() if world_size > 1 and dist.is_available() and dist.is_initialized() else 0
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    return rank, world_size


def set_epoch(loader: DataLoader, epoch: int) -> None:
    """Call at the start of every epoch, on every rank."""
    loader.dataset.set_epoch(epoch)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)


def make_loader(root: str | Path, split: str = "train", processor: DatasetProcessor | None = None,
                batch_size: int = 4, shuffle: bool | None = None, num_workers: int = 0,
                transform: BaseTransform | None = None, collate_fn=None, on_too_long: str = "drop",
                rank: int | None = None, world_size: int | None = None, drop_last: bool | None = None,
                pin_memory: bool = False, persistent_workers: bool = False, **dataset_kwargs) -> DataLoader:
    """DataLoader for single-process, DDP or FSDP training. `batch_size` is per rank.

    Under torchrun (process group initialised) every rank gets a disjoint, equally sized shard; pass
    `rank` / `world_size` to shard without a process group. Training drops the incomplete last batch by
    default so every rank sees the same number of identically sized steps. For evaluation the shards
    are padded by repeating a few samples so ranks stay in step: gather and de-duplicate by `id`."""
    if on_too_long not in ("drop", "raise", "replace"):
        raise ValueError("on_too_long must be 'drop', 'raise' or 'replace'")
    if on_too_long == "drop" and transform is not None:
        dataset_kwargs.setdefault("fits", transform.fits)   # over-long examples are never drawn
    dataset = DocumentDataset(root, split=split, processor=processor, **dataset_kwargs)
    shuffle = (split == "train") if shuffle is None else shuffle
    rank, world_size = world_info(rank, world_size)
    if world_size > 1:
        base = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=dataset.seed)
    else:
        base = RandomSampler(dataset, generator=_seeded(dataset.seed)) if shuffle else SequentialSampler(dataset)
    sampler = EpochSampler(base)
    if collate_fn is None:
        collate_fn = (TransformCollator(transform, on_too_long="replace" if on_too_long == "drop" else on_too_long)
                      if transform else collate)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      collate_fn=collate_fn, pin_memory=pin_memory,
                      drop_last=(split == "train") if drop_last is None else drop_last,
                      persistent_workers=persistent_workers and num_workers > 0)


def _seeded(seed: int):
    import torch
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
