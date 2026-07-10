from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from tokeniser import ProteinTokeniser, RESIDUE_TOKENS


DATASET_DIR = Path("dataset_PLM")
FASTA_PATH = DATASET_DIR / "uniref50.fasta.gz"
PROCESSED_DIR = DATASET_DIR / "processed"

CONTEXT_LENGTH = 512
MAX_RESIDUES = CONTEXT_LENGTH - 1

DEFAULT_MASK_PROB = 0.15
DEFAULT_BATCH_SIZE = 8
DEFAULT_EVAL_SEED = 42
DEFAULT_EVAL_STRIDE = 256


tokeniser = ProteinTokeniser()

# Residue IDs only, no [PAD], [MASK], or [CLS]. Used for the 10% random MLM replacement.
residue_ids = []
for tok in RESIDUE_TOKENS:
    residue_ids.append(tokeniser.token_to_id[tok])


class ProteinCache:
    def __init__(self, processed_dir: Path | str = PROCESSED_DIR):
        self.processed_dir = Path(processed_dir)

        tokens_path = self.processed_dir / "tokens.bin"
        offsets_path = self.processed_dir / "offsets.npy"
        lengths_path = self.processed_dir / "lengths.npy"
        train_indices_path = self.processed_dir / "train_indices.npy"
        eval_indices_path = self.processed_dir / "eval_indices.npy"
        meta_path = self.processed_dir / "meta.json"

        required_paths = [
            tokens_path,
            offsets_path,
            lengths_path,
            train_indices_path,
            eval_indices_path,
            meta_path,
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"missing processed cache file: {path}")

        self.tokens = np.memmap(tokens_path, dtype=np.uint8, mode="r")
        self.offsets = np.load(offsets_path)
        self.lengths = np.load(lengths_path)
        self.train_indices = np.load(train_indices_path)
        self.eval_indices = np.load(eval_indices_path)

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

    def __len__(self) -> int:
        return len(self.lengths)

    def get_residue_ids(self, protein_index: int) -> np.ndarray:
        start = int(self.offsets[protein_index])
        length = int(self.lengths[protein_index])
        end = start + length
        return self.tokens[start:end]


def pad_ids(
    ids: list[int],
    context_length: int = CONTEXT_LENGTH,
    pad_id: int | None = None,
) -> list[int]:
    if pad_id is None:
        pad_id = tokeniser.pad_id

    if len(ids) > context_length:
        raise ValueError(f"ids length {len(ids)} > context length {context_length}")

    num_pad = context_length - len(ids)
    return ids + [pad_id] * num_pad


def make_pad_mask(ids: list[int], pad_id: int | None = None) -> list[int]:
    if pad_id is None:
        pad_id = tokeniser.pad_id

    mask = []
    for token_id in ids:
        if token_id == pad_id:
            mask.append(0)
        else:
            mask.append(1)

    return mask


def select_15_percent(
    eligible_positions: list[int],
    mask_prob: float = DEFAULT_MASK_PROB,
    rng: random.Random | None = None,
) -> list[int]:
    if rng is None:
        rng = random

    selected = []
    for pos in eligible_positions:
        if rng.random() < mask_prob:
            selected.append(pos)

    # for short proteins batches if there are real residues, mask at least one.
    if len(selected) == 0 and len(eligible_positions) > 0:
        selected.append(rng.choice(eligible_positions))

    return selected


def MLM_mask(
    ids: list[int],
    mask_prob: float = DEFAULT_MASK_PROB,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Returns:
        input_ids: input IDs
        labels: original token IDs at MLM positions and -100 for ignore index elsewhere
        mlm_mask: 1 at selected MLM positions and 0 elsewhere
        selected: selected positions
    """
    if rng is None:
        rng = random

    input_ids = ids.copy()
    labels = [-100] * len(ids)
    mlm_mask = [0] * len(ids)

    eligible_positions = []
    for i, token_id in enumerate(ids):
        if token_id == tokeniser.pad_id:
            continue
        if token_id == tokeniser.cls_id:
            continue
        if token_id == tokeniser.mask_id:
            continue
        eligible_positions.append(i)

    selected = select_15_percent(eligible_positions, mask_prob=mask_prob, rng=rng)

    for pos in selected:
        labels[pos] = ids[pos]
        mlm_mask[pos] = 1
        r = rng.random()

        if r < 0.8:
            input_ids[pos] = tokeniser.mask_id
        elif r < 0.9:
            input_ids[pos] = rng.choice(residue_ids)
        else:
            continue

    return input_ids, labels, mlm_mask, selected


def make_example_from_residue_ids(
    residue_ids_np: np.ndarray,
    training: bool,
    mask_prob: float = DEFAULT_MASK_PROB,
    rng: random.Random | None = None,
) -> dict[str, list[int]]:
    if rng is None:
        rng = random

    length = len(residue_ids_np)

    if length > MAX_RESIDUES:
        if training:
            start = rng.randint(0, length - MAX_RESIDUES)
        else:
            start = 0
        crop = residue_ids_np[start : start + MAX_RESIDUES]
    else:
        crop = residue_ids_np

    ids = [tokeniser.cls_id]
    ids.extend(int(x) for x in crop)

    padded_ids = pad_ids(ids)
    pad_mask = make_pad_mask(padded_ids)
    input_ids, labels, mlm_mask, selected = MLM_mask(
        padded_ids,
        mask_prob=mask_prob,
        rng=rng,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pad_mask": pad_mask,
        "mlm_mask": mlm_mask,
        "selected": selected,
    }

def make_eval_windows_from_residue_ids(
    residue_ids_np: np.ndarray,
    protein_index: int,
    mask_prob: float = DEFAULT_MASK_PROB,
    eval_seed: int = DEFAULT_EVAL_SEED,
    stride: int = DEFAULT_EVAL_STRIDE,
) -> list[dict[str, list[int]]]:
    length = len(residue_ids_np)

    if length <= MAX_RESIDUES:
        starts = [0]
    else:
        last_start = length - MAX_RESIDUES
        starts = list(range(0, last_start + 1, stride))

        if starts[-1] != last_start:
            starts.append(last_start)

    examples = []

    for start in starts:
        crop = residue_ids_np[start : start + MAX_RESIDUES]

        ids = [tokeniser.cls_id]
        ids.extend(int(x) for x in crop)

        padded_ids = pad_ids(ids)
        pad_mask = make_pad_mask(padded_ids)

        rng = random.Random(eval_seed + protein_index * 1_000_003 + start)

        input_ids, labels, mlm_mask, selected = MLM_mask(
            padded_ids,
            mask_prob=mask_prob,
            rng=rng,
        )

        examples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "pad_mask": pad_mask,
                "mlm_mask": mlm_mask,
                "selected": selected,
            }
        )

    return examples

def examples_to_tensors(
    examples: list[dict[str, list[int]]],
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    batch = {}

    batch["input_ids"] = torch.tensor(
        [ex["input_ids"] for ex in examples],
        dtype=torch.long,
    )
    batch["labels"] = torch.tensor(
        [ex["labels"] for ex in examples],
        dtype=torch.long,
    )
    batch["pad_mask"] = torch.tensor(
        [ex["pad_mask"] for ex in examples],
        dtype=torch.bool,
    )
    batch["mlm_mask"] = torch.tensor(
        [ex["mlm_mask"] for ex in examples],
        dtype=torch.bool,
    )

    if device is not None:
        for key in batch:
            batch[key] = batch[key].to(device)

    return batch


def get_train_batch(
    cache: ProteinCache,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | torch.device | None = None,
    mask_prob: float = DEFAULT_MASK_PROB,
) -> dict[str, torch.Tensor]:
    examples = []

    for _ in range(batch_size):
        protein_index = int(random.choice(cache.train_indices))
        residue_ids_np = cache.get_residue_ids(protein_index)
        example = make_example_from_residue_ids(
            residue_ids_np,
            training=True,
            mask_prob=mask_prob,
        )
        examples.append(example)

    return examples_to_tensors(examples, device=device)


def get_eval_batch(
    cache: ProteinCache,
    start: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | torch.device | None = None,
    mask_prob: float = DEFAULT_MASK_PROB,
    eval_seed: int = DEFAULT_EVAL_SEED,
    stride: int = DEFAULT_EVAL_STRIDE,
) -> dict[str, torch.Tensor]:

    examples = []
    window_count = 0

    for protein_index_np in cache.eval_indices:
        protein_index = int(protein_index_np)
        residue_ids_np = cache.get_residue_ids(protein_index)

        protein_examples = make_eval_windows_from_residue_ids(
            residue_ids_np=residue_ids_np,
            protein_index=protein_index,
            mask_prob=mask_prob,
            eval_seed=eval_seed,
            stride=stride,
        )

        for example in protein_examples:
            if window_count >= start and len(examples) < batch_size:
                examples.append(example)

            window_count += 1

            if len(examples) == batch_size:
                return examples_to_tensors(examples, device=device)

    if len(examples) == 0:
        raise ValueError("empty eval batch; check start/batch_size/eval_indices")

    return examples_to_tensors(examples, device=device)


def batch_size_from_token_budget(
    max_tokens_per_batch: int,
    context_length: int = CONTEXT_LENGTH,
) -> int:
    batch_size = max_tokens_per_batch // context_length

    if batch_size < 1:
        raise ValueError(
            f"max_tokens_per_batch={max_tokens_per_batch} is smaller than context_length={context_length}"
        )

    return batch_size

def get_train_batch_by_tokens(
    cache: ProteinCache,
    max_tokens_per_batch: int,
    device: str | torch.device | None = None,
    mask_prob: float = DEFAULT_MASK_PROB,
) -> dict[str, torch.Tensor]:
    batch_size = batch_size_from_token_budget(max_tokens_per_batch)

    return get_train_batch(
        cache=cache,
        batch_size=batch_size,
        device=device,
        mask_prob=mask_prob,
    )

def get_eval_batch_by_tokens(
    cache: ProteinCache,
    start: int = 0,
    max_tokens_per_batch: int = CONTEXT_LENGTH * DEFAULT_BATCH_SIZE,
    device: str | torch.device | None = None,
    mask_prob: float = DEFAULT_MASK_PROB,
    eval_seed: int = DEFAULT_EVAL_SEED,
    stride: int = DEFAULT_EVAL_STRIDE,
) -> dict[str, torch.Tensor]:
    batch_size = batch_size_from_token_budget(max_tokens_per_batch)

    return get_eval_batch(
        cache=cache,
        start=start,
        batch_size=batch_size,
        device=device,
        mask_prob=mask_prob,
        eval_seed=eval_seed,
        stride=stride,
    )

#could be removed later
def sanity_check_batch(batch: dict[str, torch.Tensor], context_length: int = CONTEXT_LENGTH) -> None:
    assert batch["input_ids"].shape[1] == context_length
    assert batch["labels"].shape[1] == context_length
    assert batch["pad_mask"].shape[1] == context_length
    assert batch["mlm_mask"].shape[1] == context_length

    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["pad_mask"].dtype == torch.bool
    assert batch["mlm_mask"].dtype == torch.bool

    assert torch.all(batch["input_ids"][:, 0] != tokeniser.pad_id)
    assert torch.all(batch["labels"][:, 0] == -100) 
    assert (batch["labels"] != -100).sum() > 0