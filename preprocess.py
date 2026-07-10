"""preprocessing
Raw FASTA.gz ->token cache
 >> tokens.bin: all residue token IDs concatenated, uint8
 >> offsets.npy: start offset for each protein in tokens.bin
 >> lengths.npy: residue length for each protein
 >> train_indices.npy / eval_indices.npy: protein split
>> meta.json:  metadata
[CLS], [PAD] and [MASK] are added inside data.py
"""

from __future__ import annotations

import gzip
import json
import random
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm.auto import tqdm
from collections import Counter

from tokeniser import ProteinTokeniser, clean_sequence


UNIREF_50 = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"

DATASET_DIR = Path("dataset_PLM")
FASTA_PATH = DATASET_DIR / "uniref50.fasta.gz"
PROCESSED_DIR = DATASET_DIR / "processed"

CONTEXT_LENGTH = 512
MAX_RESIDUES = CONTEXT_LENGTH - 1

EVAL_FRACTION = 0.05
SPLIT_SEED = 42
TOKEN_DTYPE = np.uint8


def download_uniref50(
    path: Path | str = FASTA_PATH,
    url: str = UNIREF_50,
) -> Path:
    #Download UniRef50 FASTA if it is missing.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"found existing file in--> {path}")
        return path

    print(f"downloading UniRef50 to--> {path}, takes about 4 mins")
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))

        with open(tmp_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=path.name,
        ) as pbar:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                f.write(chunk)
                pbar.update(len(chunk))

    tmp_path.replace(path)
    return path


def stream_fasta(path: Path | str) -> Iterator[tuple[str, str]]:
    #Stream FASTA records as (header, sequence).
    open_fn = gzip.open if str(path).endswith(".gz") else open

    with open_fn(path, "rt") as f:
        header = None
        seq_lines = []

        for line in f:
            line = line.strip()

            if line == "":
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)

                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            yield header, "".join(seq_lines)


def make_train_eval_indices(
    num_sequences: int,
    eval_fraction: float = EVAL_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    indices = list(range(num_sequences))
    rng = random.Random(seed)
    rng.shuffle(indices)

    num_eval = int(eval_fraction * num_sequences)
    eval_indices = np.array(indices[:num_eval], dtype=np.int64)
    train_indices = np.array(indices[num_eval:], dtype=np.int64)

    return train_indices, eval_indices


def preprocess_fasta(
    fasta_path: Path | str = FASTA_PATH,
    processed_dir: Path | str = PROCESSED_DIR,
    eval_fraction: float = EVAL_FRACTION,
    split_seed: int = SPLIT_SEED,
    max_records: int | None = None,
) -> None:
    #Build processed cache from a FASTA/FASTA.gz file
    fasta_path = Path(fasta_path)
    processed_dir = Path(processed_dir)

    if not fasta_path.exists():
        if fasta_path == FASTA_PATH:
            fasta_path = download_uniref50(fasta_path)
        else:
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    tokeniser = ProteinTokeniser()

    tokens_path = processed_dir / "tokens.bin"
    offsets_path = processed_dir / "offsets.npy"
    lengths_path = processed_dir / "lengths.npy"
    train_indices_path = processed_dir / "train_indices.npy"
    eval_indices_path = processed_dir / "eval_indices.npy"
    meta_path = processed_dir / "meta.json"

    offsets = []
    lengths = []
    raw_residue_counts = Counter()
    skipped = 0
    current_offset = 0

    with open(tokens_path, "wb") as token_file:
        for record_idx, (header, seq) in enumerate(tqdm(stream_fasta(fasta_path), desc="preprocessing FASTA")):
            if max_records is not None and record_idx >= max_records:
                break

            try:
                clean = clean_sequence(seq)
                raw_residue_counts.update(clean)
            except ValueError:
                skipped += 1
                continue

            if len(clean) == 0:
                skipped += 1
                continue

            ids = tokeniser.encode(clean, add_cls=False)
            arr = np.asarray(ids, dtype=TOKEN_DTYPE)

            offsets.append(current_offset)
            lengths.append(len(arr))
            arr.tofile(token_file)

            current_offset += len(arr)

    offsets_arr = np.asarray(offsets, dtype=np.int64)
    lengths_arr = np.asarray(lengths, dtype=np.int32)

    train_indices, eval_indices = make_train_eval_indices(
        num_sequences=len(lengths_arr),
        eval_fraction=eval_fraction,
        seed=split_seed,
    )

    np.save(offsets_path, offsets_arr)
    np.save(lengths_path, lengths_arr)
    np.save(train_indices_path, train_indices)
    np.save(eval_indices_path, eval_indices)

    meta = {
        "context_length": CONTEXT_LENGTH,
        "max_residues": MAX_RESIDUES,
        "token_dtype": "uint8",
        "num_sequences": int(len(lengths_arr)),
        "num_skipped": int(skipped),
        "num_train": int(len(train_indices)),
        "num_eval": int(len(eval_indices)),
        "eval_fraction": float(eval_fraction),
        "split_seed": int(split_seed),
        "raw_residue_counts": dict(sorted(raw_residue_counts.items())),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("processed cache written to:", processed_dir)
    print("num sequences:", len(lengths_arr))
    print("num skipped:", skipped)
    print("num train:", len(train_indices))
    print("num eval:", len(eval_indices))
