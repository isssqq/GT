from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


@dataclass
class IMDBDataset:
    train_texts: List[str]
    train_labels: List[int]
    val_texts: List[str]
    val_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]


class _DownloadProgressBar(tqdm):
    def update_to(self, block_num: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        downloaded = block_num * block_size
        self.update(downloaded - self.n)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_imdb(data_dir: Path) -> Path:
    """Download the IMDB dataset tarball if it is not already present."""
    raw_dir = data_dir / "raw"
    _ensure_directory(raw_dir)
    target_path = raw_dir / "aclImdb_v1.tar.gz"

    if target_path.exists():
        return target_path

    with _DownloadProgressBar(unit="B", unit_scale=True, desc="Downloading IMDB") as progress:
        urllib.request.urlretrieve(IMDB_URL, filename=target_path, reporthook=progress.update_to)  # type: ignore[arg-type]
    return target_path


def extract_imdb(tar_path: Path, data_dir: Path) -> Path:
    """Extract the IMDB tarball into the provided data directory."""
    extracted_dir = data_dir / "aclImdb"
    if extracted_dir.exists():
        return extracted_dir

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    return extracted_dir


def _read_split(split_dir: Path) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    for label_name, label_value in {"pos": 1, "neg": 0}.items():
        label_dir = split_dir / label_name
        for path in sorted(label_dir.glob("*.txt")):
            texts.append(path.read_text(encoding="utf-8"))
            labels.append(label_value)
    return texts, labels


def _subset(texts: List[str], labels: List[int], limit: int | None) -> Tuple[List[str], List[int]]:
    if limit is None or limit >= len(texts):
        return texts, labels
    return texts[:limit], labels[:limit]


def load_imdb_splits(data_dir: Path, train_limit: int | None = None, test_limit: int | None = None) -> Tuple[List[str], List[int], List[str], List[int]]:
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_texts, train_labels = _read_split(train_dir)
    test_texts, test_labels = _read_split(test_dir)

    train_texts, train_labels = _subset(train_texts, train_labels, train_limit)
    test_texts, test_labels = _subset(test_texts, test_labels, test_limit)

    return train_texts, train_labels, test_texts, test_labels


def prepare_imdb_dataset(
    data_root: Path,
    val_size: float = 0.2,
    seed: int = 42,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> IMDBDataset:
    tar_path = download_imdb(data_root)
    extracted_dir = extract_imdb(tar_path, data_root)

    train_texts, train_labels, test_texts, test_labels = load_imdb_splits(
        extracted_dir, train_limit=train_limit, test_limit=test_limit
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_size, random_state=seed, stratify=train_labels
    )

    return IMDBDataset(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
    )


__all__ = [
    "IMDBDataset",
    "prepare_imdb_dataset",
    "download_imdb",
    "extract_imdb",
]
