from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import joblib


POSITIVE = 1
NEGATIVE = 0
LABEL_NAMES = {POSITIVE: "positive", NEGATIVE: "negative"}


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def predict_texts(model, texts: Iterable[str]) -> List[str]:
    numeric_preds = model.predict(list(texts))
    return [LABEL_NAMES[int(label)] for label in numeric_preds]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained sentiment model")
    parser.add_argument("model_path", type=Path, help="Path to the serialized model")
    parser.add_argument("texts", nargs="+", help="One or more texts to classify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model_path)
    predictions = predict_texts(model, args.texts)

    for text, pred in zip(args.texts, predictions):
        print(f"[{pred}] {text}")


if __name__ == "__main__":
    main()
