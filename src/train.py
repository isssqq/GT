from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

from data import IMDBDataset, prepare_imdb_dataset


def evaluate_split(model, texts, labels, split_name: str) -> Dict[str, Any]:
    predictions = model.predict(texts)
    acc = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True, digits=4)
    return {"split": split_name, "accuracy": acc, "report": report}


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IMDB sentiment classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where to download and extract the dataset")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory to store trained models")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--max-features", type=int, default=40000, help="Number of TF-IDF features")
    parser.add_argument("--ngram-max", type=int, default=2, help="Use ngrams up to this size")
    parser.add_argument("--C", type=float, default=3.0, help="Inverse of regularization strength for logistic regression")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations for logistic regression")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs for model fitting")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional limit on training examples for quick runs")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional limit on test examples for quick runs")
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/metrics.json"), help="Where to save metrics as JSON")
    parser.add_argument(
        "--model-filename", type=str, default="tfidf_logreg.joblib", help="Filename for the serialized model"
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, args.ngram_max), stop_words="english")
    classifier = LogisticRegression(max_iter=args.max_iter, n_jobs=args.jobs, C=args.C, solver="liblinear")
    return make_pipeline(vectorizer, classifier)


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    dataset: IMDBDataset = prepare_imdb_dataset(
        data_root=args.data_dir,
        val_size=args.val_size,
        seed=args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )

    model = build_model(args)
    model.fit(dataset.train_texts, dataset.train_labels)

    metrics: Dict[str, Any] = {}
    for split_name, texts, labels in [
        ("train", dataset.train_texts, dataset.train_labels),
        ("val", dataset.val_texts, dataset.val_labels),
        ("test", dataset.test_texts, dataset.test_labels),
    ]:
        metrics[split_name] = evaluate_split(model, texts, labels, split_name)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / args.model_filename
    joblib.dump(model, model_path)

    save_metrics(metrics, args.metrics_path)

    return {
        "model_path": str(model_path),
        "metrics_path": str(args.metrics_path),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    results = train_and_evaluate(args)

    print(f"Model saved to: {results['model_path']}")
    print(f"Metrics saved to: {results['metrics_path']}")
    for split_name in ["train", "val", "test"]:
        acc = results["metrics"][split_name]["accuracy"]
        print(f"{split_name.capitalize()} accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
