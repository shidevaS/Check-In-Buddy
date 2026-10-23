"""Train one local emotion classifier from both bundled CSV archives."""

import argparse
import json
import os
import random
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from app.database import PROJECT_ROOT

TEXT_LABELS = {
    "0": "sadness",
    "1": "joy",
    "2": "love",
    "3": "anger",
    "4": "fear",
    "5": "surprise",
    "sadness": "sadness",
    "joy": "joy",
    "love": "love",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
}

GO_EMOTION_LABELS = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "love",
    "confusion": "neutral",
    "curiosity": "neutral",
    "desire": "love",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "anger",
    "embarrassment": "fear",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}
LABELS = ("anger", "fear", "joy", "love", "neutral", "sadness", "surprise")
EMOTION_COLUMNS = tuple(GO_EMOTION_LABELS)


def _reservoir_rows(
    rows: Iterable[dict],
    label_for_row: Callable[[dict], Optional[str]],
    cap_per_label: int,
    rng: random.Random,
) -> Tuple[List[str], List[str], Counter]:
    reservoirs: Dict[str, List[str]] = {}
    seen = Counter()
    for row in rows:
        label = label_for_row(row)
        text = " ".join((row.get("text") or "").split())
        if label not in LABELS or not 3 <= len(text) <= 500:
            continue
        seen[label] += 1
        bucket = reservoirs.setdefault(label, [])
        if len(bucket) < cap_per_label:
            bucket.append(text)
        else:
            index = rng.randrange(seen[label])
            if index < cap_per_label:
                bucket[index] = text
    texts = []
    labels = []
    for label, examples in reservoirs.items():
        texts.extend(examples)
        labels.extend([label] * len(examples))
    return texts, labels, seen


def _read_archive(
    archive_path: Path,
    expected_csv: str,
    label_for_row: Callable[[dict], Optional[str]],
    cap_per_label: int,
    rng: random.Random,
) -> Tuple[List[str], List[str], Counter]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            name for name in archive.namelist()
            if Path(name).name.lower() == expected_csv.lower()
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one {expected_csv} file in {archive_path.name}; "
                f"found {len(candidates)}."
            )
        with archive.open(candidates[0]) as raw:
            import csv
            import io

            reader = csv.DictReader(
                io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
            )
            required_columns = {"text"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(f"{expected_csv} is missing its text column.")
            return _reservoir_rows(reader, label_for_row, cap_per_label, rng)


def _text_label(row: dict) -> Optional[str]:
    return TEXT_LABELS.get((row.get("label") or "").strip().lower())


def _go_emotion_label(row: dict) -> Optional[str]:
    if (row.get("example_very_unclear") or "").strip() == "1":
        return None
    active_labels = {
        GO_EMOTION_LABELS[column]
        for column in EMOTION_COLUMNS
        if (row.get(column) or "").strip() == "1"
    }
    if len(active_labels) != 1:
        return None
    return next(iter(active_labels))


def train(
    text_cap_per_label: int = 20000,
    go_cap_per_label: int = 6000,
    output_path: Optional[Path] = None,
) -> dict:
    """Train and persist an evaluated, balanced classifier."""
    from joblib import dump
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    rng = random.Random(20261021)
    text_x, text_y, text_counts = _read_archive(
        PROJECT_ROOT / "text.csv.zip",
        "text.csv",
        _text_label,
        text_cap_per_label,
        rng,
    )
    go_x, go_y, go_counts = _read_archive(
        PROJECT_ROOT / "go_emotions_dataset.csv.zip",
        "go_emotions_dataset.csv",
        _go_emotion_label,
        go_cap_per_label,
        rng,
    )
    texts = text_x + go_x
    labels = text_y + go_y
    counts = Counter(labels)
    if len(counts) < len(LABELS) or min(counts.values(), default=0) < 2:
        raise ValueError(
            "The datasets did not provide at least two usable examples for "
            "each supported emotion."
        )

    train_x, test_x, train_y, test_y = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=80000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                strip_accents="unicode",
                sublinear_tf=True,
                dtype=np.float32,
            ),
        ),
        (
            "classifier",
            SGDClassifier(
                loss="log_loss",
                max_iter=20,
                tol=0.001,
                class_weight="balanced",
                random_state=42,
                average=True,
            ),
        ),
    ])
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(test_x)
    accuracy = float(accuracy_score(test_y, predictions))
    report = classification_report(
        test_y,
        predictions,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )

    target = output_path or PROJECT_ROOT / "trained_models" / "emotion_classifier.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target.with_suffix(target.suffix + ".tmp")
    artifact = {
        "pipeline": pipeline,
        "labels": list(pipeline.named_steps["classifier"].classes_),
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "examples": len(texts),
        "class_counts": dict(counts),
        "source_rows": {
            "text_dataset": dict(text_counts),
            "go_emotions": dict(go_counts),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    dump(artifact, temporary_path, compress=3)
    os.replace(temporary_path, target)
    return {
        "model_path": str(target),
        "accuracy": artifact["accuracy"],
        "examples": artifact["examples"],
        "class_counts": artifact["class_counts"],
        "trained_at": artifact["trained_at"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the local emotion classifier from both bundled datasets."
    )
    parser.add_argument("--text-cap-per-label", type=int, default=20000)
    parser.add_argument("--go-cap-per-label", type=int, default=6000)
    args = parser.parse_args()
    if args.text_cap_per_label < 2 or args.go_cap_per_label < 2:
        parser.error("Per-label sample caps must be at least 2.")
    print(json.dumps(
        train(args.text_cap_per_label, args.go_cap_per_label),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
