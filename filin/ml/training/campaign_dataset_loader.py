from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

from schema import get_feature_profile, get_model_feature_columns


def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_campaign_datasets(index_path: Path, profile: str, role: str) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    paths = [Path(path) for path in index[f"{profile}_{role}"]]
    frames = [pd.read_csv(path, encoding="utf-8") for path in paths]
    if not frames or any(frame.empty for frame in frames):
        raise ValueError("Один из datasets кампании пуст.")
    features = get_feature_profile(profile)
    for frame in frames:
        if frame.get("feature_profile").nunique() != 1 or frame["feature_profile"].iloc[0] != profile:
            raise ValueError("Dataset не соответствует feature profile.")
        unexpected = set(get_model_feature_columns(frame.columns)) - set(features)
        if unexpected:
            raise ValueError("В model features обнаружены неописанные поля: " + ", ".join(sorted(unexpected)))
        if any(feature not in frame for feature in features):
            raise ValueError("В dataset отсутствуют обязательные признаки profile.")
    return pd.concat(frames, ignore_index=True), features, {str(path): calculate_file_sha256(path) for path in paths}
