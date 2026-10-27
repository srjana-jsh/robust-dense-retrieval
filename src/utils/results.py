from __future__ import annotations

import json
from pathlib import Path


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print()
    print(title)

    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")


def save_results(
    project_root: Path,
    method: str,
    results: dict,
    settings: dict | None = None,
) -> Path:
    """Save results for all datasets for one retrieval method."""

    output_dir = project_root / "reports" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{method}.json"

    payload = {
        "method": method,
        "datasets": results,
    }

    if settings:
        payload["settings"] = settings

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return output_path