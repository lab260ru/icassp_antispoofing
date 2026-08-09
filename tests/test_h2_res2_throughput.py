from __future__ import annotations

import pandas as pd

from scripts.benchmark_h2_res2_pytorch import select_fastest_safe


def test_fastest_safe_ignores_oom_and_breaks_exact_ties_by_smallest_batch() -> None:
    rows = pd.DataFrame(
        [
            {"batch_size": 1, "status": "ok", "clips_per_second": 10.0},
            {"batch_size": 2, "status": "oom", "clips_per_second": float("nan")},
            {"batch_size": 8, "status": "ok", "clips_per_second": 20.0},
            {"batch_size": 4, "status": "ok", "clips_per_second": 20.0},
        ]
    )
    assert select_fastest_safe(rows)["batch_size"] == 4


def test_fastest_safe_requires_at_least_one_completed_candidate() -> None:
    rows = pd.DataFrame([{"batch_size": 32, "status": "oom", "clips_per_second": float("nan")}])
    try:
        select_fastest_safe(rows)
    except RuntimeError as error:
        assert "No safe" in str(error)
    else:
        raise AssertionError("Expected no-safe-candidate error")
