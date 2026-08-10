"""Synthetic contract test for the sealed H9 source-only lambda selector."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.h9_pcr_training import H9_LAMBDA_GRID, H9_SEEDS


def test_source_only_lambda_cli_freezes_full_grid(tmp_path: Path) -> None:
    inputs: list[Path] = []
    for lam in H9_LAMBDA_GRID:
        for seed in H9_SEEDS:
            path = tmp_path / f"p_{lam}_{seed}.json"
            path.write_text(
                json.dumps(
                    {
                        "method": "P",
                        "lambda_rank": lam,
                        "seed": seed,
                        "source_dev_eer": lam,
                        "target_labels_read": False,
                        "target_audio_read": False,
                        "source_artifact_hashes": {
                            "source_manifest_sha256": "a" * 64,
                            "p_pairs_sha256": "b" * 64,
                            "b2_pairs_sha256": "c" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            inputs.append(path)
    output = tmp_path / "selection.json"
    command = [sys.executable, "scripts/freeze_h9_pcr_lambda_selection.py"]
    for path in inputs:
        command.extend(("--result", str(path)))
    command.extend(("--output", str(output)))
    subprocess.run(command, check=True)
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["selected_lambda_rank"] == 0.1
    assert value["target_metrics_read"] is False
    assert len(value["input_sidecars"]) == 12
