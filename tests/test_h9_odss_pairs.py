from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.h9_odss_pairs import (
    H9_DEV_FRACTION,
    H9_SPLIT_SEED,
    build_h9_odss_pairing_freeze,
    freeze_h9_odss_pairing_from_files,
    write_h9_odss_pairing_artifacts,
)


def _record(utterance_id: str, label: int, *, include_notes: bool = False) -> dict[str, object]:
    generator, corpus, speaker, stem = utterance_id.split("__", 3)
    row: dict[str, object] = {"utterance_id": utterance_id, "label": label}
    if include_notes:
        row["path"] = f"{generator}/{corpus}/{speaker}/{stem}.wav"
        row["notes"] = json.dumps(
            {
                "utterance_id": utterance_id,
                "generator": generator,
                "source_corpus": corpus,
                "speaker": speaker,
                "language": {"hifi-tts": "en", "hui-acg": "de", "openslr-es": "es"}[corpus],
            }
        )
    return row


def _complete_rows(n_groups: int = 6000, *, include_notes: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    corpora = ("hifi-tts", "hui-acg", "openslr-es")
    for index in range(n_groups):
        corpus = corpora[index % len(corpora)]
        speaker = f"s{index % 97:03d}"
        stem = f"text_{index:05d}"
        for generator, label in (("natural", 0), ("vits", 1), ("fastpitch-hifigan", 1)):
            rows.append(_record(f"{generator}__{corpus}__{speaker}__{stem}", label, include_notes=include_notes))
    return pd.DataFrame(rows)


def _records(tmp_path: Path) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels = tmp_path / "labels.csv"
    labels.write_text("utterance_id,label\nplaceholder,0\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "each natural utterance is paired with TTS re-synthesis of the same text",
                "natural/<corpus>/<speaker>/<stem>.wav",
                "vits/<corpus>/<speaker>/<stem>.wav",
                "fastpitch-hifigan/<corpus>/<speaker>/<stem>.wav",
            ]
        ),
        encoding="utf-8",
    )
    build = tmp_path / "build_parquet.py"
    build.write_text(
        "\n".join(
            [
                "utterance_id = full source-relative path with '/' -> '__'",
                'parts = rel.split("/")',
                'label = "bonafide" if gen == "natural" else "spoof"',
            ]
        ),
        encoding="utf-8",
    )
    docs: dict[str, dict[str, object]] = {}
    for name in ("TASK.md", "DATA.md", "PLAN.md", "BRAINSTORM.md"):
        path = tmp_path / name
        path.write_text(f"{name}\n", encoding="utf-8")
        docs[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "byte_size": path.stat().st_size}
    metadata = {"path": str(labels), "sha256": hashlib.sha256(labels.read_bytes()).hexdigest(), "byte_size": labels.stat().st_size}
    semantics = {
        "readme": {"path": str(readme), "sha256": hashlib.sha256(readme.read_bytes()).hexdigest(), "byte_size": readme.stat().st_size},
        "build_script": {"path": str(build), "sha256": hashlib.sha256(build.read_bytes()).hexdigest(), "byte_size": build.stat().st_size},
    }
    return metadata, semantics, docs


def _freeze(tmp_path: Path, rows: pd.DataFrame):
    metadata, semantics, docs = _records(tmp_path)
    return build_h9_odss_pairing_freeze(
        rows,
        metadata_record=metadata,
        semantics_records=semantics,
        protocol_records=docs,
    )


def test_complete_groups_have_deterministic_voice_disjoint_pairs_and_b2_control(tmp_path: Path) -> None:
    rows = _complete_rows(include_notes=True)
    first = _freeze(tmp_path / "first", rows)
    second = _freeze(tmp_path / "second", rows.sample(frac=1.0, random_state=17).reset_index(drop=True))

    pd.testing.assert_frame_equal(first.trials, second.trials)
    pd.testing.assert_frame_equal(first.pairs, second.pairs)
    pd.testing.assert_frame_equal(first.random_pairs, second.random_pairs)
    assert len(first.trials) == 18_000
    assert len(first.pairs) == 12_000
    assert first.provenance["counts"]["complete_groups"] == 6_000
    assert first.trials.groupby("voice_key")["split"].nunique().eq(1).all()
    assert set(first.trials["split"]) == {"train", "dev"}
    assert first.trials.groupby("split")["group_key"].nunique().min() >= 1_000
    assert first.trials.loc[:, ["language", "split"]].drop_duplicates().groupby("language")["split"].agg(set).eq({"train", "dev"}).all()
    assert len(first.random_pairs) == len(first.pairs)
    assert not first.random_pairs["content_key"].eq(first.random_pairs["random_bona_content_key"]).any()
    assert first.random_pairs["split"].eq(first.random_pairs["random_bona_split"]).all()
    assert first.random_pairs["language"].eq(first.random_pairs["random_bona_language"]).all()
    assert first.random_pairs["source_corpus"].eq(first.random_pairs["random_bona_source_corpus"]).all()
    assert set(first.trials["source_pool"]) == {"shared_b1_b2_p_complete_matched_only"}
    pd.testing.assert_frame_equal(
        first.pairs.loc[:, ["pair_id", "spoof_generator"]].sort_values(["pair_id", "spoof_generator"], kind="stable").reset_index(drop=True),
        first.random_pairs.loc[:, ["pair_id", "spoof_generator"]].sort_values(["pair_id", "spoof_generator"], kind="stable").reset_index(drop=True),
    )
    assert first.provenance["split_rule"]["split_seed"] == H9_SPLIT_SEED
    assert first.provenance["split_rule"]["dev_fraction"] == H9_DEV_FRACTION
    assert first.provenance["source_access"]["audio_opened_or_decoded"] is False
    assert first.provenance["source_access"]["target_data_read"] is False
    assert first.provenance["same_source_pool_guard"]["p_edge_keys_sha256"] == first.provenance["same_source_pool_guard"]["b2_edge_keys_sha256"]


def test_incomplete_vctk_like_groups_are_excluded_not_paired(tmp_path: Path) -> None:
    rows = _complete_rows()
    incomplete = [
        _record(f"vits__vctk__p{index:04d}__missing_{index:05d}", 1)
        for index in range(500)
    ]
    freeze = _freeze(tmp_path, pd.concat([rows, pd.DataFrame(incomplete)], ignore_index=True))
    assert not freeze.trials["source_corpus"].eq("vctk").any()
    assert not freeze.pairs["content_key"].str.contains("vctk/").any()
    assert len(freeze.excluded_unmatched) == 500
    assert set(freeze.excluded_unmatched["exclusion_reason"]) == {"missing_bonafide"}
    assert set(freeze.excluded_unmatched["utterance_id"]).isdisjoint(set(freeze.trials["utterance_id"]))
    assert freeze.provenance["same_source_pool_guard"]["conditions"] == ["B1", "B2", "P"]


def test_rejects_generator_label_conflict_and_ambiguous_rendering(tmp_path: Path) -> None:
    rows = _complete_rows()
    rows.loc[rows.index[rows["utterance_id"].str.startswith("vits__")][0], "label"] = 0
    with pytest.raises(ValueError, match="generator/label conflict"):
        _freeze(tmp_path / "conflict", rows)

    rows = _complete_rows()
    duplicate = rows.iloc[[0]].copy()
    duplicate.loc[:, "utterance_id"] = "natural__hifi-tts__s000__other_text"
    duplicate.loc[:, "label"] = 0
    # Make the altered UID collide with an existing content/generator tuple.
    duplicate.loc[:, "utterance_id"] = rows.iloc[0]["utterance_id"]
    with pytest.raises(ValueError, match="duplicate ODSS utterance_id"):
        _freeze(tmp_path / "duplicate", pd.concat([rows, duplicate], ignore_index=True))


def test_rejects_insufficient_complete_groups_per_split(tmp_path: Path) -> None:
    rows = _complete_rows(n_groups=1_200)
    with pytest.raises(ValueError, match="each split must retain at least"):
        _freeze(tmp_path, rows)


def test_rejects_language_without_both_voice_disjoint_splits(tmp_path: Path) -> None:
    rows = _complete_rows()
    hifi = rows["utterance_id"].str.contains("__hifi-tts__")
    rows.loc[hifi, "utterance_id"] = rows.loc[hifi, "utterance_id"].str.replace(
        r"^(natural|vits|fastpitch-hifigan)__hifi-tts__s\d+__",
        r"\1__hifi-tts__onevoice__",
        regex=True,
    )
    with pytest.raises(ValueError, match="language stratum cannot populate both voice-disjoint splits"):
        _freeze(tmp_path, rows)


def test_file_reader_verifies_semantics_and_writer_refuses_overwrite(tmp_path: Path) -> None:
    rows = _complete_rows()
    metadata = tmp_path / "odss_labels.csv"
    rows.loc[:, ["utterance_id", "label"]].to_csv(metadata, index=False)
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "each natural utterance is paired with TTS re-synthesis of the same text",
                "natural/<corpus>/<speaker>/<stem>.wav",
                "vits/<corpus>/<speaker>/<stem>.wav",
                "fastpitch-hifigan/<corpus>/<speaker>/<stem>.wav",
            ]
        ),
        encoding="utf-8",
    )
    build = tmp_path / "build_parquet.py"
    build.write_text(
        "\n".join(
            [
                "utterance_id = full source-relative path with '/' -> '__'",
                'parts = rel.split("/")',
                'label = "bonafide" if gen == "natural" else "spoof"',
            ]
        ),
        encoding="utf-8",
    )
    protocol_paths = {}
    for name in ("TASK.md", "DATA.md", "PLAN.md", "BRAINSTORM.md"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        protocol_paths[name] = path
    freeze = freeze_h9_odss_pairing_from_files(
        metadata_path=metadata,
        semantics_documents={"readme": readme, "build_script": build},
        protocol_documents=protocol_paths,
    )
    output_root = tmp_path / "hdd"
    outputs = write_h9_odss_pairing_artifacts(freeze, output_dir=output_root / "freeze", allowed_output_root=output_root)
    report = json.loads(outputs["provenance"].read_text(encoding="utf-8"))
    assert report["outputs"]["trials_csv"]["sha256"] == hashlib.sha256(outputs["trials"].read_bytes()).hexdigest()
    assert report["outputs"]["pairs_csv"]["sha256"] == hashlib.sha256(outputs["pairs"].read_bytes()).hexdigest()
    assert report["outputs"]["b2_random_pairs_csv"]["sha256"] == hashlib.sha256(outputs["random_pairs"].read_bytes()).hexdigest()
    assert report["outputs"]["excluded_unmatched_csv"]["sha256"] == hashlib.sha256(outputs["excluded_unmatched"].read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_h9_odss_pairing_artifacts(freeze, output_dir=output_root / "freeze", allowed_output_root=output_root)
    with pytest.raises(ValueError, match="must be below"):
        write_h9_odss_pairing_artifacts(freeze, output_dir=tmp_path / "outside", allowed_output_root=output_root)
