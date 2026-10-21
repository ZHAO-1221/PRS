"""Build the fixed mixed interaction split without identity leakage.

The selected test IDs come from the mixed demonstration evaluation. Any other
clip sharing a dog or person with a selected test clip is disabled instead of
being moved into training. Original fixed-test clips that are not selected and
do not share a test identity are returned to the training pool.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.manifest).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows, fieldnames = list(reader), list(reader.fieldnames or ())
    with Path(args.selection).open(encoding="utf-8-sig", newline="") as handle:
        selected_rows = list(csv.DictReader(handle))

    selected_ids = {row["clip_id"] for row in selected_rows}
    by_id = {row["clip_id"]: row for row in rows}
    missing = sorted(selected_ids - set(by_id))
    if missing:
        raise ValueError(f"selection contains unknown clips: {missing}")

    test_dogs = {by_id[clip_id]["dog_id"] for clip_id in selected_ids}
    test_people = {by_id[clip_id]["person_id"] for clip_id in selected_ids}
    output_rows = []
    excluded_identity = recovered_old_test = 0
    for row in rows:
        copied = dict(row)
        notes = copied.get("notes", "")
        if copied["clip_id"] in selected_ids:
            copied["split"] = "test"
            copied["valid"] = "1"
            tag = "mixed_test_v2"
        elif copied["dog_id"] in test_dogs or copied["person_id"] in test_people:
            copied["split"] = "excluded"
            copied["valid"] = "0"
            tag = "excluded_to_prevent_mixed_test_identity_leakage"
            excluded_identity += 1
        elif copied["split"] == "test":
            copied["split"] = "train"
            copied["valid"] = "1"
            tag = "recovered_original_fixed_test_to_train"
            recovered_old_test += 1
        else:
            tag = "mixed_v2_training_pool"
        copied["notes"] = ";".join(value for value in (notes, tag) if value)
        output_rows.append(copied)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    valid = [row for row in output_rows if row["valid"] == "1"]
    summary = {
        "output": str(destination),
        "valid_clips": len(valid),
        "train": sum(row["split"] == "train" for row in valid),
        "val": sum(row["split"] == "val" for row in valid),
        "test": sum(row["split"] == "test" for row in valid),
        "test_classes": Counter(
            row["interaction"] for row in valid if row["split"] == "test"
        ),
        "excluded_identity": excluded_identity,
        "recovered_old_test": recovered_old_test,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
