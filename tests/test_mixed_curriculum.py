from __future__ import annotations

import json
from pathlib import Path

from scripts.make_agentcoder_mixed_curriculum import build_mixed_records, write_jsonl


def test_mixed_curriculum_samples_real_records_and_repeats_curated(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps({"text": "plain text row should not be selected"}),
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "Fix add."}],
                        "trace": [{"type": "assistant", "content": "Use addition."}],
                    }
                ),
                json.dumps(
                    {
                        "repo_context": "file: app.py",
                        "messages": [{"role": "user", "content": "Where is main?"}],
                        "final": "main is in app.py",
                    }
                ),
            ]
        )
        + "\n"
    )

    records, manifest = build_mixed_records(
        [str(raw)],
        max_real_records=1,
        max_real_document_chars=1000,
        curated_repeats=2,
        seed=17,
    )

    assert manifest["real_stats"]["plain_text_records"] == 1
    assert manifest["real_stats"]["structured_records"] == 2
    assert manifest["real_records_selected"] == 1
    assert manifest["curated_base_records"] == 96
    assert manifest["curated_records"] == 192
    assert manifest["total_records"] == 193
    assert sum(1 for record in records if str(record.get("_curriculum_source", "")).startswith("curated:")) == 192

    out = tmp_path / "mixed.jsonl"
    write_jsonl(out, records)
    assert len(out.read_text().splitlines()) == 193
