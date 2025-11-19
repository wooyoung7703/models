import gzip
import json
from pathlib import Path

from backend.app import seq_buffer
from backend.app.core.config import settings


def test_save_and_load_sequence_buffers(tmp_path):
    seq_buffer.clear_buffers()
    buf = seq_buffer.get_buffer("xrpusdt")
    buf.append([float(i) for i in range(4)])
    buf.append([float(i) for i in range(4, 8)])
    target = tmp_path / "seq_snapshot.json"
    saved = seq_buffer.save_buffers_to_path(str(target))
    assert saved == 1

    seq_buffer.clear_buffers()
    loaded = seq_buffer.load_buffers_from_path(str(target))
    assert loaded == 1
    restored = seq_buffer.get_buffer("xrpusdt")
    assert len(restored) == 2
    assert restored.to_list()[0][0] == 0.0


def test_load_ignores_invalid_vectors(tmp_path):
    data = {
        "version": 1,
        "seq_len": 30,
        "buffers": {
            "xrpusdt": [
                ["1", "bad", 3],
                "not-a-list",
            ]
        }
    }
    target = tmp_path / "bad.json"
    Path(target).write_text(json.dumps(data), encoding="utf-8")

    seq_buffer.clear_buffers()
    loaded = seq_buffer.load_buffers_from_path(str(target), seq_len=2)
    assert loaded == 1
    restored = seq_buffer.get_buffer("xrpusdt")
    assert len(restored) == 1  # second entry ignored, capacity=2 but invalid entry dropped
    assert restored.to_list()[0][1] == 0.0  # "bad" coerced to 0.0


def test_snapshot_archival_rotation(tmp_path):
    seq_buffer.clear_buffers()
    buf = seq_buffer.get_buffer("btcusdt")
    buf.append([0.0, 1.0])
    buf.append([2.0, 3.0])
    target = tmp_path / "seq_snapshot.json"
    archive_dir = tmp_path / "archives"

    orig_dir = settings.SEQ_BUFFER_SNAPSHOT_ARCHIVE_DIR
    orig_compress = settings.SEQ_BUFFER_SNAPSHOT_COMPRESS
    orig_keep = settings.SEQ_BUFFER_SNAPSHOT_ARCHIVE_KEEP
    try:
        settings.SEQ_BUFFER_SNAPSHOT_ARCHIVE_DIR = str(archive_dir)
        settings.SEQ_BUFFER_SNAPSHOT_COMPRESS = True
        settings.SEQ_BUFFER_SNAPSHOT_ARCHIVE_KEEP = 2
        # generate three snapshots to test rotation
        for _ in range(3):
            seq_buffer.save_buffers_to_path(str(target))
        archives = sorted(archive_dir.glob("seq_buffer_*.json.gz"))
        assert len(archives) == 2  # retention enforced
        # ensure payload remained intact inside compressed archive
        with gzip.open(archives[-1], "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["buffers"]
    finally:
        settings.SEQ_BUFFER_SNAPSHOT_ARCHIVE_DIR = orig_dir
        settings.SEQ_BUFFER_SNAPSHOT_COMPRESS = orig_compress
        settings.SEQ_BUFFER_SNAPSHOT_ARCHIVE_KEEP = orig_keep
        seq_buffer.clear_buffers()
