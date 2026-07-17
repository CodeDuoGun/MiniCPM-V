"""Small dependency-free checks for complete common raster image containers."""

from __future__ import annotations

from pathlib import Path


def is_complete_image(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if size < 32:
            return False
        with path.open("rb") as handle:
            head = handle.read(16)
            handle.seek(max(0, size - 256))
            tail = handle.read(256)
        if head.startswith(b"\xff\xd8"):
            # Some valid exporters append a small trailer after the JPEG EOI.
            return b"\xff\xd9" in tail
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return b"IEND\xaeB`\x82" in tail
        if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            return int.from_bytes(head[4:8], "little") + 8 == size
        if head.startswith(b"BM"):
            return int.from_bytes(head[2:6], "little") <= size
        return False
    except OSError:
        return False
