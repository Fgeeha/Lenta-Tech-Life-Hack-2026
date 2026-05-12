"""ByteTrack-обёртка (заглушка)."""

import logging

logger = logging.getLogger(__name__)


class Tracker:
    """Тонкая обёртка над supervision ByteTrack."""

    def __init__(self) -> None:
        self._tracker = None

    def _load(self) -> None:
        import supervision as sv

        self._tracker = sv.ByteTrack()

    def update(self, detections):  # type: ignore[return]
        if self._tracker is None:
            self._load()
        return self._tracker.update_with_detections(detections)
