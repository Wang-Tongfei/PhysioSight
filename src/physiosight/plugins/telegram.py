"""Telegram notifications (opt-in). Credentials come from the environment:
PHYSIOSIGHT_TG_TOKEN / PHYSIOSIGHT_TG_CHAT (legacy PHYSIO_TG_* also accepted)."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None) -> None:
        self._token = token or os.environ.get("PHYSIOSIGHT_TG_TOKEN") or os.environ.get("PHYSIO_TG_TOKEN", "")
        self._chat = chat_id or os.environ.get("PHYSIOSIGHT_TG_CHAT") or os.environ.get("PHYSIO_TG_CHAT", "")
        try:
            import requests  # noqa: F401
            has_requests = True
        except ImportError:
            has_requests = False
        self.enabled = bool(self._token and self._chat and has_requests)
        self._threads: list[threading.Thread] = []

    def send_message(self, text: str) -> None:
        if self.enabled:
            self._spawn("sendMessage", {"chat_id": self._chat, "text": text})

    def send_video(self, path: Path, caption: str = "") -> None:
        if self.enabled:
            self._spawn("sendVideo", {"chat_id": self._chat, "caption": caption}, path)

    def flush(self, timeout: float = 60) -> None:
        for th in list(self._threads):
            th.join(timeout)

    def _spawn(self, method: str, data: dict, video: Path | None = None) -> None:
        self._threads = [th for th in self._threads if th.is_alive()]
        th = threading.Thread(target=self._post, args=(method, data, video), daemon=True)
        th.start()
        self._threads.append(th)

    def _post(self, method: str, data: dict, video: Path | None) -> None:
        import requests

        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            if video is not None:
                with open(video, "rb") as fh:
                    resp = requests.post(url, data=data, timeout=60,
                                         files={"video": (video.name, fh, "video/mp4")})
            else:
                resp = requests.post(url, data=data, timeout=60)
            if not resp.ok:
                log.warning("telegram %s failed: %s", method, resp.status_code)
        except Exception as exc:  # network errors must never break analysis
            log.warning("telegram %s error: %s", method, exc)
