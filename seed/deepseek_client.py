from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


def load_local_env(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        local_env = Path(__file__).resolve().parents[1] / ".env.local"
        load_local_env(local_env)
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return cls(api_key=api_key, model=model, base_url=base_url)


class DeepSeekClient:
    def __init__(self, config: DeepSeekConfig):
        self.config = config

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 800) -> Dict[str, Any]:
        try:
            return self._chat_via_curl(messages, temperature=temperature, max_tokens=max_tokens)
        except RuntimeError:
            return self._chat_via_urllib(messages, temperature=temperature, max_tokens=max_tokens)

    def _chat_via_curl(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        result = subprocess.run(
            [
                "curl",
                "-sS",
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                "-H",
                f"Authorization: Bearer {self.config.api_key}",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(payload, ensure_ascii=False),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "curl request failed")
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid DeepSeek JSON response: {result.stdout[:500]}") from exc
        if "error" in data:
            raise RuntimeError(f"DeepSeek API error: {data['error']}")
        return data

    def _chat_via_urllib(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc}") from exc
