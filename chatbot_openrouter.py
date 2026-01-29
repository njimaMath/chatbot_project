from __future__ import annotations

import json
import os
import pathlib
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

BASE_DIR = pathlib.Path(__file__).resolve().parent
SYSTEM_PROMPT = ""


def _load_dotenv(dotenv_path: pathlib.Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_knowledge_base(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise RuntimeError(f"Knowledge file not found: {path}") from e


def _build_system_prompt(knowledge_base: str) -> str:
    return (
        "あなたは、**東京確率セミナーの事務局を担当する、丁寧で親切な秘書AI**です。"
        "以下に提供されたセミナー情報のみに基づいて回答してくださいペンギン。\n\n"
        "【ルール】\n"
        "- 常に敬語ですペンギン\n"
        "- 語尾に必ず「ペンギン」を付けますペンギン\n"
        "- 情報がなければ、"
        "「申し訳ございません。提供された情報には、その件に関する記載がございませんでしたペンギン。」"
        "と答えますペンギン\n\n"
        "【セミナー情報】\n"
        f"{knowledge_base}"
    )


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _openai_compatible_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    extra_headers: dict[str, str] | None = None,
    timeout_s: int = 60,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(url=url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", f"Bearer {api_key}")
    if extra_headers:
        for k, v in extra_headers.items():
            if v:
                request.add_header(k, v)

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upstream error ({e.code}): {raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach upstream: {e}") from e

    parsed = json.loads(raw)
    try:
        return parsed["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected upstream response: {parsed}") from e


def _validate_messages(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        raise ValueError("Body must be a JSON object.")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Body.messages must be a JSON array.")

    cleaned: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Each message must be an object.")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError("Each message.role must be 'user' or 'assistant'.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each message.content must be a non-empty string.")
        cleaned.append({"role": role, "content": content})

    if not cleaned:
        raise ValueError("Body.messages must not be empty.")
    return cleaned


class ChatbotHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            _json_response(self, HTTPStatus.OK, {"ok": True})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0 or content_length > 1_000_000:
            _json_response(
                self,
                HTTPStatus.BAD_REQUEST,
                {"error": "Invalid Content-Length."},
            )
            return

        try:
            raw = self.rfile.read(content_length).decode("utf-8", errors="replace")
            payload = json.loads(raw)
            user_messages = _validate_messages(payload)

            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            if not api_key.strip():
                raise RuntimeError(
                    "Missing OPENROUTER_API_KEY (or OPENAI_API_KEY). Set it in .env."
                )

            base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").strip()
            model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()

            extra_headers = {
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "").strip(),
                "X-Title": os.getenv("OPENROUTER_X_TITLE", "").strip(),
            }

            reply = _openai_compatible_chat_completion(
                base_url=base_url,
                api_key=api_key.strip(),
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *user_messages],
                temperature=0.1,
                extra_headers=extra_headers,
            )

            _json_response(self, HTTPStatus.OK, {"reply": reply})
        except (ValueError, json.JSONDecodeError) as e:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(e)})
        except Exception as e:
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})


def main() -> int:
    _load_dotenv(BASE_DIR / ".env")

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")

    knowledge_base = _read_knowledge_base(BASE_DIR / "website_data.txt")
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = _build_system_prompt(knowledge_base)

    server = ThreadingHTTPServer((host, port), ChatbotHandler)
    print(f"Serving on http://{host}:{port}  (GET /, POST /api/chat)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
