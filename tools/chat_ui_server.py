#!/usr/bin/env python3
"""Small local chat UI/proxy for the nano-vllm-jax Flask server."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>nano-vllm-jax chat</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: #f6f7f9;
      color: #16181d;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    header {
      padding: 14px 18px;
      border-bottom: 1px solid #d9dee7;
      background: #ffffff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    h1 {
      font-size: 16px;
      margin: 0;
      font-weight: 650;
    }
    .status {
      font-size: 13px;
      color: #5d6675;
      white-space: nowrap;
    }
    main {
      overflow-y: auto;
      padding: 18px;
    }
    .messages {
      max-width: 980px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      max-width: min(780px, 100%);
      border: 1px solid #d9dee7;
      border-radius: 8px;
      padding: 12px 14px;
      background: #ffffff;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .msg.user {
      align-self: flex-end;
      background: #ecf4ff;
      border-color: #bfd7fb;
    }
    .msg.assistant {
      align-self: flex-start;
    }
    .meta {
      margin-top: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 12px;
      color: #5d6675;
    }
    .composer {
      border-top: 1px solid #d9dee7;
      padding: 12px;
      background: #ffffff;
    }
    form {
      max-width: 980px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 8px;
      align-items: end;
    }
    textarea {
      resize: vertical;
      min-height: 54px;
      max-height: 220px;
      border: 1px solid #b9c1cf;
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      background: #ffffff;
      color: inherit;
    }
    label {
      display: grid;
      gap: 4px;
      font-size: 12px;
      color: #5d6675;
    }
    input {
      width: 82px;
      border: 1px solid #b9c1cf;
      border-radius: 8px;
      padding: 9px 10px;
      font: inherit;
      background: #ffffff;
      color: inherit;
    }
    button {
      border: 1px solid #1d4ed8;
      border-radius: 8px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 600;
      color: white;
      background: #2563eb;
      cursor: pointer;
      min-height: 40px;
    }
    button.secondary {
      border-color: #b9c1cf;
      color: #2f3541;
      background: #ffffff;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    @media (max-width: 760px) {
      form {
        grid-template-columns: 1fr 1fr;
      }
      textarea {
        grid-column: 1 / -1;
      }
    }
    @media (prefers-color-scheme: dark) {
      body { background: #111318; color: #eef1f6; }
      header, .composer, .msg, textarea, input, button.secondary { background: #181b22; }
      header, .composer, .msg, textarea, input, button.secondary { border-color: #343a46; }
      .msg.user { background: #17243a; border-color: #315176; }
      .status, .meta, label { color: #a8b0bf; }
      button.secondary { color: #eef1f6; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>nano-vllm-jax chat</h1>
      <div id="status" class="status">Checking server...</div>
    </header>
    <main>
      <section id="messages" class="messages"></section>
    </main>
    <section class="composer">
      <form id="form">
        <textarea id="prompt" placeholder="Send a message" required autofocus></textarea>
        <label>Max tokens<input id="maxTokens" type="number" min="1" max="512" value="96"></label>
        <label>Temp<input id="temperature" type="number" min="0" max="2" step="0.1" value="0"></label>
        <div>
          <button id="send" type="submit">Send</button>
          <button class="secondary" id="reset" type="button">Reset</button>
        </div>
      </form>
    </section>
  </div>
  <script>
    const messagesEl = document.getElementById("messages");
    const statusEl = document.getElementById("status");
    const form = document.getElementById("form");
    const promptEl = document.getElementById("prompt");
    const sendEl = document.getElementById("send");
    const resetEl = document.getElementById("reset");
    const maxTokensEl = document.getElementById("maxTokens");
    const temperatureEl = document.getElementById("temperature");
    const turns = [];

    function addMessage(role, text, meta = "") {
      const node = document.createElement("article");
      node.className = `msg ${role}`;
      const body = document.createElement("div");
      body.textContent = text;
      node.appendChild(body);
      if (meta) {
        const metaNode = document.createElement("div");
        metaNode.className = "meta";
        metaNode.textContent = meta;
        node.appendChild(metaNode);
      }
      messagesEl.appendChild(node);
      node.scrollIntoView({block: "end"});
    }

    function buildMessages(nextUserText) {
      return turns.concat([{role: "user", content: nextUserText}]);
    }

    function resultText(payload) {
      if (!payload || !payload.results) return "";
      if (Array.isArray(payload.results)) {
        return payload.results[0]?.text || "";
      }
      return payload.results.text || "";
    }

    function statsText(payload, wallMs) {
      const stats = payload.stats || {};
      const ttft = stats.ttft_ms_mean == null ? "n/a" : `${stats.ttft_ms_mean.toFixed(1)} ms`;
      const tps = stats.tokens_per_second == null ? "n/a" : `${stats.tokens_per_second.toFixed(2)} tok/s`;
      const completion = stats.completion_tokens == null ? "n/a" : `${stats.completion_tokens} tokens`;
      return `TTFT ${ttft} | TPS ${tps} | completion ${completion} | wall ${(wallMs / 1000).toFixed(2)} s`;
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/api/health");
        const payload = await response.json();
        statusEl.textContent = payload.model_loaded ? `Ready | JIT entries ${payload.jit_cache_entries}` : "Server loading";
      } catch (err) {
        statusEl.textContent = "Model server unavailable";
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const userText = promptEl.value.trim();
      if (!userText) return;
      promptEl.value = "";
      addMessage("user", userText);
      sendEl.disabled = true;
      statusEl.textContent = "Generating...";
      const started = performance.now();
      try {
        const response = await fetch("/api/generate_trace", {
          method: "POST",
          headers: {"content-type": "application/json"},
            body: JSON.stringify({
            messages: buildMessages(userText),
            max_tokens: Number(maxTokensEl.value || 96),
            temperature: Number(temperatureEl.value || 0),
            ignore_eos: false
          })
        });
        const payload = await response.json();
        if (!response.ok || payload.error) {
          throw new Error(payload.error || `HTTP ${response.status}`);
        }
        const text = resultText(payload).trim();
        turns.push({role: "user", content: userText});
        turns.push({role: "assistant", content: text});
        addMessage("assistant", text || "(empty response)", statsText(payload, performance.now() - started));
        await refreshHealth();
      } catch (err) {
        addMessage("assistant", `Error: ${err.message || err}`, `wall ${((performance.now() - started) / 1000).toFixed(2)} s`);
        statusEl.textContent = "Request failed";
      } finally {
        sendEl.disabled = false;
        promptEl.focus();
      }
    });

    resetEl.addEventListener("click", () => {
      turns.length = 0;
      messagesEl.innerHTML = "";
      promptEl.focus();
    });

    refreshHealth();
    setInterval(refreshHealth, 5000);
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _proxy_json(backend_url: str, path: str, payload: bytes | None = None) -> tuple[int, bytes, str]:
    url = backend_url.rstrip("/") + path
    headers = {"accept": "application/json"}
    method = "POST" if payload is not None else "GET"
    if payload is not None:
        headers["content-type"] = "application/json"
    req = request.Request(url, data=payload, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=600) as response:
            content_type = response.headers.get("content-type", "application/json")
            return response.status, response.read(), content_type
    except error.HTTPError as exc:
        content_type = exc.headers.get("content-type", "application/json")
        return exc.code, exc.read(), content_type
    except Exception as exc:
        return 502, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json"


def make_handler(backend_url: str):
    class ChatHandler(BaseHTTPRequestHandler):
        server_version = "NanoVLLMJaxChatUI/0.1"

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/health":
                status, body, content_type = _proxy_json(backend_url, "/health")
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/generate_trace":
                _json_response(self, 404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("content-length", "0"))
            except ValueError:
                _json_response(self, 400, {"error": "invalid content-length"})
                return
            payload = self.rfile.read(length)
            status, body, content_type = _proxy_json(backend_url, "/v1/generate_trace", payload)
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ChatHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6789)
    parser.add_argument("--backend-url", default="http://127.0.0.1:6791")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.backend_url))
    print(f"chat_ui_ready=http://{args.host}:{args.port} backend={args.backend_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
