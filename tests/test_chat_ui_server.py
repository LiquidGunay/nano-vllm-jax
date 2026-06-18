import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request

from tools.chat_ui_server import make_handler


class _BackendHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(
            {"status": "healthy", "model_loaded": True, "jit_cache_entries": 3}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/generate_trace":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        body = json.dumps(
            {
                "results": {"text": "Hello from backend", "token_ids": [1, 2]},
                "events": [],
                "stats": {
                    "ttft_ms_mean": 12.5,
                    "tokens_per_second": 42.0,
                    "completion_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_server(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_chat_ui_serves_page_and_proxies_trace():
    backend = _start_server(_BackendHandler)
    backend_url = f"http://127.0.0.1:{backend.server_address[1]}"
    ui = _start_server(make_handler(backend_url))
    ui_url = f"http://127.0.0.1:{ui.server_address[1]}"

    try:
        page = request.urlopen(ui_url + "/", timeout=5).read().decode()
        assert "nano-vllm-jax chat" in page
        assert "/api/generate_trace" in page
        assert "messages: buildMessages" in page
        assert "buildPrompt" not in page

        health = json.loads(request.urlopen(ui_url + "/api/health", timeout=5).read())
        assert health["model_loaded"] is True

        payload = json.dumps({"prompt": "hello", "max_tokens": 2}).encode()
        req = request.Request(
            ui_url + "/api/generate_trace",
            data=payload,
            headers={"content-type": "application/json"},
            method="POST",
        )
        traced = json.loads(request.urlopen(req, timeout=5).read())
        assert traced["results"]["text"] == "Hello from backend"
        assert traced["stats"]["tokens_per_second"] == 42.0
    finally:
        ui.shutdown()
        ui.server_close()
        backend.shutdown()
        backend.server_close()
