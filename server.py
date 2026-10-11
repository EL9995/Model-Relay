#!/usr/bin/env python3
"""Local web page for the relay: type a task, watch it travel down the chain and back up.

Run:   python3 server.py        then open http://localhost:8765
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import relay

PORT = 8765
PAGE = Path(__file__).with_name("ui.html")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            self.reply(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif url.path == "/models":
            try:
                models = [{"name": m["name"], "gb": round(m["size"] / 1e9, 1)}
                          for m in relay.post_get("/api/tags").get("models", [])]
                body = {"models": models, "default": relay.DEFAULT_CHAIN.split(",")}
            except OSError as e:
                body = {"error": f"Can't reach Ollama at {relay.OLLAMA}. Is the app running? ({e})"}
            self.reply(200, "application/json", json.dumps(body).encode())
        elif url.path == "/runs":
            self.reply(200, "application/json", json.dumps(relay.list_runs()).encode())
        elif url.path.startswith("/runs/"):
            r = relay.load_run(url.path[len("/runs/"):])
            if r:
                self.reply(200, "application/json", json.dumps(r, ensure_ascii=False).encode())
            else:
                self.reply(404, "text/plain", b"no such run")
        elif url.path == "/run" and q.get("task"):
            self.run_relay(q["task"], [m for m in q.get("chain", "").split(",") if m],
                           q.get("mode", "relay"), q.get("rule", ""))
        else:
            self.reply(404, "text/plain", b"not found")

    def run_relay(self, task, chain, mode, rule):
        """Stream relay events to the page as server-sent events, one per line pair."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(event):
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

        try:
            r = relay.run(task, chain or relay.DEFAULT_CHAIN.split(","), mode, rule, send)
            send({"type": "done", **{k: v for k, v in r.items() if k != "trace"}})
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page was closed mid-run
        except Exception as e:
            send({"type": "error", "message": str(e)})

    def reply(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Model Relay UI on http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
