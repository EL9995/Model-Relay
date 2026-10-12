#!/usr/bin/env python3
"""Local web page for the relay: type a task, watch it travel down the chain and back up.
Also runs blind, cipher, translate and vision (read an image, then hand the text on) modes.

Run:   python3 server.py        then open http://localhost:8765
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import relay

PORT = 8765
PAGE = Path(__file__).with_name("ui.html")
RAM_GB = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)  # GiB, as Apple advertises it
MAX_BODY = 60 * 2**20  # uploaded photos, base64-encoded


def caps(name):
    try:
        return relay.capabilities(name)
    except OSError:
        return []


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            self.reply(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif url.path == "/models":
            try:
                tags = relay.post_get("/api/tags").get("models", [])
                with ThreadPoolExecutor(8) as pool:
                    found = list(pool.map(caps, [m["name"] for m in tags]))
                models = [{"name": m["name"], "gb": round(m["size"] / 1e9, 1), "vision": "vision" in c,
                           "tools": "tools" in c} for m, c in zip(tags, found)]
                body = {"models": models, "default": relay.DEFAULT_CHAIN.split(","), "ram_gb": RAM_GB,
                        "languages": relay.LANGUAGES[:-1], "route": relay.DEFAULT_ROUTE}
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
        elif url.path.startswith("/images/"):
            name = url.path[len("/images/"):]
            path = relay.RUNS / "images" / name
            if re.fullmatch(r"[0-9a-f]{16}\.jpg", name) and path.is_file():
                self.reply(200, "image/jpeg", path.read_bytes())
            else:
                self.reply(404, "text/plain", b"no such image")
        elif url.path == "/run" and q.get("task"):
            self.run_relay(q)
        else:
            self.reply(404, "text/plain", b"not found")

    def do_POST(self):
        """Same as GET /run, but the settings come as JSON, so an image fits."""
        size = int(self.headers.get("Content-Length") or 0)
        if urlparse(self.path).path != "/run" or not 0 < size <= MAX_BODY:
            return self.reply(413 if size > MAX_BODY else 404, "text/plain", b"bad request")
        try:
            q = json.loads(self.rfile.read(size))
        except ValueError:
            return self.reply(400, "text/plain", b"body isn't JSON")
        self.run_relay(q)

    def run_relay(self, q):
        """Stream relay events to the page as server-sent events, one per line pair."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(event):
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

        split = lambda v: v if isinstance(v, list) else [x for x in str(v or "").split(",") if x]
        mode, cipher, sab = q.get("mode", "relay"), q.get("cipher", "base64"), str(q.get("saboteur", ""))
        try:
            if mode == "cipher" and cipher not in [*relay.CIPHERS, "mix"]:
                raise ValueError(f"unknown cipher {cipher!r}")
            if mode == "vision" and q.get("format", "text") not in relay.FORMATS:
                raise ValueError(f"unknown format {q.get('format')!r}")
            r = relay.run(q.get("task", ""), split(q.get("chain")) or relay.DEFAULT_CHAIN.split(","), mode,
                          q.get("rule", ""), send, saboteur=int(sab) if sab.isdigit() else None,
                          cipher_name=cipher, route=split(q.get("langs")),
                          images=q.get("images") or [q.get("image")], fmt=q.get("format", "text"),
                          combiner=q.get("combiner") or "granite4.2:8b", strategy=q.get("strategy", "batch"))
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
