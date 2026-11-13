"""Serve dp4a_probe_web/probe.html with the probe data and yaar's onnxruntime-web 1.30.0 dist.

POST /log appends to <data>/browser.log, POST /result writes <data>/result.json.

    python project/webgpu_int8/src/dp4a_serve.py [--port 8765]
"""

import argparse
import functools
import http.server
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
ORT = os.path.expanduser(
    "~/yaar/node_modules/.bun/onnxruntime-web@1.30.0/node_modules/onnxruntime-web/dist"
)


class H(http.server.SimpleHTTPRequestHandler):
    data = os.path.join(ROOT, "output/webgpu_int8/dp4a_probe")

    def translate_path(self, path):
        path = path.split("?")[0]
        if path.startswith("/ort/"):
            return os.path.join(ORT, path[5:])
        if path.startswith("/data/"):
            return os.path.join(self.data, path[6:])
        return os.path.join(HERE, "dp4a_probe_web", path.lstrip("/") or "probe.html")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/log":
            with open(os.path.join(self.data, "browser.log"), "ab") as f:
                f.write(body + b"\n")
        else:
            with open(os.path.join(self.data, "result.json"), "wb") as f:
                f.write(body)
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
