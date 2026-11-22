"""Run the local coal-flow OBB annotation workbench."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
from urllib.parse import parse_qs, urlsplit

from dataset import Dataset
from export import build_export
from store import Conflict, Store


ROOT = Path(__file__).resolve().parent
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
FRAME = re.compile(r"/api/frames/(?P<sample>[a-zA-Z0-9_-]+)(?P<image>/image)?")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.dispatch("GET")

    def do_PUT(self):
        self.dispatch("PUT")

    def is_local(self):
        try:
            host = self.headers.get("Host", "").lower()
            allowed = {f"{name}:{self.server.server_port}" for name in ("localhost", "127.0.0.1")}
            if host not in allowed:
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin == f"http://{host}"
        except ValueError:
            return False

    def dispatch(self, method):
        try:
            if not self.is_local():
                self.send_json({"error": "标注服务仅供本机使用。"}, HTTPStatus.FORBIDDEN)
                return
            url = urlsplit(self.path)
            path, query = url.path, parse_qs(url.query)
            store = self.server.store
            route = FRAME.fullmatch(path)
            if method == "PUT" and route and not route["image"]:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("保存内容须为 JSON。")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2 * 1024**2:
                    raise ValueError("保存请求为空或超出大小限制。")
                self.send_json(store.save(route["sample"], json.loads(self.rfile.read(size))))
            elif method == "GET":
                self.read_route(path, query, route)
            else:
                raise KeyError("页面或接口不存在。")
        except Conflict as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except (ValueError, UnicodeError, OverflowError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except (KeyError, FileNotFoundError) as exc:
            self.send_json({"error": str(exc.args[0])}, HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self.log_error("%s: %s", type(exc).__name__, exc)
            self.send_json({"error": "本机服务出错，请查看启动窗口。"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_route(self, path, query, route):
        store = self.server.store
        if path == "/api/health":
            self.send_json({"status": "ok"})
        elif path == "/api/frames":
            self.send_json(store.catalog(query.get("batch", ["pilot"])[0]))
        elif path == "/api/export":
            if query.get("factor", ["0.5"]) != ["0.5"]:
                raise ValueError("导出范围固定为 0.5L。")
            with build_export(store.dataset, store.export_frames(), temp_dir=store.path.parent) as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(0)
                self.send_headers("application/zip", size, disposition='attachment; filename="4911-AB-0.5L.zip"')
                shutil.copyfileobj(stream, self.wfile)
        elif route:
            if route["image"]:
                self.send_bytes(store.dataset.image_path(route["sample"]).read_bytes(), "image/jpeg")
            else:
                self.send_json(store.frame(route["sample"]))
        else:
            name, content_type = STATIC.get(path.removeprefix("/static") if path.startswith("/static/") else path, (None, None))
            if name is None:
                raise KeyError("页面或接口不存在。")
            self.send_bytes((ROOT / "static" / name).read_bytes(), content_type)

    def send_headers(self, content_type, size, status=HTTPStatus.OK, disposition=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()

    def send_bytes(self, content, content_type, status=HTTPStatus.OK):
        self.send_headers(content_type, len(content), status)
        self.wfile.write(content)

    def send_json(self, content, status=HTTPStatus.OK):
        self.send_bytes(json.dumps(content, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)


def main():
    parser = argparse.ArgumentParser(description="煤流异物与邻域物料标注")
    parser.add_argument("--port", type=int, default=5188)
    parser.add_argument("--dataset", type=Path, default=ROOT.parent.parent / "new_dataset")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".local")
    args = parser.parse_args()
    store = Store(Dataset(args.dataset), args.data_dir.resolve())
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.store = store
        print(f"YOLO OBB workbench: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
