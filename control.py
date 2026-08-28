"""Control channel: a unix socket the running pipeline listens on.

Gesture recognition will never be perfect, so every effect must also be
reachable by hand. One socket serves all the front ends — the Tk panel, the
`aicamctl` command, and later any hotkey daemon — so the pipeline itself only
ever deals with one kind of message.
"""

import json
import os
import queue
import socket
import threading


def default_socket_path():
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/aicam-{os.getuid()}"
    os.makedirs(runtime, exist_ok=True)
    return os.path.join(runtime, "aicam.sock")


class ControlServer:
    """Accepts newline-delimited JSON objects and queues them for the main loop.

    The socket runs on its own thread so a slow or stalled client can never hold
    up frame delivery; the main loop only ever does a non-blocking drain.
    """

    def __init__(self, path=None):
        self.path = path or default_socket_path()
        self.commands = queue.Queue()
        self.state = {}
        self._stop = threading.Event()

        if os.path.exists(self.path):
            os.unlink(self.path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.path)
        self._sock.listen(8)
        self._sock.settimeout(0.5)

        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn:
            conn.settimeout(2.0)
            try:
                data = conn.recv(65536).decode("utf-8")
            except (socket.timeout, OSError):
                return
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self._reply(conn, {"ok": False, "error": "invalid json"})
                    continue
                if msg.get("query") == "state":
                    self._reply(conn, {"ok": True, "state": self.state})
                else:
                    self.commands.put(msg)
                    self._reply(conn, {"ok": True})

    @staticmethod
    def _reply(conn, obj):
        try:
            conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))
        except OSError:
            pass

    def drain(self):
        """Non-blocking: every command queued since the last call."""
        out = []
        while True:
            try:
                out.append(self.commands.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        finally:
            if os.path.exists(self.path):
                os.unlink(self.path)


def send(message, path=None, timeout=2.0):
    """Send one command and return the reply. Raises ConnectionError if nobody listens."""
    path = path or default_socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect(path)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise ConnectionError(f"no aicam listening on {path}") from exc
        s.sendall((json.dumps(message) + "\n").encode("utf-8"))
        try:
            return json.loads(s.recv(65536).decode("utf-8") or "{}")
        except (socket.timeout, json.JSONDecodeError):
            return {}
