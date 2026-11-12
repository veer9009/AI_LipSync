"""Local, dependency-free browser interface for Wav2Lip.

Run with: venv\\Scripts\\python.exe web_app.py
Then open: http://127.0.0.1:7860
"""

from __future__ import annotations

import cgi
import html
import os
import shutil
import subprocess
import sys
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
RESULT_DIR = BASE_DIR / "web_results"
CHECKPOINT = BASE_DIR / "checkpoints" / "wav2lip_gan.pth"
HOST, PORT = "127.0.0.1", 7860
ALLOWED_FACE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4"}


def page(message: str = "", error: bool = False, allow_html: bool = False) -> bytes:
    notice = ""
    if message:
        kind = "error" if error else "success"
        content = message if allow_html else html.escape(message)
        notice = f'<p class="{kind}">{content}</p>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wav2Lip — Local Studio</title><style>
body{{font-family:system-ui,sans-serif;max-width:720px;margin:48px auto;padding:0 20px;background:#10131a;color:#edf1f7}}
h1{{margin-bottom:4px}} .muted{{color:#aab4c3}} form{{margin-top:28px;padding:24px;background:#1b2130;border-radius:12px}}
label{{display:block;margin:18px 0 6px;font-weight:650}} input{{width:100%;box-sizing:border-box;padding:10px;background:#0f1420;color:#fff;border:1px solid #3b4558;border-radius:7px}}
button{{margin-top:24px;background:#6d5dfc;border:0;border-radius:7px;color:#fff;padding:12px 18px;font-weight:700;cursor:pointer}}
.error,.success{{padding:12px;border-radius:7px}} .error{{background:#4b1f27;color:#ffdbe0}} .success{{background:#173d30;color:#c9f7de}}
</style></head><body><h1>Wav2Lip Local Studio</h1><p class="muted">Runs only on this laptop. Generation can take several minutes on CPU.</p>{notice}
<form action="/generate" method="post" enctype="multipart/form-data" onsubmit="this.querySelector('button').disabled=true;this.querySelector('button').textContent='Generating…';">
<label for="face">Face video or image</label><input id="face" name="face" type="file" accept="video/*,image/*" required>
<label for="audio">Audio</label><input id="audio" name="audio" type="file" accept="audio/*,video/*" required>
<button type="submit">Generate lip-sync video</button></form></body></html>""".encode()


class App(BaseHTTPRequestHandler):
    def send_html(self, content: bytes, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_html(page())
            return
        if self.path.startswith("/results/"):
            name = Path(self.path).name
            target = RESULT_DIR / name
            if target.suffix == ".mp4" and target.is_file():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(target.stat().st_size))
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.end_headers()
                with target.open("rb") as video:
                    shutil.copyfileobj(video, self.wfile)
                return
        self.send_html(page("Page not found.", True), HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/generate":
            self.send_html(page("Page not found.", True), HTTPStatus.NOT_FOUND)
            return
        if not CHECKPOINT.is_file():
            self.send_html(page(f"Checkpoint not found: {CHECKPOINT.name}", True), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if shutil.which("ffmpeg") is None:
            self.send_html(page("FFmpeg is not installed or not on PATH. Install FFmpeg, restart the terminal, then start this app again.", True), HTTPStatus.SERVICE_UNAVAILABLE)
            return

        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
        face, audio = form["face"], form["audio"]
        if not getattr(face, "filename", "") or not getattr(audio, "filename", ""):
            self.send_html(page("Choose both a face video/image and an audio file.", True), HTTPStatus.BAD_REQUEST)
            return
        face_ext = Path(face.filename).suffix.lower()
        audio_ext = Path(audio.filename).suffix.lower()
        if face_ext not in ALLOWED_FACE_EXTENSIONS or audio_ext not in ALLOWED_AUDIO_EXTENSIONS:
            self.send_html(page("Unsupported file type. Use a common video/image for face and a common audio file.", True), HTTPStatus.BAD_REQUEST)
            return

        job = uuid.uuid4().hex
        job_dir = UPLOAD_DIR / job
        job_dir.mkdir(parents=True, exist_ok=True)
        face_path, audio_path = job_dir / f"face{face_ext}", job_dir / f"audio{audio_ext}"
        with face_path.open("wb") as output:
            shutil.copyfileobj(face.file, output)
        with audio_path.open("wb") as output:
            shutil.copyfileobj(audio.file, output)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULT_DIR / f"lipsync-{job}.mp4"
        command = [sys.executable, "inference.py", "--checkpoint_path", str(CHECKPOINT), "--face", str(face_path), "--audio", str(audio_path), "--outfile", str(output_path)]
        try:
            subprocess.run(command, cwd=BASE_DIR, check=True, timeout=3600, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except subprocess.TimeoutExpired:
            self.send_html(page("Generation timed out after one hour. Try a shorter, lower-resolution video.", True), HTTPStatus.GATEWAY_TIMEOUT)
            return
        except subprocess.CalledProcessError as exc:
            detail = (exc.stdout or "Unknown error").splitlines()[-1:]
            self.send_html(page("Generation failed: " + (detail[0] if detail else "unknown error"), True), HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_html(page(f'Finished! <a href="/results/{output_path.name}">Download your video</a>', allow_html=True))

    def log_message(self, format: str, *args: object) -> None:
        print("[web]", format % args)


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    print(f"Open http://{HOST}:{PORT} in your browser. Press Ctrl+C to stop.")
    HTTPServer((HOST, PORT), App).serve_forever()
