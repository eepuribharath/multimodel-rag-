"""Ingest everything in library/ into the search index.

Usage (from repo root):
    venv/bin/python scripts/ingest_library.py

Drop your files into library/ (subfolders fine) and run this. Re-running is
safe: chunk IDs are deterministic, so already-ingested files are overwritten
in place, never duplicated.

Works whether or not the API server is running: embedded Qdrant allows only
one process, so if the server is up the files are uploaded through its
/ingest endpoint; otherwise the pipeline runs in-process.
"""

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARY = REPO_ROOT / "library"
API = "http://localhost:8000"

SUPPORTED = {
    ".md", ".markdown", ".txt", ".rst", ".pdf",
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".wav", ".mp3", ".m4a", ".flac", ".ogg",
}


def server_running() -> bool:
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=2):
            return True
    except Exception:
        return False


def ingest_via_api(path: Path) -> int:
    import mimetypes
    import uuid

    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{path.name}\"\r\nContent-Type: {mime}\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{API}/ingest", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        import json

        return json.loads(response.read())["chunks"]


def main() -> int:
    files = sorted(
        p for p in LIBRARY.rglob("*")
        if p.is_file() and not p.name.startswith(".")
        and p != LIBRARY / "README.txt"  # the folder's own instructions
    )
    if not files:
        print(f"library/ is empty — drop files into {LIBRARY} and re-run")
        return 0

    use_api = server_running()
    print(f"ingesting via {'running server' if use_api else 'local pipeline'}\n")

    pipeline = None
    if not use_api:
        sys.path.insert(0, str(REPO_ROOT / "backend"))
        from app.pipeline import RagPipeline

        pipeline = RagPipeline()

    ok = skipped = failed = 0
    for path in files:
        rel = path.relative_to(LIBRARY)
        if path.suffix.lower() not in SUPPORTED:
            print(f"  skip  {rel}  (unsupported type)")
            skipped += 1
            continue
        try:
            chunks = (
                ingest_via_api(path) if use_api
                else pipeline.ingest_file(str(path))
            )
            print(f"  ok    {rel}  ({chunks} chunk{'s' if chunks != 1 else ''})")
            ok += 1
        except Exception as exc:
            print(f"  FAIL  {rel}  ({exc})")
            failed += 1

    print(f"\n{ok} ingested, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
