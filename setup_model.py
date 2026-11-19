"""Helper script to download and extract dlib's 68-point facial landmark model."""

import bz2
import os
import urllib.request

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"


def ensure_landmark_model(model_path: str = MODEL_FILENAME) -> str:
    """Download and decompress the 68-point facial landmark predictor if missing."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 50_000_000:
        return model_path

    print(f"[INFO] Downloading {MODEL_FILENAME} (~64 MB compressed)...")
    compressed_path = f"{model_path}.bz2"

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100.0, (block_num * block_size * 100.0) / total_size)
            print(f"\r[INFO] Download progress: {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, compressed_path, reporthook=_progress)
    print("\n[INFO] Extracting model file...")

    with bz2.BZ2File(compressed_path, "rb") as src, open(model_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)

    if os.path.exists(compressed_path):
        os.remove(compressed_path)

    print(f"[INFO] Model ready at: {os.path.abspath(model_path)}")
    return model_path


if __name__ == "__main__":
    ensure_landmark_model()
