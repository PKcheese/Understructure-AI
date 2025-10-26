# call_api.py
from pathlib import Path
import requests

API_URL = "http://127.0.0.1:8000/maquette"

# adjust if the filename differs
IMAGE_PATH = Path.home() / "Downloads" / "yoga_body_images-slide-NY4R-jumbo.jpg"
OUTPUT_GLB = Path.cwd() / "maquette_modified_rings.glb"


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    with IMAGE_PATH.open("rb") as fh:
        files = {"image": (IMAGE_PATH.name, fh, "image/jpeg")}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        OUTPUT_GLB.write_bytes(response.content)
        print(f"Saved GLB to {OUTPUT_GLB}")
    else:
        print(f"Request failed: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
