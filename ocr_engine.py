
import os
from functools import lru_cache
import cv2
import numpy as np


@lru_cache(maxsize=1)
def load_easyocr():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as e:
        print(f"ERROR loading EasyOCR: {e}")
        return None


def process_image(image_input: str | np.ndarray) -> list[dict]:
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError("image_input must be a file path or numpy array")

    reader = load_easyocr()
    if reader is None:
        return []

    results = reader.readtext(image)
    blocks = []
    for idx, (bbox, text, conf) in enumerate(results):
        if text.strip():
            pts = [[int(p[0]), int(p[1])] for p in bbox]
            blocks.append({
                "id": idx + 1,
                "box": pts,
                "text": text,
                "confidence": float(conf),
                "source": "easyocr"
            })
    return blocks


if __name__ == "__main__":
    path = "input/test_safe.jpg"
    if not os.path.exists(path):
        print(f"File {path} not found.")
    else:
        print(f"\n--- Testing {path} ---")
        res = process_image(path)
        print(f"Total blocks: {len(res)}")
        for b in res:
            print(f"  [{b['id']}] (conf={b['confidence']:.2f}) {b['text']}")
