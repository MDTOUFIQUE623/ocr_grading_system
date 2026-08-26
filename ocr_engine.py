import os
os.environ["FLAGS_use_onednn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from functools import lru_cache
import cv2
import numpy as np


@lru_cache(maxsize=1)
def load_paddleocr():
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(ocr_version="PP-OCRv4", lang="en", enable_mkldnn=False)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_trocr():
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer, ViTImageProcessor
        # ponytail: transformers v5.15+ defaults to fast tokenizer which doesn't exist for trocr-base-handwritten
        image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        return processor, model
    except Exception as e:
        print(f"ERROR loading TrOCR: {e}")
        return None, None


def run_paddleocr(ocr, image: np.ndarray) -> list:
    if ocr is None:
        return []
    res = ocr.ocr(image)
    if not res:
        return []

    target = res[0] if isinstance(res, list) and len(res) > 0 else res
    if isinstance(target, dict):
        polys = target.get('dt_polys', target.get('rec_polys', []))
        texts = target.get('rec_texts', [])
        scores = target.get('rec_scores', [])
        items = []
        for i, poly in enumerate(polys):
            text = texts[i] if i < len(texts) else ""
            score = scores[i] if i < len(scores) else 0.0
            items.append((poly, (text, score)))
        return items

    return target if isinstance(target, list) else []


def run_trocr(crop: np.ndarray) -> str:
    processor, model = load_trocr()
    if processor is None or model is None:
        return ""
    try:
        from PIL import Image
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_new_tokens=50) # ponytail: max_new_tokens silences warning
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"TrOCR Error: {e}")
        return ""


def group_into_lines(detections: list, y_threshold_ratio: float = 0.5) -> list[list]:
    """Group paddle word boxes into lines by y-coordinate proximity."""
    if not detections:
        return []
    
    def get_y_center(item):
        return np.mean(np.array(item[0])[:, 1])
        
    def get_height(item):
        pts = np.array(item[0])
        return np.max(pts[:, 1]) - np.min(pts[:, 1])
        
    def get_x_min(item):
        return np.min(np.array(item[0])[:, 0])

    sorted_dets = sorted(detections, key=get_y_center)
    lines = []
    current_line = [sorted_dets[0]]
    
    for item in sorted_dets[1:]:
        h = get_height(item)
        threshold = h * y_threshold_ratio
        if abs(get_y_center(item) - get_y_center(current_line[-1])) < threshold:
            current_line.append(item)
        else:
            lines.append(sorted(current_line, key=get_x_min))
            current_line = [item]
            
    if current_line:
        lines.append(sorted(current_line, key=get_x_min))
        
    return lines


def crop_line(image: np.ndarray, line_items: list, pad: int = 10) -> tuple[np.ndarray, list]:
    """Crop a bounding box that covers all items in a line."""
    all_pts = []
    for item in line_items:
        all_pts.extend(item[0])
    pts = np.array(all_pts, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    y1, y2 = max(0, y - pad), min(image.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(image.shape[1], x + w + pad)
    crop = image[y1:y2, x1:x2]
    return crop, [x1, y1, x2, y2]


def process_image(image_input: str | np.ndarray, confidence_threshold: float = 0.85, pad: int = 5) -> list[dict]:
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError("image_input must be a file path (str) or a numpy array")

    ocr = load_paddleocr()
    detections = run_paddleocr(ocr, image)

    if not detections:
        return []

    lines = group_into_lines(detections)
    blocks = []
    
    for idx, line_items in enumerate(lines):
        paddle_text = " ".join([item[1][0] for item in line_items])
        paddle_conf = np.mean([item[1][1] for item in line_items]) if line_items else 0.0
        
        line_crop, bbox = crop_line(image, line_items, pad=pad)
        trocr_text = run_trocr(line_crop) if line_crop.size > 0 else ""
        
        # ponytail: prioritize TrOCR for handwritten answers, fallback to paddle
        final_text = trocr_text if trocr_text.strip() else paddle_text
        source = "trocr" if trocr_text.strip() else "paddleocr"
        
        blocks.append({
            "id": idx + 1,
            "box": bbox,
            "text": final_text,
            "paddle_text": paddle_text,
            "trocr_text": trocr_text,
            "paddle_confidence": float(paddle_conf),
            "source": source
        })

    return blocks


if __name__ == "__main__":
    path = "output/2fca7d6c/enhanced_gray.jpg"
    if os.path.exists(path):
        print(f"\n--- Testing {path} ---")
        res = process_image(path)
        print(f"Total blocks for {path}: {len(res)}")
        if res:
            print("First 3 blocks sample:", res[:3])
    else:
        print(f"File {path} not found.")
