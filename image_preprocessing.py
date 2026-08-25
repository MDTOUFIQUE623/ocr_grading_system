import os
import cv2
import numpy as np

OUTPUT_DIR = "output"


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def resize_image(image: np.ndarray, width: int = 1200) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (width, int(h * (width / w)))) if w > width else image


def remove_shadows(gray: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def fix_orientation(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    pts = np.column_stack(np.where(gray < 200))
    if len(pts) == 0:
        return image
    angle = cv2.minAreaRect(pts)[-1]
    angle = -(90 + angle) if angle < -45 else (90 - angle if angle > 45 else angle)
    if abs(angle) < 0.5 or abs(angle) > 45:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


deskew = fix_orientation


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect


def detect_document(image: np.ndarray, min_area_ratio: float = 0.15) -> np.ndarray | None:
    resized = resize_image(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = resized.shape[0] * resized.shape[1]
    scale = image.shape[1] / resized.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_area_ratio * img_area:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return (approx.reshape(4, 2) * scale).astype("float32")
    return None


def perspective_correct(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (w, h))


def enhance_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return remove_shadows(gray)


def create_threshold(image: np.ndarray) -> np.ndarray:
    blur = cv2.bilateralFilter(image, 9, 75, 75)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def preprocess(image_path: str, output_dir: str = OUTPUT_DIR) -> tuple[np.ndarray, np.ndarray]:
    os.makedirs(output_dir, exist_ok=True)
    img = load_image(image_path)
    doc = detect_document(img)
    corrected = perspective_correct(img, doc) if doc is not None else img
    enhanced = enhance_image(corrected)
    deskewed = fix_orientation(enhanced)
    threshold = create_threshold(deskewed)

    cv2.imwrite(os.path.join(output_dir, "original_corrected.jpg"), corrected)
    cv2.imwrite(os.path.join(output_dir, "enhanced_gray.jpg"), enhanced)
    cv2.imwrite(os.path.join(output_dir, "threshold.jpg"), threshold)
    return deskewed, threshold


if __name__ == "__main__":
    sample_path = os.path.join("input", "sample_answer.jpg")
    if os.path.exists(sample_path):
        preprocess(sample_path)