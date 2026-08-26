import os
import shutil
import vlm_ocr_engine
import ocr_engine
import image_preprocessing
import evaluator
from difflib import SequenceMatcher
import json

def run_test():
    orig_path = r"input\76a581b4_bad handwriting but f it, we ball😮_💨.jpg"
    safe_path = "input/test_safe.jpg"
    
    if not os.path.exists(orig_path):
        print(f"File not found: {orig_path}")
        return
        
    shutil.copy(orig_path, safe_path)
    print(f"Copied to {safe_path}")
    
    print("\n--- Preprocessing ---")
    image_preprocessing.preprocess(safe_path, output_dir="output/test_safe")
    
    print("\n--- Running VLM OCR Engine (Gemini) ---")
    vlm_text = vlm_ocr_engine.extract_text_with_vlm(safe_path)
    print(f"VLM Output:\n{vlm_text}")
    
    print("\n--- Running Traditional OCR Engine (Paddle+TrOCR) ---")
    ocr_blocks = ocr_engine.process_image(safe_path)
    ocr_text = "\n".join([b["text"] for b in ocr_blocks])
    print(f"OCR Output:\n{ocr_text}")
    
    similarity = SequenceMatcher(None, vlm_text, ocr_text).ratio()
    print(f"\n--- Accuracy / Similarity Metric ---")
    print(f"Similarity: {similarity:.1%}")
    
    print("\n--- Generating Demo Reference ---")
    ref = evaluator.generate_demo_reference(vlm_text)
    print(f"Reference:\n{ref}")
    
    print("\n--- Evaluating Answer ---")
    result = evaluator.evaluate_answer(vlm_text, ref)
    print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    run_test()
