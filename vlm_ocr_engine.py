import os
import sys
import base64
def encode_image(image_path: str) -> str:
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_with_vlm(image_path: str) -> str:
    """
    Uses Gemini Vision to extract handwriting from an image.
    (Groq decommissioned Llama 3.2 Vision preview models, falling back to Gemini)
    """
    # ponytail: load .env manually to avoid extra dependencies
    if os.path.exists(".env"):
        with open(".env") as env_f:
            for line in env_f:
                if line.strip() and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"\'')

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required in .env")

    from google import genai
    from google.genai import types
    from PIL import Image

    client = genai.Client(api_key=api_key)
    
    print("Sending image to Gemini Vision...")
    
    pil_img = Image.open(image_path)
    prompt = "Extract all handwritten text from this image exactly as written. Do not explain, just output the extracted text."
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[pil_img, prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
        ),
    )
    
    return response.text

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("input", "download (5).jpg")
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        sys.exit(1)
        
    try:
        extracted_text = extract_text_with_vlm(img_path)
        print("\n--- Extracted Text from Llama 3.2 Vision ---")
        print(extracted_text)
        
        # Save to extracted_text.txt for evaluator
        with open("extracted_text_vlm.txt", "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print("\nSaved to extracted_text_vlm.txt")
            
    except Exception as e:
        print(f"Error running VLM: {e}")
