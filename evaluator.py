import os
import json
from google import genai
from google.genai import types

def evaluate_answer(extracted_text: str, reference_answer: str, max_score: int = 5) -> dict:
    """
    Evaluates the extracted OCR text against a reference answer using Gemini.
    Returns a dictionary with score, feedback, and the original text.
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
        # ponytail: fail fast if no key
        raise ValueError("GEMINI_API_KEY environment variable is required")

    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert teacher grading a student's answer.
    
    Here is the reference answer:
    {reference_answer}
    
    Here is the student's answer (extracted via OCR, may contain slight typos):
    {extracted_text}
    
    Grade the student's answer out of {max_score}. 
    Focus on meaning and concepts rather than exact wording or OCR spelling errors.
    
    Return the result in JSON format with exactly these two keys:
    - "score": (integer) the score out of {max_score}
    - "feedback": (string) a short 1-2 sentence explanation of why they got this score.
    """
    
    # ponytail: one-line structured output request
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    
    try:
        result = json.loads(response.text)
    except json.JSONDecodeError:
        result = {"score": 0, "feedback": "Failed to parse LLM response."}
        
    result["extracted_text"] = extracted_text
    return result

def generate_demo_reference(extracted_text: str) -> str:
    """Auto-generates a reference answer based on the student's text for demo purposes."""
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = f"Based on this student's answer, infer the likely original question and write a perfect, concise reference answer for it. Do not include the question, just the reference answer.\nStudent answer:\n{extracted_text}"
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return response.text.strip()

if __name__ == "__main__":
    import sys
    
    text_file = sys.argv[1] if len(sys.argv) > 1 else "extracted_text_vlm.txt"
    if not os.path.exists(text_file):
        print(f"Error: {text_file} not found.")
        sys.exit(1)
        
    with open(text_file, "r", encoding="utf-8") as f:
        extracted = f.read()
        
    # hardcoded sample reference for testing
    ref = "BOD (Biochemical Oxygen Demand) measures the amount of oxygen required by microorganisms to decompose organic matter. COD (Chemical Oxygen Demand) measures the total amount of oxygen required to oxidize all organic and inorganic compounds in water."
    
    print("Evaluating...")
    res = evaluate_answer(extracted, ref)
    print(json.dumps(res, indent=2))
