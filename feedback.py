import json
from datetime import datetime

def save_feedback(record: dict, filepath: str = "feedback.jsonl"):
    """
    Saves evaluation feedback for future system improvement.
    """
    record["timestamp"] = datetime.utcnow().isoformat()
    # ponytail: simple append to jsonl, no database needed yet
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    # ponytail: self-check demo
    demo_record = {
        "extracted_text": "BOD measures oxgen",
        "reference": "BOD measures oxygen required...",
        "score": 4,
        "llm_feedback": "Minor typo in oxygen.",
        "user_correction": "Score should be 5."
    }
    save_feedback(demo_record)
    print("Feedback saved to feedback.jsonl")
