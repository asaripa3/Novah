import json

def load_profile(filepath: str):
    """Load the first JSON object from a JSONL profile file."""
    with open(filepath, "r") as f:
        return json.loads(f.readline().strip())

def save_profile(profile: dict, filepath: str):
    """Save the profile dictionary to a JSONL file (single line)."""
    with open(filepath, "w") as f:
        f.write(json.dumps(profile, ensure_ascii=False) + "\n")
