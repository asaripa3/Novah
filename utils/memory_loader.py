import json

def load_core_memories(filepath="data/core_memories.jsonl"):
    print(f"Loading memories from: {filepath}")  # Debug log
    memories = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                memory = json.loads(line.strip())
                memories.append(memory)

    except Exception as e:
        return []  # Return empty list on error to prevent crashes
    return memories

def save_core_memories(memories, filepath="data/core_memories.jsonl"):
    try:
        with open(filepath, "w") as f:
            for memory in memories:
                f.write(json.dumps(memory) + "\n")
        return True
    except Exception as e:
        return False