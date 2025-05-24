import json

def load_core_memories(filepath="data/core_memories.jsonl"):
    memories = []
    with open(filepath, "r") as f:
        for line in f:
            memories.append(json.loads(line.strip()))
    return memories