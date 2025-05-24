import json
from typing import List, Dict, Optional
from utils.memory_loader import load_core_memories
from pathlib import Path
from utils.text_utils import normalize_list

class MemoryRetrievalAgent:
    def __init__(self, memory_file: str = "data/core_memories.jsonl"):
        self.memory_file = Path(memory_file)
        self.memories = self._load_memories()
        self.tag_index = self._build_tag_index(self.memories)

    def _load_memories(self) -> List[Dict]:
        return load_core_memories(str(self.memory_file))

    def _build_tag_index(self, memories: List[Dict]) -> Dict[str, List[Dict]]:
        index = {}
        for mem in memories:
            for tag in normalize_list(mem["tags"]):
                index.setdefault(tag, []).append(mem)
        return index

    def _get_emotion_similarity(self, query_emotion: str, memory_emotion: str) -> float:
        if query_emotion == memory_emotion:
            return 1.0
        similarity_map = {
            ("curious", "calm"): 0.5,
            ("curious", "content"): 0.4,
            ("curious", "happy"): 0.3,
            ("happy", "content"): 0.7,
            ("anxious", "calm"): 0.6,
            ("anxious", "confused"): 0.5,
            ("curious", "confused"): 0.3,
        }
        return similarity_map.get((query_emotion, memory_emotion), 0.0)

    def retrieve(
        self,
        query_keywords: List[str],
        emotion: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict]:
        normalized_keywords = set(query_keywords)

        candidates = {}
        for keyword in normalized_keywords:
            for mem in self.tag_index.get(keyword, []):
                candidates[mem["id"]] = mem
        filtered = list(candidates.values())

        # Score by normalized tag overlap and emotion similarity
        def score(mem):
            normalized_tags = set(normalize_list(mem["tags"]))
            overlap_score = len(normalized_tags & normalized_keywords)
            emotion_bonus = self._get_emotion_similarity(emotion, mem["emotion"])
            return (3 * overlap_score) + emotion_bonus

        for mem in filtered:
            print(f"\n[Memory Scoring Debug]")
            print(f"Memory: {mem['text']}")
            print(f"Tags: {mem['tags']}")
            print(f"Score: {score(mem)}")

        sorted_memories = sorted(filtered, key=score, reverse=True)
        return sorted_memories[:top_k]