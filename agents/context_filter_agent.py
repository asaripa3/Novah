from typing import List, Dict, Optional
from utils.text_utils import normalize_list
from datetime import datetime

class ContextFilterAgent:
    def __init__(self, known_vocabulary: Optional[List[str]] = None):
        self.known_vocabulary = set(normalize_list(known_vocabulary or []))

    def update_known_vocabulary(self, new_vocab: List[str]):
        """Replace known vocabulary with an updated list."""
        self.known_vocabulary = set(normalize_list(new_vocab or []))

    def filter(self, memories: List[Dict], query_emotion: Optional[str] = None) -> List[Dict]:
        def score(mem):
            normalized_vocab = set(normalize_list(mem.get("vocabulary", [])))
            vocab_score = len(normalized_vocab & self.known_vocabulary)

            if vocab_score == 0:
                return -1e6  # Effectively discards this memory

            importance_bonus = 0.5 * mem.get("importance_score", 0)

            # Recency bonus
            recency_bonus = 0.0
            try:
                last_used = datetime.strptime(mem.get("last_used", ""), "%Y-%m-%d")
                days_ago = (datetime.today() - last_used).days
                recency_bonus = max(0, 1.0 - (days_ago / 30))  # Linearly decay over 30 days
            except Exception:
                pass

            return vocab_score + importance_bonus + recency_bonus

        sorted_memories = sorted(memories, key=score, reverse=True)
        return sorted_memories