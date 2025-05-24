

import nltk
from nltk.tokenize import word_tokenize
from utils.text_utils import normalize_list

def update_profile_from_input(user_input, parsed, profile):
    tokens = word_tokenize(user_input)
    pos_tags = nltk.pos_tag(tokens)
    filtered_tokens = [word for word, pos in pos_tags if pos.startswith('NN') or pos.startswith('VB')]
    new_words = normalize_list(filtered_tokens)
    updated = False

    for word in new_words:
        if word not in profile["known_vocabulary"]:
            profile["known_vocabulary"].append(word)
            updated = True

    if parsed["emotion"] in {"anxious", "sad", "angry"}:
        for word in new_words:
            if word not in profile.get("trigger_words", []):
                profile["trigger_words"].append(word)
                updated = True

    frequent_topic_keywords = {"books", "games", "videos", "animals", "trains"}
    for word in new_words:
        if word in frequent_topic_keywords and word not in profile.get("preferred_topics", []):
            profile["preferred_topics"].append(word)
            updated = True

    return updated