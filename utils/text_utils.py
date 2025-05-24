import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("tokenizers/punkt/english.pickle")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab/english")
except LookupError:
    nltk.download("punkt_tab")

import re
try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    from nltk import pos_tag
    lemmatizer = WordNetLemmatizer()
except ImportError:
    lemmatizer = None

def get_wordnet_pos(word):
    """Map POS tag to WordNet POS format"""
    try:
        tag = pos_tag([word])[0][1][0].upper()
        tag_map = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_map.get(tag, wordnet.NOUN)
    except Exception:
        return wordnet.NOUN

def normalize_word(word: str) -> str:
    word = word.lower()
    word = re.sub(r"'s$", "", word)      # Remove possessive
    word = re.sub(r"[^\w\s]", "", word)  # Remove punctuation
    if lemmatizer:
        pos = get_wordnet_pos(word)
        word = lemmatizer.lemmatize(word, pos=pos)
    return word

def normalize_list(words):
    if not lemmatizer:
        return [normalize_word(word) for word in words]

    tagged = pos_tag(words)
    valid_tags = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    filtered_words = [word for word, tag in tagged if tag in valid_tags]
    return [normalize_word(word) for word in filtered_words]