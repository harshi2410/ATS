"""
NLP Preprocessing Module.
Handles text cleaning, tokenization, stop word removal, and lemmatization,
with special preservation for programming terms (C++, C#, Node.js, CI/CD, .NET).
"""

import re
import os
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Ensure local nltk_data in workspace is prioritized
workspace_nltk_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nltk_data"))
if os.path.exists(workspace_nltk_data):
    nltk.data.path.insert(0, workspace_nltk_data)

# Load stopwords with robust fallback
try:
    STOP_WORDS: Set[str] = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves"
    }

# Initialize Lemmatizer with fallback to PorterStemmer
_lemmatizer = None
try:
    _lemmatizer = WordNetLemmatizer()
    # Test lemmatizer to verify WordNet corpus availability
    _lemmatizer.lemmatize("developers", pos="n")
except Exception:
    _lemmatizer = None

_stemmer = PorterStemmer()

# Mapping to protect special technical acronyms and tokens before punctuation stripping
TECH_TOKEN_MAP = {
    "c++": "cplusplus",
    "c#": "csharp",
    ".net": "dotnet",
    "node.js": "nodejs",
    "react.js": "reactjs",
    "vue.js": "vuejs",
    "ci/cd": "cicd",
    "ci / cd": "cicd",
    "tcp/ip": "tcpip"
}
REVERSE_TECH_MAP = {v: k for k, v in TECH_TOKEN_MAP.items()}


def protect_tech_tokens(text: str) -> str:
    """Replaces tech tokens containing punctuation with alphanumeric placeholders."""
    lower_text = text.lower()
    for token, placeholder in TECH_TOKEN_MAP.items():
        # Match standalone token with boundary checks
        pattern = re.escape(token)
        lower_text = re.sub(rf"(?:\b|(?<=\s)){pattern}(?:\b|(?=\s)|$)", f" {placeholder} ", lower_text)
    return lower_text


def tokenize_text(text: str) -> List[str]:
    """
    Cleans and tokenizes text into lowercase tokens, handling technical abbreviations.
    """
    if not text:
        return []

    # Protect technical keywords like c++, c#, .net
    protected = protect_tech_tokens(text)

    # Normalize whitespace and strip URLs / emails to focus on domain content
    protected = re.sub(r"https?://\S+|www\.\S+", " ", protected)
    protected = re.sub(r"\S+@\S+", " ", protected)

    # Replace non-alphanumeric (except underscores and spaces)
    cleaned = re.sub(r"[^\w\s-]", " ", protected)

    # Tokenize on whitespace and hyphens
    tokens = re.findall(r"\b[a-zA-Z0-9_#+.-]+\b", cleaned)
    return [t.lower() for t in tokens if len(t) > 1]


def lemmatize_or_stem(token: str) -> str:
    """Applies WordNet lemmatization if available, else PorterStemmer."""
    # Don't stem or lemmatize protected tech placeholders
    if token in REVERSE_TECH_MAP:
        return token

    if _lemmatizer is not None:
        try:
            return _lemmatizer.lemmatize(token, pos="v") or _lemmatizer.lemmatize(token, pos="n")
        except Exception:
            pass
    return _stemmer.stem(token)


def preprocess_text(text: str, remove_stop_words: bool = True, lemmatize: bool = True) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Lowercasing & tech term protection
    2. Tokenization
    3. Stop word removal
    4. Lemmatization
    5. Clean string reassembly
    """
    tokens = tokenize_text(text)

    if remove_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    if lemmatize:
        tokens = [lemmatize_or_stem(t) for t in tokens]

    return " ".join(tokens)


def get_token_list(text: str) -> List[str]:
    """Returns preprocessed token list."""
    processed = preprocess_text(text, remove_stop_words=True, lemmatize=True)
    return processed.split()
