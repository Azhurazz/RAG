import hashlib
import re
from typing import List

from langchain_core.documents import Document

# \w alone splits Thai words at vowel/tone marks (category Mn), so include the Thai block.
_WORD_RE = re.compile(r"[\w\u0E00-\u0E7F]+", re.UNICODE)
_THAI_RE = re.compile(r"[฀-๿]")


def doc_key(d: Document) -> str:
    return d.metadata.get("chunk_id") or hashlib.sha1(d.page_content.encode()).hexdigest()


def tokenize(text: str) -> List[str]:
    """
    Tokenizer for BM25.

    - Strips punctuation, so "attention?" matches "attention".
    - Thai has no spaces between words, so whitespace/regex splitting turns a
      whole Thai sentence into one token. If pythainlp is installed, Thai text
      is word-segmented with it; otherwise we fall back to the regex.
    """
    text = text.lower()
    if _THAI_RE.search(text):
        try:
            from pythainlp.tokenize import word_tokenize
            return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False)
                    if _WORD_RE.search(t)]
        except ImportError:
            pass
    return _WORD_RE.findall(text)
