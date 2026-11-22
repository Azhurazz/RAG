import hashlib
from langchain_core.documents import Document

def doc_key(d: Document) -> str:
    return d.metadata.get("chunk_id") or hashlib.sha1(d.page_content.encode()).hexdigest()
