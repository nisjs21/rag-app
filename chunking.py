

def chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end]
        
        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return [text for text in chunks if text.strip()]