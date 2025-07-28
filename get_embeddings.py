from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
import nltk
import os
from nltk.tokenize import sent_tokenize



# def split_text(text: str):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
#     return splitter.split_text(text)





def split_text(text: str) -> List[str]:
    """
    Use NLTK's punkt tokenizer for better sentence detection.
    Requires: pip install nltk
    """
    import nltk
    from nltk.tokenize import sent_tokenize

    max_chunk_size = 700

    # Download punkt tokenizer if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks





# Load the model once (not inside the function for efficiency)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings_batch(chunks: List[str]) -> List[Tuple[str, List[float]]]:
    """
    Batch version - processes all chunks at once for maximum efficiency.
    """
    if not chunks:
        return []

    # Process all chunks in one call (very efficient)
    embeddings_array = embedding_model.encode(
        chunks,
        batch_size=32,  # Process 32 chunks at a time
        show_progress_bar=True,  # Show progress for large batches
        convert_to_tensor=False  # Return as numpy arrays
    )

    # Return
    return [(chunk, embedding.tolist()) for chunk, embedding in zip(chunks, embeddings_array)]


def get_question_embedding(question: str) -> List[float]:
    return embedding_model.encode(question).tolist()