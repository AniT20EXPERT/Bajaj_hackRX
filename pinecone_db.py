from pinecone import Pinecone, ServerlessSpec
from typing import List, Tuple
import time
import os
from get_embeddings import get_question_embedding
# Initialize Pinecone client (updated syntax for v6.0.0+)
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Index configuration for SentenceTransformer all-MiniLM-L6-v2
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors



# Get the index
def get_index():
    """Get the Pinecone index instance."""
    return pc.Index(INDEX_NAME)


# Updated function to store embeddings
def store_embeddings_in_pinecone(embeddings: List[Tuple[str, List[float]]], doc_id: str):
    """
    Store embeddings in Pinecone with improved error handling and batch processing.

    Args:
        embeddings: List of tuples containing (text, embedding_vector)
    """
    index = get_index()

    # Prepare vectors for upsert
    vectors_to_upsert = []

    for i, (text, vector) in enumerate(embeddings):
        vectors_to_upsert.append({
            "id": f"{doc_id}-chunk-{i}",
            "values": vector,
            "metadata": {"text": text}
        })

    # Batch upsert (more efficient than one-by-one)
    batch_size = 100  # Pinecone recommends batches of 100-1000

    try:
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]

            # Upsert the batch
            upsert_response = index.upsert(
                vectors=batch,
                namespace=""  # Use default namespace or specify your own
            )

            print(f"Upserted batch {i // batch_size + 1}: {upsert_response['upserted_count']} vectors")

    except Exception as e:
        print(f"Error during upsert: {e}")
        raise


def clear_pinecone_index():
    """
    Clear all records from a specific Pinecone index.

    Args:
        index_name (str): Name of the Pinecone index to clear
        api_key (str): Pinecone API key
        environment (str): Pinecone environment (optional for newer versions)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        index_name = INDEX_NAME
        api_key = os.getenv("PINECONE_API_KEY")

        # Initialize Pinecone (for newer versions)
        pc = Pinecone(api_key=api_key)

        # Connect to the index
        index = pc.Index(index_name)

        # Delete all vectors (this clears the entire index)
        index.delete(delete_all=True)

        return [f"{index_name}", True]

    except Exception as e:
        index_name = INDEX_NAME
        print(f"Error clearing index '{index_name}': {str(e)}")
        return [f"{index_name}", False]

def query_pinecone_for_context(question: str):

    question_vector = get_question_embedding(question)

    # print(f"question_vector: {question_vector}")

    index = get_index()
    results = index.query(
        vector=question_vector,
        top_k=5,
        include_metadata=True
    )

    # print(f"results: {results}")

    matches = results.get("matches", [])

    # print(f"matches: {matches}")

    if not matches:
        context = "No relevant context found."
        return context

    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
    # context = "\n**********\n".join([match["metadata"]["text"] for match in results["matches"]])

    return context
