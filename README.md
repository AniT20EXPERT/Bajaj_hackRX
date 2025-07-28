
# 🧠 HackRx RAG API — Document Q&A using LLaMA-3 + Pinecone

This project is a production-ready **Retrieval-Augmented Generation (RAG) API** that takes in a user-uploaded PDF and a list of natural language questions, performs contextual retrieval over the document, and returns accurate answers powered by **Groq’s ultra-fast LLaMA-3.1-70B Versatile model**.

---

## 🚀 Features

- 📄 Accepts any publicly accessible PDF document
- ❓ Accepts multiple user questions in one request
- 🧩 Uses **sentence-aware chunking** (NLTK) to split documents at sentence boundaries
- 📌 Uses **hash-based chunk IDs** to prevent duplicate embeddings of the same document
- 💾 Stores and retrieves embeddings via **Pinecone vector database**
- ⚡ Embeddings are generated **locally using Huggingface Sentence-Transformers**
- 🤖 Question answering powered by **LLaMA-3.1-70B Versatile via Groq API**

---

## 🛠️ Tech Stack

| Layer                  | Tool/Service                             |
|------------------------|-------------------------------------------|
| ⚙ Backend              | FastAPI                                   |
| 📄 Embeddings          | `sentence-transformers` (local, Huggingface) |
| 🔍 Vector Store        | [Pinecone](https://www.pinecone.io/)      |
| 🧠 LLM (Answering)     | [`LLaMA-3.1-70B Versatile`](https://console.groq.com/) via Groq API |
| 🧠 Sentence Chunking   | [NLTK](https://www.nltk.org/) for sentence splitting |
| 🔐 Auth                | Bearer token-based authorization          |
| 🔄 Deduplication       | SHA256 hash of document + chunk IDs       |

---

## 📦 API Overview

### 🔹 `POST /hackrx/run`

#### Request Example:
```json
{
  "documents": "https://example.com/your-policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What are the exclusions under maternity benefits?"
  ]
}
````

#### Header:

```
Authorization: Bearer <your_api_key>
```

#### Response Example:

```json
{
  "answers": [
    "A 30-day grace period is provided for premium payments.",
    "Maternity benefits are covered after 24 months with sub-limits."
  ]
}
```

---

## 📁 Folder Structure

```
.
├── main1.py                  # FastAPI entrypoint
├── get_text_pdf.py         # PDF text extraction logic
├── get_embeddings.py       # Sentence splitting + local embedding generation
├── llm_answering.py        # Groq API answering using LLaMA-3
├── pinecone_db.py          # Pinecone upsert/query utilities
├── hash_text.py            # SHA256 hashing for deduplication
├── .env                    # Store your API keys here
```

---

## 🔐 Environment Variables

Create a `.env` file in your project root with the following content:

```env
MY_API_KEY_FOR_AUTH=your_local_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=hackrx-index
GROQ_API_KEY=your_groq_api_key
```

---

## 🧠 Deduplication Logic

Each uploaded document is hashed using `SHA256`. This ensures:

* ✅ You never embed the same document twice
* ✅ Pinecone stores each chunk under an ID like:

```
{doc_hash}-chunk-{i}
```

If the same PDF is uploaded again, embedding is **skipped** and only retrieval happens.

---

## 🧠 Embedding Model (Local)

This app uses [HuggingFace Sentence Transformers](https://www.sbert.net/) locally
Fast and suitable for short to mid-length documents.

---

## 📎 Sentence-Aware Chunking (NLTK)

Documents are first tokenized into sentences using NLTK's `sent_tokenize()`.

Chunks are then built using sentence boundaries (not blindly splitting by character length), improving semantic consistency.

**Overlap** between chunks is handled at sentence level for better context flow.


---

## 📌 Chunk Splitting and ID Logic

* Chunk size: \~700 characters
* Overlap: 1–2 sentences from the previous chunk
* Chunk ID: `{doc_hash}-chunk-{i}`

This allows:

* 🔄 Consistent re-chunking
* 🧼 Idempotent upserts into Pinecone

---

## 🔐 Authorization

All API endpoints are protected via **Bearer Token**.
Provide this header in every request:

```http
Authorization: Bearer your_local_api_key
```

---

## 🧪 Running Locally

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI server

```bash
uvicorn main1:app --reload
```

Then go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the API with Swagger UI.

---

## 🧼 Optional: Clear Pinecone Index

To clear all document vectors (dev-only):

```bash
curl -X POST http://localhost:8000//hackrx/clear
```

---

## 🧠 Model Used

Answering is powered by:

> **LLaMA-3.1-70B Versatile**
> served via [Groq API](https://console.groq.com/) — ultra low-latency inference with OpenAI-compatible API interface.

---

## 📄 License

MIT License

---

## 🙌 Acknowledgements

* [HuggingFace](https://huggingface.co/sentence-transformers) for MiniLM sentence encoders
* [Groq](https://groq.com/) for blazing-fast LLM inference
* [Pinecone](https://pinecone.io/) for scalable vector storage
* [NLTK](https://www.nltk.org/) for sentence-aware chunking
* [FastAPI](https://fastapi.tiangolo.com/) for building production APIs fast

---

