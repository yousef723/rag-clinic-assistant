# Clinic FAQ Assistant (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a clinic's policies by retrieving relevant chunks from a document and using an LLM to answer.

## What it does

- Loads a clinic policy document
- Splits it into chunks
- Embeds chunks with Sentence Transformers
- Stores embeddings in Pinecone (vector database)
- On a user question: retrieves top-3 relevant chunks and asks GPT-4o-mini to answer using only those chunks
- Streamlit UI for chatting

## Tech Stack

- Language: Python 3.14
- Framework: LangChain (text splitting)
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector DB: Pinecone
- LLM: OpenAI GPT-4o-mini
- UI: Streamlit

## Project Structure

rag-clinic-assistant/
  app.py               # Streamlit UI + RAG pipeline
  rag.py               # Data ingestion (creates embeddings, uploads to Pinecone)
  clinic_policy.txt    # Source document (fictional clinic FAQ)
  requirements.txt     # Python dependencies
  .gitignore
  README.md

## How to Run

1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Create .env with OPENAI_API_KEY and PINECONE_API_KEY
4. Ingest the document: python rag.py
5. Run the UI: streamlit run app.py

## Design Choices

- Chunking: RecursiveCharacterTextSplitter with chunk_size=200, overlap=50
- Embedding model: all-MiniLM-L6-v2 (fast, free, good for English)
- Vector DB: Pinecone (managed cloud)
- Top-K: 3

## Example Questions

- What are your hours?
- Do you accept Aetna?
- How do I cancel an appointment?

## Known Limitations

- Retrieval quality depends on chunk size and top-K
- LLM can hallucinate if retrieved chunk is missing the answer
- No reranking yet

## Future Improvements

- Add reranking with a cross-encoder
- Add hybrid search
- Add RAGAS evaluation
- Support multiple documents
