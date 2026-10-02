
## How to Run

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` with `OPENAI_API_KEY` and `PINECONE_API_KEY`
4. Ingest the document: `python rag.py`
5. Run the UI: `streamlit run app.py`

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
- LLM can hallucinate if the retrieved chunk is missing the answer
- Reranking (cross-encoder) has been prototyped separately but is not yet wired into this app
- Single-document only; no multi-document support yet

## Future Improvements

- Integrate cross-encoder reranking into the live pipeline
- Add hybrid search (BM25 + vector)
- Add RAGAS evaluation
- Support multiple documents
