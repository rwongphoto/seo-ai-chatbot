# SEO AI Chatbot

History-aware retrieval-augmented chatbot that answers SEO and marketing questions from a Qdrant-indexed knowledge base of your own content.

## What it does

- **Retrieves** the most relevant passages from a Qdrant Cloud collection (`OpenAIEmbeddings` + Langchain retriever).
- **Rewrites** follow-up questions into standalone search queries using chat history (Langchain `create_history_aware_retriever`) — so "what about for e-commerce?" still finds the right context.
- **Answers** with `gpt-3.5-turbo` grounded in the retrieved docs.
- **Streamlit chat UI** with persistent conversation state.

## Stack

- Streamlit (chat UI)
- Langchain (`langchain_openai`, `langchain_qdrant`, retrieval chains)
- Qdrant Cloud (vector store)
- OpenAI (`text-embedding-ada-002` via `OpenAIEmbeddings`, `gpt-3.5-turbo`)

## Setup

```bash
pip install -r requirements.txt
```

Set these environment variables:

- `OPENAI_API_KEY`
- `QDRANT_URL` — your Qdrant Cloud cluster URL
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION` — name of the pre-ingested collection

Then:

```bash
streamlit run the-seo-consultant.py
```

## Ingestion

This repo is the **chat interface only** — it expects the Qdrant collection to already exist. Use the companion [`qdrant-app`](https://github.com/rwongphoto/qdrant-app) repo to crawl a site and populate the collection.
