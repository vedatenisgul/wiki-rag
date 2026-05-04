# WikiRAG — What Can Be Done

## Infrastructure
ChromaDB can be swapped with **Pinecone** or **Weaviate** for a managed, horizontally scalable vector store. SQLite databases can be migrated to **PostgreSQL** for reliability and backups. The entire stack can be containerized with **Docker** and deployed on **AWS ECS/Fargate** or **Google Cloud Run**.

## LLM & Embeddings
Ollama can be replaced with **vLLM** or **HuggingFace TGI** to handle concurrent users in production. The model can be upgraded to **Llama 3 70B** or **Mistral 7B** for significantly better answers. Embeddings can be swapped to **OpenAI `text-embedding-3-small`** for improved multilingual quality.

## Retrieval
A **cross-encoder reranker** can be added on top of vector search results for better ranking. **Hybrid search** (BM25 + dense vectors via OpenSearch or Weaviate) can be introduced to handle keyword-heavy queries more reliably. Retrieval quality can be tracked objectively using **RAGAS** on a small evaluation set.

## Reliability
Full tracing can be added with **LangSmith** or **Langfuse** to debug retrieval and generation separately. The response cache can be moved to **Redis** to support multiple replicas sharing one hot cache. Ingestion can be offloaded to background workers with **Celery**, and rate limiting can be added before exposing the system to multiple users.