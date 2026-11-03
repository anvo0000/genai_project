# AWS Bedrock RAG

Minimal reference scripts for building a RAG pipeline on **AWS Bedrock** instead of OpenAI/Ollama —
the Stage 3 "Platforms and Ecosystem" example for Bedrock, paired with FAISS for vector storage.

Both scripts use Amazon Titan models via `langchain_aws` and `boto3`:

| File | What it does |
|---|---|
| `rag.py` | RAG over a few hardcoded in-memory sentences — the smallest possible example of embed → store in FAISS → retrieve → answer. |
| `rag_pdf.py` | Same pipeline, but loads and chunks a real PDF (`files/MLA-C01.pdf`) instead of inline text. |

## Prerequisites

- An AWS account with Bedrock model access enabled for:
  - `amazon.titan-text-express-v1` (text generation)
  - `amazon.titan-embed-text-v1` (embeddings)
- AWS credentials configured locally (e.g. `aws configure`, or environment variables) with permission
  to call `bedrock-runtime` in `us-west-2`.

## Install

```bash
pip install langchain-aws langchain-community faiss-cpu boto3 pypdf
```

## Run

```bash
python rag.py
python rag_pdf.py
```

## Notes

- Region and model IDs are hardcoded at the top of each file (`BEDROCK_REGION`, `BEDROCK_TEXT_MODEL`,
  `BEDROCK_EMBED_MODEL`) — change them there if you want a different region or model.
- These are intentionally simple, single-file scripts (no app/UI), unlike the Stage 4 RAG apps in
  `4-advanced-architectures/` — the point here is isolating "how does Bedrock plug into LangChain",
  not building a full product.
