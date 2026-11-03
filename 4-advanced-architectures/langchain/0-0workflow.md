### Full Workflow:

```mermaid
graph LR
    RD[0-Raw Data] --> DL[1-Document Loader] --> TS[2-Text Splitters] --> TC[2-Text Chunks] 
    TC --> clean_text[3-Clean Text] --> EM[4-Embeddings] --> VS[5-Vector Store] --> RETRIEVER[6-Retriever]
    
    QUERY[Query]  --> RETRIEVER[Retriever]
    RETRIEVER --> VECTORSTORE[Vector Store] --> DOC[Relevant Documents]
    RETRIEVER --> KW[Keyword Search] --> DOC[Relevant Documents]
    RETRIEVER --> HB[Hybrid Search] --> DOC[Relevant Documents]
```
### Components Details:
#### Data Preparation:
Loading, preprocessing and structuring.
```mermaid
graph LR
    RD[Raw Data] --> DL[Document Loader] --> TS[Text Splitters] --> TC[Text Chunks] --> EM[Embeddings] --> VS[Vector Store]
```

#### Indexes:
Data structures that organize documents for efficient retrieval
```mermaid
graph LR
    IDX[Indexes] 
    IDX --> VSI[Vector Store Index]
    VSI --> FAISS[FAISS] 
    VSI --> PN[Pinecone]
    VSI --> CH[ChromaDB]
    
    IDX --> LI[List Index]
    IDX --> SQL[SQL Index]
    IDX --> GR[Graph Index]
```


#### Retrievers:
Finders of relevant documents based on query

```mermaid
graph LR
    QUERY[Query]  --> RETRIEVER[Retriever]
    RETRIEVER --> VECTORSTORE[Vector Store] --> DOC[Relevant Documents]
    RETRIEVER --> KW[Keyword Search] --> DOC[Relevant Documents]
    RETRIEVER --> HB[Hybrid Search] --> DOC[Relevant Documents]
```
