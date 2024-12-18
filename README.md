# Secure RAG Challenge by Understand Tech, Competition Winners

Welcome to the official repository documenting the solutions of the top five participants in our recent data science competition focused on building a secure offline Retrieval-Augmented Generation (RAG) pipeline. Below, you'll find detailed descriptions of each winner's RAG pipeline, highlighting their technical approaches and final results.

## Winners

1. [Abdoulaye Sayouti Souleymane](#Abdoulaye)
2. [Ahmed Benmessaoud](#Ahmed)
3. [Param Thakkar team](#Param)
5. [Shravan Koninti](#Shravan)
6. [Dao Duong](#Dao)

---

## 1. Abdoulaye SAYOUTI SOULEYMANE

[Repository Link](https://github.com/Abdoulaye-Sayouti/Secure-Offline-RAG-System)
[AI Assistant Link](https://github.com/Abdoulaye-Sayouti/Secure-Offline-RAG-System)

### RAG Pipeline Overview

Abdoulaye's solution stands out for its innovative combination of dense embeddings with sparse retrieval methods, complemented by an effective ranking mechanism. This hybrid approach ensures high precision in retrieving relevant information while maintaining computational efficiency.

#### Components:

- **Preprocessing:**
  - **Document Handling:** Supports a wide range of file formats including `.txt`, `.md`, `.pdf`, `.docx`, `.py`, and Jupyter notebooks (`.ipynb`).
  - **Conversion:** Utilizes LibreOffice for converting Office documents to PDF, followed by Docling for parsing PDFs into Markdown to preserve formatting and structure.
  - **Text Splitting:** Implements LangChain's loaders and text splitters with content-aware strategies to divide documents into optimized chunks.

- **Encoding:**
  - **Dense Embeddings:** Uses Nomic Embed to generate high-quality semantic embeddings capturing contextual nuances.
  - **Sparse Encoding:** Employs BM25 with TF-IDF weighting for keyword-based retrieval, enhancing the system's ability to handle precise queries.

- **Storage & Retrieval:**
  - **Vector Storage:** Implements FAISS for efficient similarity searches on dense embeddings.
  - **Sparse Representation:** Maintains a separate TF-IDF index for sparse representations, facilitating quick keyword-based retrieval.

- **Retrieval Mechanism:**
  - **Hybrid Search:** Combines FAISS-based dense search with BM25-based keyword search to retrieve top K relevant documents.
  - **Re-ranking:** Applies the BGE Reranker model to refine and prioritize retrieved documents based on relevance.

- **Generation:**
  - **Language Model:** Integrates Llamafile using the Mistral-Nemo-Instruct model, optimized for performance and computational efficiency.
  - **Response Generation:** Generates coherent and contextually accurate responses by leveraging the re-ranked documents.

### Final Results

Abdolay's pipeline achieved the highest accuracy on the private leaderboard, demonstrating superior performance in semantic similarity metrics and efficient handling of diverse content types. The hybrid encoding and re-ranking mechanisms contributed significantly to its top-tier results.

---

## 2. Ahmed

[Repository Link](https://github.com/benx13/grogChalleng/tree/master)

### RAG Pipeline Overview

Ahmed's approach leverages a parallel retrieval system combining vector-based and graph-based methods, enhancing the depth and relevance of retrieved information. This dual strategy, paired with an advanced re-ranking mechanism, ensures high-quality response generation.

#### Components:

- **Indexing & Knowledge Base Construction:**
  - **Knowledge Graph:** Constructs a knowledge graph using Lightrag, enabling entity and relationship extraction.
  - **Vector Store:** Employs Melvis VectorDB for storing dense embeddings with a hybrid search mechanism combining BM25 and dense retrieval.

- **Retrieval Mechanism:**
  - **Parallel Retrieval:** 
    - **Vector Store Retrieval:** Combines BM25 with dense search techniques for comprehensive information retrieval.
    - **Graph Store Retrieval:** Uses a custom fuzzy search clustering retriever to navigate the knowledge graph for contextual data.
  - **Re-ranking:** Implements an offline re-ranker based on Flan-T5 to prioritize relevant documents.

- **Generation:**
  - **Language Model:** Integrates Qwen 7b-instruct via Ollama for generating coherent and contextually appropriate responses.
  - **Response Synthesis:** Combines re-ranked data to produce accurate and comprehensive answers.

### Final Results

Ahmed's innovative retrieval and re-ranking strategy earned him the second position. The integration of graph-based contextual retrieval alongside vector-based methods allowed for enriched information sourcing, resulting in precise and relevant responses.

---

## 3. Param

[Repository Link](https://github.com/username/student-repo)

### RAG Pipeline Overview

Param's solution emphasizes simplicity and cost-effectiveness without compromising essential functionalities. This balance makes it an excellent choice for deployments with limited computational resources.

#### Components:

- **Preprocessing:**
  - **Document Handling:** Capable of ingesting PDFs, text files, and Markdown documents using LangChain Community's document loaders.
  - **Text Splitting:** Utilizes RecursiveCharacterTextSplitter with a chunk size of 1024 characters and 100-character overlap to maintain context.

- **Encoding:**
  - **Multi-language Embeddings:** Leverages `nextfire/paraphrase-multilingual-minilm:l12-v2` embeddings, supporting over 29 languages to ensure broad applicability.
  - **Embedding Storage:** Uses OllamaEmbeddings integrated with FAISS for efficient storage and retrieval of dense vectors.

- **Storage & Retrieval:**
  - **Vector Store:** Implements FAISS for rapid similarity searches.
  - **Retriever Configuration:** Sets up a retriever with configurable parameters for top K results and similarity thresholds.

- **Generation:**
  - **Language Model:** Integrates ChatOllama with a quantized version of Qwen 0.5B, optimized for speed and low memory usage.
  - **Response Generation:** Achieves fast inference times (1-2 seconds) on standard laptops equipped with Nvidia RT3050 GPUs.

### Final Results

Param's pipeline secured the third position by delivering a straightforward yet effective solution. Its multilingual support and minimal resource consumption make it highly adaptable for various deployment scenarios, ensuring responsive and accurate responses.

---
## 4. Shravan

[Repository Link](https://github.com/shravankoninti/Secure_Offline_RAG_System)

### RAG Pipeline Overview

Shravan developed a robust RAG pipeline optimized for handling complex technical documentation. While the solution demonstrates high reliability and accuracy, it necessitates GPU resources to achieve optimal performance.

#### Components:

- **Text Processing:**
  - **EnhancedTextSplitter:** Custom splitter preserving technical terms and maintaining contextual continuity with a chunk size of 512 tokens and 50-token overlap.
  - **Preprocessing:** Handles question-answer pairs to maintain relational context within chunks.

- **Encoding:**
  - **Embeddings:** Uses `multi-qa-mpnet-base-dot-v1` from SentenceTransformer for generating dense embeddings, optimized for multi-query tasks.
  - **GPU Acceleration:** Leverages GPU resources to accelerate embedding generation and reduce processing time.

- **Storage & Retrieval:**
  - **Vector Store:** Implements FAISS for efficient similarity searches.
  - **Retriever Setup:** Configures FAISS-based retrievers with optimized search parameters for rapid document retrieval.

- **Generation:**
  - **Language Model:** Utilizes Qwen2.5-14B-Instruct with 4-bit quantization to balance performance and memory usage.
  - **Response Generation:** Structures prompts to include detailed instructions, temperature settings, and repetition penalties to enhance response quality.

### Final Results

Shravan's solution secured the fourth position by delivering a highly reliable and technically robust RAG pipeline. The meticulous text processing and GPU-optimized encoding ensure accurate handling of complex queries, although the dependency on GPU resources may limit scalability in resource-constrained environments.

---

## 5. Dao

[Repository Link](https://github.com/duongkstn/trustii-secure-rag-system)

### RAG Pipeline Overview

Dao's pipeline integrates a straightforward retrieval mechanism with effective contextual response generation. While functional and efficient, it ranks fifth due to its comparatively basic retrieval strategies.

#### Components:

- **Retrieval:**
  - **Embedding Model:** Uses `all-MiniLM-L6-v2` from Sentence Transformers for generating dense embeddings.
  - **Vector Store:** Implements FAISS for similarity searches, enabling quick retrieval of relevant documents.
  - **Retriever Configuration:** Sets top K results to 20, ensuring a broad context base for response generation.

- **Generation:**
  - **Language Model:** Integrates Qwen2.5 (7B) with multi-language capabilities to generate responses.
  - **Prompt Construction:** Combines retrieved question-answer pairs with user queries to form comprehensive prompts.
  - **Expert Answer Generation:** Instructs the LLM to act as a technical expert, providing detailed answers or indicating when information is unavailable.

- **Optimization:**
  - **Model Serving:** Utilizes the Ollama framework for simplified deployment and efficient model serving on local hardware.
  - **Resource Management:** Designed to operate with minimal CPU and GPU usage, leveraging Docker for consistent environment setup.

### Final Results

Dao's solution achieved the fifth position by delivering a reliable and efficient RAG pipeline. While it effectively handles retrieval and response generation, the absence of advanced encoding and re-ranking mechanisms limits its performance compared to the higher-ranked solutions.

---

## Conclusion

The competition showcased a range of innovative and technically sophisticated RAG pipeline solutions. **Abdoulaye** and **Ahmed** led with a hybrid encoding and re-ranking approach, followed by **Param** with a simple and cost-effective solution, while **Shravan** provided a robust GPU-optimized pipeline. Lastly, **Dao** offered a functional and efficient solution, rounding out the top five participants.

Each of these solutions contributes unique strengths to the field of *Open Source* Retrieval-Augmented Generation, offering valuable insights and methodologies for future developments.
