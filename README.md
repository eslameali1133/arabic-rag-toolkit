# RAG Utils — Bilingual (Arabic/English) Retrieval-Augmented Generation

A reusable, drop-in Python module for building RAG (Retrieval-Augmented Generation) pipelines with **LangChain**, **Chroma**, **Hugging Face embeddings**, and **Groq**. Built to handle bilingual Arabic/English documents correctly — including legacy PDFs with broken Arabic font encodings.

Two functions, one import:

```python
from rag_utils import ingest_documents, query_documents
```

## ✨ Features

- 📥 **`ingest_documents()`** — load a folder of documents, split into chunks, embed, and persist to a Chroma vector store
- 🔍 **`query_documents()`** — retrieve relevant chunks and get a grounded answer from an LLM
- 🌐 **Bilingual by default** — Arabic + English OCR (`languages=["ara", "eng"]`) and a multilingual embedding model (`intfloat/multilingual-e5-large`)
- 🛡️ **OCR-first PDF handling** — uses `strategy="ocr_only"` by default to avoid the garbled/reversed Arabic text that comes from legacy PDFs with broken font encodings (common in old Word-exported Arabic documents)
- 🚫 **Anti-hallucination prompt** — answers are grounded strictly in retrieved context; the model is instructed to say so explicitly rather than invent an answer when the document doesn't contain it
- ⚡ **Cached model loading** — embedding and LLM clients are cached (`lru_cache`) so repeated calls in the same process don't reload models
- 🔌 **Framework-agnostic** — works from a script, notebook, or any other Python project; no dependency on Streamlit or any specific app

## 📁 Files

```
rag_utils.py           # the two reusable functions
example_usage.py       # minimal example: ingest then query
smoke_test.py          # quick manual end-to-end check
test_rag_utils.py      # pytest suite
pytest.ini             # registers the `integration` test marker
```

## 🛠️ Requirements

- Python 3.11 or 3.12 (avoid 3.13+/3.14 — some dependencies like `nltk` and `unstructured` aren't fully compatible yet)
- [Poppler](https://poppler.freedesktop.org/) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed system-wide (with the Arabic language pack)
- A [Groq API key](https://console.groq.com/)

### System dependencies (macOS / Homebrew)

```bash
brew install poppler
brew install tesseract
brew install tesseract-lang   # adds Arabic + other language packs
```

Verify:
```bash
pdftoppm -v
tesseract --list-langs   # should include "ara"
```

## 🚀 Setup

### 1. Create a virtual environment (Python 3.12 recommended)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install langchain langchain-community langchain-classic langchain-chroma \
            langchain-huggingface langchain-groq langchain-text-splitters \
            "unstructured[pdf]" pytesseract pdf2image nltk python-dotenv
pip install pytest --break-system-packages   # optional, for running tests
```

### 3. Download the NLTK tokenizer data (one-time)

```python
import nltk
nltk.download("punkt_tab")
```

### 4. Add your Groq API key

Create a `.env` file in your project root:

```
GROQ_API_KEY=your_actual_key_here
```

`rag_utils.py` loads this automatically via `python-dotenv` — never hardcode the key in your notebook or script.

## 📖 Usage

### Ingest documents

```python
from rag_utils import ingest_documents

ingest_documents(
    docs_dir_path="/path/to/your/Docs_dir",
    vector_db_path="/path/to/your/vector_db",
    collection_name="document_collection",
    # optional overrides — shown with their defaults:
    glob_pattern="./*.pdf",
    embedding_model_name="intfloat/multilingual-e5-large",
    chunk_size=1000,
    chunk_overlap=150,
    languages=["ara", "eng"],
    ocr_strategy="ocr_only",   # use "fast" for clean, native-text PDFs
)
```

### Query the vector store

```python
from rag_utils import query_documents

response = query_documents(
    query="ما هي محاسن استخدام الحاسب الآلي؟",
    vector_db_path="/path/to/your/vector_db",
    collection_name="document_collection",
    # optional overrides — shown with their defaults:
    embedding_model_name="intfloat/multilingual-e5-large",
    llm_model_name="llama-3.1-8b-instant",
    temperature=0.0,
    k=4,
    return_sources=True,
)

print(response["answer"])
for source in response["sources"]:
    print(source)
```

Response shape:

```python
{
    "answer": "...",
    "sources": [{"source": "/path/to/file.pdf"}, ...]  # or None if return_sources=False
}
```

## 🧪 Testing

### Quick manual smoke test

```bash
python smoke_test.py
```

Creates a throwaway bilingual text file in a temp folder, ingests it, asks one English and one Arabic question, and verifies the answers — no dependency on your real documents or vector store.

### Full pytest suite

```bash
pytest test_rag_utils.py -v
```

All tests call the real embedding model and Groq API (marked `@pytest.mark.integration`), so `GROQ_API_KEY` must be set. Each test runs in an isolated temp directory via pytest's `tmp_path` fixture.

## ⚠️ Known Issues / Notes

- **Arabic text in the terminal may look reversed/scrambled even when correct.** Most terminals don't fully implement bidi rendering for mixed RTL/LTR text (numbered lists, markdown bold, punctuation). This is a display-only issue — check output in a notebook or GUI viewer to confirm the actual string is correct.
- **Legacy Arabic PDFs** (especially ones exported from old Word documents) can have broken font `ToUnicode` mappings, causing scrambled numbers or disconnected letters if read via native text extraction. `ocr_only` strategy (the default here) avoids this by rendering pages to images and running OCR instead.
- **PDF export of Arabic chat/RAG output** using default fonts (e.g. ReportLab's Helvetica) won't render Arabic — use a font like Amiri plus `arabic-reshaper` and `python-bidi` if you need Arabic PDF output elsewhere in your project.

## 🗺️ Roadmap

- [ ] Auto-detect whether a PDF needs OCR vs. fast native extraction (skip OCR when not needed, for speed)
- [ ] Support for incremental ingestion (add new documents without re-embedding existing ones)
- [ ] Configurable retrieval strategies (MMR, similarity score threshold)
- [ ] Optional reranking step for improved retrieval precision

## 📄 License

MIT

## 👤 Author

**Eslam Ali** —  AI Engineer & Mobile Tech Lead 
