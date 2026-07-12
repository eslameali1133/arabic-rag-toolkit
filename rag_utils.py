"""
rag_utils.py — Reusable RAG (Retrieval-Augmented Generation) utilities.

Two main entry points:
    - ingest_documents(...)  -> builds/updates a Chroma vector store from a folder of files
    - query_documents(...)   -> answers a query using retrieval + an LLM

Designed to be dropped into any project: just import and call.
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()


# ---------------------------------------------------------------------------
# Shared / cached resources
# ---------------------------------------------------------------------------
# Embedding models are expensive to load — cache by model name so repeated
# calls to ingest_documents()/query_documents() in the same process (or
# notebook) don't reload the model from disk every time.

@lru_cache(maxsize=4)
def _get_embedding(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=4)
def _get_llm(model_name: str, temperature: float) -> ChatGroq:
    return ChatGroq(model=model_name, temperature=temperature)


# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------
def ingest_documents(
    docs_dir_path: str,
    vector_db_path: str,
    collection_name: str,
    glob_pattern: str = "./*.pdf",
    embedding_model_name: str = "intfloat/multilingual-e5-large",
    chunk_size: int = 2000,
    chunk_overlap: int = 500,
    languages: Optional[list] = None,
    ocr_strategy: str = "ocr_only",
) -> Chroma:
    """
    Load documents from a folder, split them into chunks, embed them, and
    persist them into a Chroma vector store.

    Parameters
    ----------
    docs_dir_path : str
        Folder containing the source documents (pdf, docx, txt, etc.).
    vector_db_path : str
        Folder where the Chroma vector store will be persisted.
    collection_name : str
        Name of the Chroma collection (namespace within the store).
    glob_pattern : str
        Glob pattern for selecting files, e.g. "./*.pdf" or "./**/*.docx".
    embedding_model_name : str
        HuggingFace embedding model. Default supports Arabic + English.
    chunk_size : int
        Target chunk size in characters.
    chunk_overlap : int
        Overlap between consecutive chunks, in characters.
    languages : list[str] | None
        Languages for OCR/Unstructured (e.g. ["ara", "eng"]).
        Defaults to ["ara", "eng"] if not provided.
    ocr_strategy : str
        Unstructured loader strategy. "ocr_only" is safest for PDFs with
        legacy/broken Arabic font encodings (garbled numbers/letters).
        Use "fast" for clean, purely-native-text PDFs to save time.

    Returns
    -------
    Chroma
        The persisted vector store (already populated with the new chunks).
    """
    if languages is None:
        languages = ["ara", "eng"]

    loader = DirectoryLoader(
        path=docs_dir_path,
        glob=glob_pattern,
        loader_cls=UnstructuredFileLoader,
        loader_kwargs={
            "languages": languages,
            "strategy": ocr_strategy,
        },
    )

    documents = loader.load()
    if not documents:
        raise ValueError(
            f"No documents found in '{docs_dir_path}' matching '{glob_pattern}'."
        )

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    text_chunks = text_splitter.split_documents(documents)

    embedding = _get_embedding(embedding_model_name)

    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=vector_db_path,
    )

    print(
        f"Ingested {len(documents)} document(s) -> "
        f"{len(text_chunks)} chunk(s) into collection '{collection_name}'."
    )
    return vector_store


# ---------------------------------------------------------------------------
# 2. Retrieval / Q&A
# ---------------------------------------------------------------------------
def query_documents(
    query: str,
    vector_db_path: str,
    collection_name: str,
    embedding_model_name: str = "intfloat/multilingual-e5-large",
    llm_model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    k: int = 4,
    return_sources: bool = True,
) -> dict:
    """
    Answer a query using retrieval-augmented generation against an
    already-ingested Chroma vector store.

    Parameters
    ----------
    query : str
        The user's question.
    vector_db_path : str
        Folder where the Chroma vector store is persisted.
    collection_name : str
        Name of the Chroma collection to query.
    embedding_model_name : str
        Must match the model used during ingestion.
    llm_model_name : str
        Groq model name (e.g. "llama-3.1-8b-instant", "llama-3.3-70b-versatile").
    temperature : float
        LLM sampling temperature.
    k : int
        Number of chunks to retrieve.
    return_sources : bool
        Whether to include source document metadata in the result.

    Returns
    -------
    dict
        {
            "answer": str,
            "sources": list[dict] | None   # metadata for each retrieved chunk
        }
    """
    embedding = _get_embedding(embedding_model_name)
    llm = _get_llm(llm_model_name, temperature)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=vector_db_path,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=return_sources,
    )

    response = qa_chain.invoke({"query": query})

    result = {"answer": response["result"]}
    if return_sources:
        result["sources"] = [
            doc.metadata for doc in response.get("source_documents", [])
        ]
    else:
        result["sources"] = None

    return result