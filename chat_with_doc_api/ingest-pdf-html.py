# setting device on GPU if available, else CPU
import os
from timeit import default_timer as timer
from typing import List

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS

from app_modules.init import *


def load_documents(source_path) -> List:
    loader = PyPDFDirectoryLoader(source_path, silent_errors=True)
    documents = loader.load()

    loader = DirectoryLoader(
        source_path, glob="**/*.html", silent_errors=True, show_progress=True
    )
    documents.extend(loader.load())
    return documents


def split_chunks(documents: List, chunk_size, chunk_overlap) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def generate_index(
    chunks: List, embeddings: HuggingFaceInstructEmbeddings
) -> VectorStore:
    if using_faiss:
        faiss_instructor_embeddings = FAISS.from_documents(
            documents=chunks, embedding=embeddings
        )

        faiss_instructor_embeddings.save_local(index_path)
        return faiss_instructor_embeddings
    else:
        chromadb_instructor_embeddings = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=index_path
        )

        chromadb_instructor_embeddings.persist()
        return chromadb_instructor_embeddings


# Constants
device_type, hf_pipeline_device_type = get_device_types()
hf_embeddings_model_name = (
    os.environ.get("HF_EMBEDDINGS_MODEL_NAME") or "hkunlp/instructor-xl"
)
index_path = os.environ.get("FAISS_INDEX_PATH") or os.environ.get("CHROMADB_INDEX_PATH")
using_faiss = os.environ.get("FAISS_INDEX_PATH") is not None
source_path = os.environ.get("SOURCE_PATH")
chunk_size = os.environ.get("CHUNCK_SIZE")
chunk_overlap = os.environ.get("CHUNK_OVERLAP")

start = timer()
embeddings = HuggingFaceInstructEmbeddings(
    model_name=hf_embeddings_model_name, model_kwargs={"device": device_type}
)
end = timer()

print(f"Completed in {end - start:.3f}s")

start = timer()

if not os.path.isdir(index_path):
    print(
        f"The index persist directory {index_path} is not present. Creating a new one."
    )
    os.mkdir(index_path)

    print(f"Loading PDF & HTML files from {source_path}")
    sources = load_documents(source_path)
    # print(sources[359])

    print(f"Splitting {len(sources)} HTML pages in to chunks ...")

    chunks = split_chunks(
        sources, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
    )
    print(chunks[3])
    print(f"Generating index for {len(chunks)} chunks ...")

    index = generate_index(chunks, embeddings)
else:
    print(f"The index persist directory {index_path} is present. Loading index ...")
    index = (
        FAISS.load_local(index_path, embeddings)
        if using_faiss
        else Chroma(embedding_function=embeddings, persist_directory=index_path)
    )
    query = "hi"
    print(f"Load relevant documents for standalone question: {query}")

    start2 = timer()
    docs = index.as_retriever().get_relevant_documents(query)
    end = timer()

    print(f"Completed in {end - start2:.3f}s")
    print(docs)

end = timer()

print(f"Completed in {end - start:.3f}s")
