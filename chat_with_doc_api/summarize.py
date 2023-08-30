# setting device on GPU if available, else CPU
import os
import sys
from timeit import default_timer as timer
from typing import List

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS

from app_modules.init import app_init, get_device_types
from app_modules.llm_summarize_chain import SummarizeChain


def load_documents(source_pdfs_path, keep_page_info) -> List:
    loader = PyPDFDirectoryLoader(source_pdfs_path, silent_errors=True)
    documents = loader.load()
    if not keep_page_info:
        for doc in documents:
            if doc is not documents[0]:
                documents[0].page_content = (
                    documents[0].page_content + "\n" + doc.page_content
                )
        documents = [documents[0]]
    return documents


def split_chunks(documents: List, chunk_size, chunk_overlap) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


llm_loader = app_init(False)[0]

source_pdfs_path = (
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SOURCE_PDFS_PATH")
)
chunk_size = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CHUNCK_SIZE")
chunk_overlap = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("CHUNK_OVERLAP")
keep_page_info = (
    sys.argv[3] if len(sys.argv) > 3 else os.environ.get("KEEP_PAGE_INFO")
) == "true"

sources = load_documents(source_pdfs_path, keep_page_info)

print(f"Splitting {len(sources)} documents in to chunks ...")

chunks = split_chunks(
    sources, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
)

print(f"Summarizing {len(chunks)} chunks ...")
start = timer()

summarize_chain = SummarizeChain(llm_loader)
result = summarize_chain.call_chain(
    {"input_documents": chunks},
    None,
    None,
    True,
)

end = timer()
total_time = end - start

print("\n\n***Summary:")
print(result["output_text"])

print(f"Total time used: {total_time:.3f} s")
print(f"Number of tokens generated: {llm_loader.streamer.total_tokens}")
print(
    f"Average generation speed: {llm_loader.streamer.total_tokens / total_time:.3f} tokens/s"
)
