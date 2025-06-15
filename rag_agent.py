import os
import tiktoken
import sympy as sp
from dotenv import load_dotenv
from typing import List
from PyPDF2 import PdfReader

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

PDF_FOLDER = "data/pdfs"
INDEX_FOLDER = "outputs/faiss_index"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b:free"
MAX_RETRIEVAL_TOKENS = 2000
FAISS_INDEX = None

tokenizer = tiktoken.encoding_for_model("gpt-4")
def count_tokens(text: str) -> int:
    """
    Returns the number of tokens that "text" would consume for a model
    """
    return(len(tokenizer.encode(text)))

def load_pdfs_as_documents(pdf_folder: str) -> List[Document]:
    """
    Loads each PDF file (page by page) into a list of LangChain Documents.
    """
    docs = []
    for fname in os.listdir(pdf_folder):
        if not fname.lower().endswith(".pdf"):
            continue            # Skip non-pdf files
        
        filepath = os.path.join(pdf_folder, fname)
        reader = PdfReader(filepath)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue        # Skip empty pages or pages with no extractable text

            metadata = {"source": fname, "page": page_num}
            docs.append(Document(page_content=text, metadata=metadata))
    return docs

def split_documents(docs: List[Document]) -> List[Document]:
    """
    Breaks each PDF-page Document into ~500-character chunks, skipping any chunk under 50 characters
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    all_chunks = []
    for doc in docs:
        pieces = text_splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(pieces):
            trimmed = chunk_text.strip()
            if len(trimmed) < 50:
                continue        # Skip very small/noisy chunks

            new_metadata = doc.metadata.copy()
            new_metadata.update({"chunk_id": i})
            all_chunks.append(Document(page_content=chunk_text, metadata=new_metadata))
    return all_chunks

def build_faiss_index(index_path: str, embeddings_model_name: str) -> FAISS:
    """
    Build or Load a FAISS index. If the folder at index_path exists, load from disk.
    Otherwise:
    1. Load all PDFs, split and produce chunks
    2. Embed via HuggingFaceEmbeddings.
    3. Save the FAISS index at index_path
    
    Returns: FAISS vectorstore object.
    """
    embeddings = HuggingFaceEmbeddings(model=embeddings_model_name)

    if os.path.exists(index_path):
        print("Loading existing FAISS index from disk...")
        vectorstore = FAISS.load_local(
            folder_path=index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        # Load raw PDFs into Documents (page level)
        print("Loading PDF files...")
        docs = load_pdfs_as_documents(PDF_FOLDER)
        print(f" -> Loaded {len(docs)} page-level documents from PDFs.")

        # Split into smaller chunks (coherant, ~500 chars each)
        print("Splitting documents into chunks...")
        chunks = split_documents(docs)
        print(f" -> Created {len(chunks)} total chunks (with ~500 chars each).")
        
        # Build FAISS index from scratch
        print("Creating new FAISS index from scratch...")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        vectorstore.save_local(folder_path=index_path)
    return vectorstore

def extract_content(llm_output) -> str:
    """
     Given whatever ChatOpenAI(prompt) returned, return a plain string.
    - If it has a `.content` attribute, return that.
    - If it's a dict with OpenAI-style choices, extract choices[0].message.content.
    - Otherwise, assume it's already a string and return it.
    """
    if hasattr(llm_output, "content"):
        return llm_output.content
    if isinstance(llm_output, dict):
        try:
            return llm_output["choices"][0]["message"]["content"]
        except Exception:
            return str(llm_output)
    return llm_output    


def rag_retrieval_tool(query: str, k: int=3) -> str:
    """
    Retrieve the top-k chunks from FAISS_INDEX (global).
    Prefix each chunk with [source | page | chunk_id].
    If combined token count > MAX_RETRIEVAL_TOKENS, drop the last (least-relevant) chunk(s) until token count ≤ MAX_RETRIEVAL_TOKENS.
    
    Returns: a single string of concatenated chunks (with citation prefixes).
    """
    global FAISS_INDEX
    docs = FAISS_INDEX.similarity_search(query, k=k)        # returns List[langchain.schema.Document]

    chunk_texts = []
    for doc in docs:
        source = doc.metadata.get("source", "UnknownFile")
        page = doc.metadata.get("page", "UnknownPage")
        chunk_id = doc.metadata.get("chunk_id", "UnknownChunk")
        text = doc.page_content

        prefix = f"[{source} | page {page} | chunk {chunk_id}]\n"
        chunk_texts.append(prefix + text)

    def joined_text(texts: List[str]) -> str:
        return "\n\n---\n\n".join(texts)
    
    context_parts = chunk_texts.copy()
    joined = joined_text(context_parts)
    while count_tokens(joined) > MAX_RETRIEVAL_TOKENS and len(context_parts) > 1:
        context_parts.pop()
        joined = joined_text(context_parts)

    # Join with a visible separator so the LLM can see chunk boundaries.
    return joined

def rag_summary_tool(query: str, k: int=3) -> str:
    """
    Retrieve up to k chunks (token-trimmed) via rag_retrieval_tool(...)
    Prompt the LLM: "Summarize these chunks in ≤100 words, citing with [filename | page | chunk_id]."
    Returns that summary string.
    """
    retrieved_text = rag_retrieval_tool(query, k=k)
    prompt = (
        "Here are the top retrieved chunks from the PDF documents:\n\n"
        f"{retrieved_text}\n\n"
        "Please summarize the above text in 100 words or fewer, while keeping the user question in context. "
        f"The user question: {query}\n\n"
        "For each fact, explicitly cite it using the format "
        "[filename | page | chunk_id]."
    )

    llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0.5)
    try:
        summary = extract_content(llm.invoke(prompt))
        return summary.strip()
    except Exception as e:
        return f"Error in summarization: {e}"
    
def rag_answer_tool(query: str, k: int=3) -> str:
    """
    Call rag_summary_tool(query, k) to get a 100-word, citation-heavy summary.
    Build a custom prompt:
    “Using only the short summary below, which already cites all facts, answer the original question.”
    Returns the LLM's final answer (with citations).
    """
    summary = rag_summary_tool(query, k=k)
    prompt = (
        "Below is a short, 100-word summary of the most relevant PDF content, "
        "with every fact cited as [filename | page | chunk_id]:\n\n"
        f"{summary}\n\n"
        f"Original question: \"{query}\"\n\n"
        "Using *only* the information in the summary above, "
        "answer the original question. Provide your answer clearly, "
        "and include citations in brackets exactly as they appear. "
        "If the summary does not contain enough information, say \"Insufficient information in documents.\""
    )

    llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0.5)

    try:
        final_answer = extract_content(llm.invoke(prompt))
        return final_answer.strip()
    except Exception as e:
        return f"Error generating final answer: {e}"
    
def calculator_tool(expression: str) -> str:
    """
    Basic calculator: evaluates a Sympy expression and returns the result as a string.
    If parsing fails, returns an error message.
    """
    try:
        result = sp.sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"
    
def wiki_search_tool(query: str) -> str:
    """
    Simple wrapper around WikipediaAPIWrapper.
    Given a query string, returns the Wikipedia summary (first section).
    """
    wiki = WikipediaAPIWrapper()
    try:
        result = wiki.run(query)
        return result
    except Exception as e:
        return f"Error fetching from Wikipedia: {e}"

def main():
    print("Lifespan startup: Build or Load FAISS index...")
    # Build or Load FAISS index
    global FAISS_INDEX
    FAISS_INDEX = build_faiss_index(INDEX_FOLDER, EMBEDDINGS_MODEL_NAME)

if __name__ == "__main__":
    main()