from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.tools import tool 

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "space.pdf"
CHROMA_DIR  = str(BASE_DIR/"chroma_db") 

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if Path(CHROMA_DIR).exists():
        print("Loaded existing index from disk")
        return Chroma(persist_directory=CHROMA_DIR,embedding_function=embeddings) 

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"pdf not found {PDF_PATH}. put space.pdf next to rag_tool.py") 

    docs = PyPDFLoader(str(PDF_PATH)).load() 

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = splitter.split_documents(docs) 

    print(f"Built new index: {len(docs)} pages, {len(chunks)} chunks")
    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR) 


vector_store = get_vector_store() 
retriver = vector_store.as_retriever(search_kwargs={"k":3}) 


@tool
def search_document(query:str)->str:
    """Search the knowledge-base PDF documents and return the most relevant passages
    with their source file and page. Use this for questions about the documents' content."""
    results = retriver.invoke(query) 
    if not results:
        return "No relevant passages found in the document." 

    parts = [] 
    for r in results:
        source = Path(r.metadata.get("source","unknow")).name
        page = r.metadata.get("page") 
        page_label = page + 1 if isinstance(page,int) else "?"
        parts.append(f"[{source}, page{page_label}]\n{r.page_content}") 
    return "\n\n".join(parts) 

if __name__ == "__main__":
    print("Knowledge-base search test, type 'exit' to 'quit' ")
    while True:
        q = input("search:      ").strip()
        if q.lower() in ["exit","quit"]:
            break
        if q:
            print(search_document.invoke({"query":q}))
            print("-" * 50) 