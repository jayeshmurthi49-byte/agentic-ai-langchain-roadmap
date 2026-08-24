# Vector Store with Chroma - Create, Add, Similarity Search, Update, Delete

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- CREATE: initial documents with IDs ---
docs = [
    Document(page_content="LangChain is a framework for building LLM-powered applications.", metadata={"topic": "langchain"}),
    Document(page_content="Vector stores hold embeddings for fast similarity search.", metadata={"topic": "vector_store"}),
    Document(page_content="Chroma is a lightweight, open-source vector database for local development.", metadata={"topic": "chroma"}),
    Document(page_content="RAG combines retrieval with generation to answer questions using external data.", metadata={"topic": "rag"}),
]

doc_ids = ["doc1","doc2","doc3","doc4"] 

vectore_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    ids=doc_ids,
    persist_directory="./chroma_db"   # saves to disk automatically

)

print("1. CREATE - ADDED",len(docs),"documents\n") 

# --- READ: similarity search ---
query = "What is used to store embedding?"
results = vectore_store.similarity_search("What holds embedding?",k=2)
print(f"2. READ - Query: {query}") 
for i,results in enumerate(results):
    print(f"{i+1}. {results.page_content} (topic : {results.metadata["topic"]})")
print() 

# --- UPDATE: replace doc2's content with new content, same ID ---
updated_doc = Document(
    page_content="Vector stores like Chroma and FAISS hold embeddings and support fast similarity search.",
    metadata={"topic": "vector_store"}
)

vectore_store.update_document(document_id="doc2",document=updated_doc)
print("3. UPDATE - doc2 updated")
updated_results = vectore_store.similarity_search("What holds embedding?",k=1)
print("   New top match:", updated_results[0].page_content, "\n")


# --- DELETE: remove doc4 ---
vectore_store.delete(ids=["doc4"]) 
print("4. Delete - doc4 removed") 

remaining = vectore_store.get()
print("   Remaining document IDs:", remaining["ids"])