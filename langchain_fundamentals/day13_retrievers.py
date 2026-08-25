# Vector Store-based Retriever + Multi-Query Retriever

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="openai/gpt-oss-20b")

docs = [
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
    Document(page_content="Vector stores like Chroma hold embeddings for fast similarity search."),
    Document(page_content="Retrievers wrap vector stores into a standard interface for RAG chains."),
    Document(page_content="Multi-Query Retriever rewrites a question multiple ways to improve search recall."),
    Document(page_content="RAG combines retrieval with generation to answer questions using external data."),
]

vector_store = Chroma.from_documents(docs, embeddings)

# --- 1. Basic Vector Store-based Retriever ---
basic_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
query = "How does a chatbot find relevant information?"
basic_results = basic_retriever.invoke(query)

print("1. Vector Store-based Retriever:")
for i, doc in enumerate(basic_results):
    print(f"   {i+1}. {doc.page_content}")

# --- 2. Multi-Query Retriever - rewrites the query multiple ways ---
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    llm=llm
)
multi_results = multi_query_retriever.invoke(query)

print("\n2. Multi-Query Retriever:")
for i, doc in enumerate(multi_results):
    print(f"   {i+1}. {doc.page_content}") 


# --- 3. Wikipedia Retriever --- 
# NOTE: Skipped due to a known issue where the 'wikipedia' package fails 
# to get valid JSON from Wikipedia's API on this network/environment.
# Concept verified via LangChain docs — WikipediaRetriever queries Wikipedia 
# directly and returns article content as Documents, no vector store needed.
# wiki_retriever = WikipediaRetriever(top_k_results=1)
# wiki_results = wiki_retriever.invoke("LangChain software")
# from langchain_community.retrievers import WikipediaRetriever

# wiki_retriever = WikipediaRetriever(top_k_results=1)
# wiki_results = wiki_retriever.invoke("LangChain software")
# print("\n3. Wikipedia Retriever:")
# print(wiki_results[0].page_content[:300], "...")

mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":2,"fetch_k":4} 
)

mmr_results = mmr_retriever.invoke(query)
print("\n4. MMR Retriever:")
for i, doc in enumerate(mmr_results):
    print(f"   {i+1}. {doc.page_content}")  


# --- 5. Contextual Compression Retriever ---
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor


compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever(search_kwargs={"k":2})
) 

compressed_results = compression_retriever.invoke(query)
print("\n5. Contextual Compression Retriever:")
for i, doc in enumerate(compressed_results):
    print(f"   {i+1}. {doc.page_content}") 