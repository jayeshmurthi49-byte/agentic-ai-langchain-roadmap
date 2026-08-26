# Complete RAG Pipeline: Load -> Split -> Embed/Store -> Retrieve -> Augment -> Generate

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv 

load_dotenv() 

# --- 1. LOAD ---
with open("rag_sample.txt", "w", encoding="utf-8") as f:
    f.write("""
LangChain is a framework for building applications powered by large language models.
It provides chains, prompts, memory, and agents to simplify development.
RAG (Retrieval Augmented Generation) combines retrieval with generation to answer 
questions using external data, reducing hallucinations and keeping answers current.
Vector stores like Chroma hold embeddings for fast similarity search.
Groq provides extremely fast LLM inference through an API, often used for free-tier 
development because of its generous rate limits and open-source model support.
""")
loader = TextLoader("rag_sample.txt", encoding="utf-8")
raw_docs = loader.load() 

# --- 2. SPLIT ---

splitter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=30)
chunks = splitter.split_documents(raw_docs)
print(f"1-2 Loaded and split on {len(chunks)} chunks")

# --- 3. EMBED + STORE --- 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks,embeddings) 
print("3. Embedded and stored in Chroma")  


# --- 4. RETRIEVE --- 
retriever = vector_store.as_retriever(search_kwargs={"k":2})


# --- 5 & 6. AUGMENT + GENERATE (via LCEL chain) ---
llm = ChatGroq(model="openai/gpt-oss-20b")
parser = StrOutputParser()


rag_prompt = PromptTemplate(
        template="""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't know based on the given context."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) 

rag_chain = (
    {"context":retriever | format_docs,"question":RunnablePassthrough()}
    | rag_prompt 
    |llm
    |parser
)

# --- Ask a question ---
question = "Why is Groq used for free-tier development?"
answer = rag_chain.invoke(question)

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")