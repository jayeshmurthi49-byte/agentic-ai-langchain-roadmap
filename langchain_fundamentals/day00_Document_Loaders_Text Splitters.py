from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("revision_sample.txt", "w", encoding="utf-8") as f:
    f.write("LangChain helps you build LLM-powered apps. Document loaders bring in external data. "
            "Text splitters break it into chunks. This is the foundation of any RAG pipeline you'll build.")

loader = TextLoader("revision_sample.txt", encoding="utf-8")
docs = loader.load() 

splitter = RecursiveCharacterTextSplitter(chunk_size=50,chunk_overlap=10)
chunks = splitter.split_text(docs[0].page_content) 

print(f"Loaded document, split into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")