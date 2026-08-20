# CharacterTextSplitter vs RecursiveCharacterTextSplitter 

from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

sample_text = """
LangChain is a framework for building applications powered by large language models.
It provides chains, prompts, memory, and agents to simplify development.
Document loaders help bring external data into LangChain's standard format.
Text splitters break large documents into smaller, manageable chunks.
This is essential for RAG pipelines, where chunks are embedded and stored in a vector database for retrieval.
""" 

# --- 1. CharacterTextSplitter - splits by a fixed separator + character count ---
char_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20
) 

char_chunks = char_splitter.split_text(sample_text)
print(f"1. CharacterTextSplitter ({len(char_chunks)} chunks):")
for i,chunk in enumerate(char_chunks):
    print(f"Chunk {i+1}: {chunk}\n") 

# --- 2. RecursiveCharacterTextSplitter - smarter, tries to preserve structure ---
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
) 
recursive_chunks = recursive_splitter.split_text(sample_text)
print(f"\n2. RecursiveCharacterTextSplitter ({len(recursive_chunks)} chunks):")
for i,chunk in enumerate(recursive_chunks):
    print(f"chunk {i+1}: {chunk}\n")