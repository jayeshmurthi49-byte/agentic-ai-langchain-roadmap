from langchain_community.document_loaders import TextLoader,PyPDFLoader

# --- 1. TextLoader - create a simple text file first, then load it ---

with open("sample.txt","w",encoding="utf-8") as f:
     f.write("LangChain is a framework for building LLM-powered applications. "
                "It provides chains, prompts, memory, and agents to simplify development.")

text_loader = TextLoader("sample.txt",encoding="utf-8")
text_docs = text_loader.load() 

print("1. TextLoader output")
print("content",text_docs[0].page_content)
print("Metadata:",text_docs[0].metadata)

# --- 2. PyPDFLoader - loads a PDF (place any small PDF named sample.pdf in this folder) ---
try:
    pdf_loader = PyPDFLoader("sample.pdf")
    pdf_docs = pdf_loader.load()
    print(f"\n2. PyPDFLoader Output ({len(pdf_docs)} pages loaded):")
    print("First page content (first 200 chars):", pdf_docs[0].page_content[:200])
    print("First page metadata:", pdf_docs[0].metadata)
except FileNotFoundError:
    print("\n2. PyPDFLoader: place a small PDF named 'sample.pdf' in this folder to test this part.")