from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")
response = llm.invoke("What is LangChain? Explain in 2 lines.")
print(response.content)