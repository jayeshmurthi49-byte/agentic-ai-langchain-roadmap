# Exploring LangChain Models - Chat Model with parameters

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv() 

# Step 1: Create model with temperature parameter
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=100
)

# Step 2: Send a query
response = llm.invoke("Explain what a Chat Model is in LangChain, in 2 lines.")

# Step 3: Print result and metadata 
print("Response:",response.content)
print("\n Model used",response.response_metadata.get("model_name"))
print("Tokens used:",response.response_metadata.get("token_usage"))