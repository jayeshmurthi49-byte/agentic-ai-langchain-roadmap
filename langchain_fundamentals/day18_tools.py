# Custom Tools with @tool decorator + bind_tools()

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define custom tools using @tool decorator ---
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

# Inspect what the decorator generated automatically
print("Tool name:", add.name)
print("Tool description:", add.description)
print("Tool schema:", add.args)

# --- 2. Bind tools to the LLM ---
llm = ChatGroq(model="openai/gpt-oss-20b")
llm_with_tools = llm.bind_tools([add, multiply])

# --- 3. Ask something that should trigger a tool call ---
response = llm_with_tools.invoke("What is 7 plus 15?")

print("\nDid the model call a tool?", bool(response.tool_calls))
if response.tool_calls:
    print("Tool call details:", response.tool_calls)
    
    # --- 4. Actually execute the tool (your code does this, not the LLM) ---
    tool_call = response.tool_calls[0]
    if tool_call["name"] == "add":
        result = add.invoke(tool_call["args"])
        print("Tool execution result:", result)