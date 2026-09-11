# Full Tool Calling cycle: Bind -> Call -> Execute -> Final Answer

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
import requests

load_dotenv()

# --- 1. Define a currency conversion tool ---
@tool
def get_conversion_rate(base_currency: str, target_currency: str) -> float:
    """Get the currency conversion rate between two currencies."""
    url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    response = requests.get(url)
    data = response.json()
    return data["rates"][target_currency]

@tool
def convert_currency(amount: float, rate: float) -> float:
    """Convert an amount using a given exchange rate."""
    return amount * rate

llm = ChatGroq(model="openai/gpt-oss-20b")
llm_with_tools = llm.bind_tools([get_conversion_rate, convert_currency])

# --- 2. Start the conversation ---
messages = [HumanMessage(content="What is 100 USD in INR?")]

# --- 3. First call - model decides to call a tool ---
ai_response = llm_with_tools.invoke(messages)
messages.append(ai_response)
print("Tool calls requested:", ai_response.tool_calls)

# --- 4. Execute each tool call and feed results back ---
available_tools = {"get_conversion_rate": get_conversion_rate, "convert_currency": convert_currency}

for tool_call in ai_response.tool_calls:
    tool_name = tool_call["name"]
    selected_tool = available_tools[tool_name]
    result = selected_tool.invoke(tool_call["args"])
    print(f"Executed {tool_name} -> {result}")
    
    # Feed the tool's result back into the conversation
    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

# --- 5. Final call - model uses tool results to generate a real answer ---
final_response = llm_with_tools.invoke(messages)
print("\nFinal Answer:", final_response.content)