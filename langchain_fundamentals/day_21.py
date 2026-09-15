from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests

load_dotenv() 

@tool
def get_conversion_rate(base_currency:str,target_currency:str) -> float:
    """Get the currency conversion rate between two currencies."""
    url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    response = requests.get(url)
    data = response.json()
    return data["rates"][target_currency]

@tool
def convert_currency(amount:float,rate:float)->float:
    """Convert an amount using a given exchange rate."""
    return amount * rate 


llm = ChatGroq(model="openai/gpt-oss-20b")

agent = create_agent(llm,tools=[get_conversion_rate,convert_currency])

response = agent.invoke({"messages": [{"role": "user", "content": "What is 100 USD in INR?"}]})

print("Final Answer",response['messages'][-1].content) 