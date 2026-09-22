from langgraph.graph import StateGraph,END
from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv() 

llm = ChatGroq(model="openai/gpt-oss-20b")

class Grapstate(TypedDict):
    topic:str
    outline:str
    post:str


# --- Node 1: Generate an outline from the topic ---
def generate_outline(state:Grapstate)->Grapstate:
    response = llm.invoke(f"create a short 3-point outline for a blog post about {state['topic']}")
    return {"topic":state["topic"],"outline":response.content,"post":""} 

# --- Node 2: Write the full post using the outline ---

def generate_post(state:Grapstate)->Grapstate:
    response = llm.invoke(
        f"Write a short blog post (under 150 words) based on this ouline:\n{state['outline']}"
    )
    return {"topic":state['topic'],"outline":state['outline'],'post':response.content} 


graph = StateGraph(Grapstate)
graph.add_node("outline",generate_outline)
graph.add_node("post",generate_post)

graph.set_entry_point("outline") 
graph.add_edge("outline","post")  #sequential edge - always go outline -> post
graph.add_edge("post",END) 

app = graph.compile() 

result = app.invoke({"topic": "Why RAG matters for LLM applications", "outline": "", "post": ""})

print("Outline:\n", result["outline"])
print("\nFinal Post:\n", result["post"])