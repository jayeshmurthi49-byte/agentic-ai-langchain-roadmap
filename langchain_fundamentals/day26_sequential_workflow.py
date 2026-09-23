from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,END
from typing import TypedDict
from pydantic import BaseModel,Field
from dotenv import load_dotenv 

load_dotenv() 

llm = ChatGroq(model="openai/gpt-oss-20b")


class Grapstate(TypedDict):
    topic:str
    outline:str
    post:str
    rating:int
    feedback:str



# --- Node 1: Generate outline ---
def generate_outline(state:Grapstate)->Grapstate:
    response = llm.invoke(f"create a short 3 point outline for a blog about: {state['topic']}")
    return {**state,"outline":response.content} 


# --- Node 2: Generate post from outline ---
def generate_post(state:Grapstate)->Grapstate:
    response = llm.invoke(f"write a short blog post (under 150 words) based on this outline:\n {state['outline']}")
    return {**state,"post":response.content} 


# --- Node 3: NEW - Evaluate the post, rate it 1-10 ---
class PostEvaluation(BaseModel):
    rating: int = Field(description="Quality rating of the blog post, an integer from 1 to 10,no decimal point")
    feedback:str = Field(description="One - line feedback explaining the rating") 


def evaluation_post(state:Grapstate)->Grapstate:
    structured_llm = llm.with_structured_output(PostEvaluation)
    evaluation = structured_llm.invoke(
        f"Evaluate this blog post on clarity, structure, and how well it matches its outline. "
        f"Give a rating from 1 to 10.\n\nOutline:\n{state['outline']}\n\nPost:\n{state['post']}"
    )
    return {**state,"rating":evaluation.rating,"feedback":evaluation.feedback}


graph = StateGraph(Grapstate)
graph.add_node("outline",generate_outline)
graph.add_node("post",generate_post)
graph.add_node("evaluate",evaluation_post)


graph.set_entry_point("outline")
graph.add_edge("outline","post")
graph.add_edge("post","evaluate")
graph.add_edge("evaluate",END)

app = graph.compile() 

result = app.invoke({
    "topic": "Why RAG matters for LLM applications",
        "outline": "", "post": "", "rating": 0, "feedback": ""
})

print("Outline:\n", result["outline"])
print("\nFinal Post:\n", result["post"])
print(f"\nRating: {result['rating']}/10")
print("Feedback:", result["feedback"])
