from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv() 

llm = ChatGroq(model="openai/gpt-oss-20b") 


class EssayState(TypedDict):
    essay: str
    clarity_score: int
    clarity_feedback: str
    structure_score: int
    structure_feedback: str
    depth_score: int
    depth_feedback: str
    final_summary: str  


class Evaluation(BaseModel):
    score: int = Field(description="Score from 1 to 10, integer, no decimals")
    feedback:str = Field(description="One-line feedback") 


# --- 3 independent evaluation nodes (these run in PARALLEL) --- 
def evaluate_clarity(state:EssayState)->EssayState:
    structured_llm = llm.with_structured_output(Evaluation)
    result = structured_llm.invoke(f"Evaluete the clarity of this essay (1-10)\n{state['essay']}")
    return {"clarity_score":result.score,"clarity_feedback":result.feedback} 

def evaluate_structure(state:EssayState)->EssayState:
    structured_llm = llm.with_structured_output(Evaluation)
    result = structured_llm.invoke(f"Evaluate the structure of this essay (1-10)\n{state['essay']}")
    return {"structure_score":result.score,"structure_feedback":result.feedback}

def evaluate_depth(state:EssayState)->EssayState:
    structured_llm = llm.with_structured_output(Evaluation)
    result = structured_llm.invoke(f"Evaluate the DEPTH of argument in this essay (1-10):\n{state['essay']}")
    return {"depth_score": result.score, "depth_feedback": result.feedback} 


def merge_evaluations(state:EssayState)->EssayState:
    avg_score = (state["clarity_score"] + state["structure_score"] + state["depth_score"]) / 3
    summary = (
        f"Average Score: {avg_score:.1f}/10\n"
        f"Clarity: {state['clarity_score']}/10 - {state['clarity_feedback']}\n"
        f"Structure: {state['structure_score']}/10 - {state['structure_feedback']}\n"
        f"Depth: {state['depth_score']}/10 - {state['depth_feedback']}"
    )
    return {"final_summary": summary} 


# --- Build the graph: fan-out to 3 parallel nodes, then fan-in to merge ---
graph = StateGraph(EssayState)
graph.add_node("clarity", evaluate_clarity)
graph.add_node("structure", evaluate_structure)
graph.add_node("depth", evaluate_depth)
graph.add_node("merge", merge_evaluations)

# Fan-out: START connects to all 3 nodes - they all run in parallel
graph.add_edge(START, "clarity")
graph.add_edge(START, "structure")
graph.add_edge(START, "depth")

# Fan-in: all 3 converge into merge
graph.add_edge("clarity", "merge")
graph.add_edge("structure", "merge")
graph.add_edge("depth", "merge")

graph.add_edge("merge", END)

app = graph.compile()

sample_essay = """
Artificial intelligence is transforming industries worldwide. From healthcare 
to finance, AI systems are automating tasks that once required human expertise. 
However, this rapid adoption raises important questions about job displacement, 
ethical use of data, and the need for proper regulation to ensure AI benefits 
society as a whole rather than concentrating power in a few hands.
"""

result = app.invoke({
    "essay": sample_essay, "clarity_score": 0, "clarity_feedback": "",
    "structure_score": 0, "structure_feedback": "", "depth_score": 0, 
    "depth_feedback": "", "final_summary": ""
})

print(result["final_summary"])

