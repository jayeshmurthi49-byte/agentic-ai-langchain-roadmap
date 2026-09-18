# day18_langchain_vs_langgraph.py
# First LangGraph install check + minimal graph structure

from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define the "state" that flows through the graph
class GraphState(TypedDict):
    message: str

# Define a simple node (just a function)
def greet_node(state: GraphState) -> GraphState:
    return {"message": state["message"] + " -> processed by LangGraph node"}

# Build the graph
graph = StateGraph(GraphState)
graph.add_node("greet", greet_node)
graph.set_entry_point("greet")
graph.add_edge("greet", END)

app = graph.compile()

# Run it
result = app.invoke({"message": "Hello"})
print(result)