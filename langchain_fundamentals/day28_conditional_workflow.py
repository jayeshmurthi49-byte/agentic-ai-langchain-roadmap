from langgraph.graph import StateGraph,START,END
from typing import TypedDict
import math

class Quadstate(TypedDict):
    a: float
    b: float
    c: float
    discriminant: float
    result: str 


def calcualte_discriminant(state:Quadstate)->Quadstate:
    d = state["b"]**2 - (4*state["a"]*state["c"])
    return {**state,"discriminant":d}

def route_discriminant(state:Quadstate) -> str:
    if state["discriminant"] > 0:
        return "two_roots"
    elif state["discriminant"] == 0:
        return "one_root"
    else:
        return "no_roots" 


def two_discriminant(state:Quadstate)->Quadstate:
    a,b,d = state['a'],state["b"],state["discriminant"]
    root1 = (-b + math.sqrt(d)) / (2*a)
    root2 = (-b - math.sqrt(d)) / (2*a)
    return {**state,"result":f"Two real roots:{root1:.2f} and {root2:.2f}"} 

def one_real_root(state: Quadstate) -> Quadstate:
    a, b = state["a"], state["b"]
    root = -b / (2 * a)
    return {**state, "result": f"One repeated root: {root:.2f}"}

def no_real_roots(state: Quadstate) -> Quadstate:
    return {**state, "result": "No real roots"}


# --- Build the graph ---
graph = StateGraph(Quadstate)
graph.add_node("discriminant",calcualte_discriminant)
graph.add_node("two_roots",two_discriminant)
graph.add_node("one_root",one_real_root)
graph.add_node("no_roots",no_real_roots)

graph.add_edge(START,"discriminant")
graph.add_conditional_edges(
    "discriminant",
    route_discriminant,
    {"two_roots": "two_roots", "one_root": "one_root", "no_roots": "no_roots"}
)
graph.add_edge("two_roots", END)
graph.add_edge("one_root", END)
graph.add_edge("no_roots", END)

app = graph.compile()

# --- Test with different equations ---
print(app.invoke({"a": 1, "b": -3, "c": 2, "discriminant": 0, "result": ""}))   # two real roots
print(app.invoke({"a": 1, "b": 2, "c": 1, "discriminant": 0, "result": ""}))    # one repeated root
print(app.invoke({"a": 1, "b": 0, "c": 1, "discriminant": 0, "result": ""})) 