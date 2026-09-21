from langgraph.graph import StateGraph,END
from typing import TypedDict

class Grapstate(TypedDict):
    number:int
    result:str


def check_number(state:Grapstate)->Grapstate:
    return state

def even_number(state:Grapstate)->Grapstate:
    return{"number":state["number"] ,"result":f"{state['number']} is even number"}

def odd_number(state:Grapstate)->Grapstate:
    return{"number":state["number"],"result":f"{state['number']} is odd number"}

def route_number(state:Grapstate)->str:
    if state["number"] %2 == 0:
        return "even"
    else:
        return "odd" 

graph = StateGraph(Grapstate)
graph.add_node("check",check_number)
graph.add_node("even",even_number)
graph.add_node("odd",odd_number)  

graph.set_entry_point("check")

graph.add_conditional_edges(
    "check",
    route_number,
    {"even":"even","odd":"odd"}
)


graph.set_entry_point("check")
graph.add_edge("even", END)
graph.add_edge("odd", END)

app = graph.compile()


# --- Test with different inputs ---
print(app.invoke({"number": 4, "result": ""}))
print(app.invoke({"number": 7, "result": ""}))
