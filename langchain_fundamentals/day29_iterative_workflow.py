from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import operator

load_dotenv() 

llm = ChatGroq(model="openai/gpt-oss-20b")

# Structured output for the evaluator 
class TweetEvaluation(BaseModel):
    evaluation: Literal["approved","needs_improvement"] = Field(description="Final verdict of the tweet") 
    feedback:str = Field(description="One or two lines of feedback") 

evaluator_llm = llm.with_structured_output(TweetEvaluation) 


class TweetState(TypedDict):
    topic:str
    tweet:str
    evaluation:str
    feedback:str
    iteration:int
    max_iteration:int
    tweet_history: Annotated[list[str],operator.add]
    feedback_history: Annotated[list[str],operator.add] 



 # --- Node 1: generate the first tweet ---   
def generate_tweet(state:TweetState):
    response = llm.invoke(
        f"write a short original,funny tweet(under 200 character) about:{state['topic']}"
    )
    return{"tweet":response.content,"tweet_history":[response.content]}



 
# --- Node 2: evaluate the current tweet ---
def evaluate_tweet(state: TweetState):
    result = evaluator_llm.invoke(
        f"Evaluate this tweet. Approve it only if it is original, funny, and under 200 characters. "
        f"Otherwise mark it needs_improvement.\n\nTweet: {state['tweet']}"
    )
    return {
        "evaluation": result.evaluation,
        "feedback": result.feedback,
        "feedback_history": [result.feedback],
    } 


# --- Node 3: improve the tweet using the feedback ---
def optimize_tweet(state:TweetState):
    response = llm.invoke(
         f"Improve this tweet based on the feedback. Keep it under 200 characters.\n\n"
                f"Tweet: {state['tweet']}\nFeedback: {state['feedback']}"
    )
    return{
        "tweet":response.content,
        "iteration":state["iteration"] + 1,
        "tweet_history":[response.content] 
    }


# --- Routing function: decides whether to stop or loop back ---
def route_evaluation(state:TweetState)->str:
    if state["evaluation"] == "approved" or state['iteration'] >= state["max_iteration"]:
        return "approved"
    return "needs_improvement" 



# --- Build the graph ---
graph = StateGraph(TweetState)
graph.add_node("generate",generate_tweet)
graph.add_node("evaluate",evaluate_tweet)
graph.add_node("optimize",optimize_tweet)  

graph.add_edge(START,"generate")
graph.add_edge("generate","evaluate")
graph.add_conditional_edges(
    "evaluate",
    route_evaluation,
    {"approved": END, "needs_improvement": "optimize"},
)
graph.add_edge("optimize", "evaluate")
app = graph.compile() 


# --- Run it ---
result = app.invoke({
    "topic": "learning LangGraph as a fresher",
    "tweet": "", "evaluation": "", "feedback": "",
    "iteration": 1, "max_iteration": 3,
    "tweet_history": [], "feedback_history": [],
})

print("Final tweet:", result["tweet"])
print("Final verdict:", result["evaluation"])
print("Iterations used:", result["iteration"])
print("\nAll versions:")
for i, t in enumerate(result["tweet_history"], 1):
    print(f"  v{i}: {t}")

