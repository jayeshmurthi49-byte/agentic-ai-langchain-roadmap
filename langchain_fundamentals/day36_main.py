from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from typing import Optional
import json
import os 

app = FastAPI()
DATA_FILE = "questions.json" 

# --- Request body model: this is the validation layer ---
class QuestionIn(BaseModel):
    text:str = Field(min_length=3,max_length=200)
    topic:Optional[str] = None 


def load_data():
    if not os.path.exists(DATA_FILE):
        return  {}
    with open(DATA_FILE,"r") as f:
        return json.load(f) 

def save_data(data):
    with open(DATA_FILE,"w") as f:
        json.dump(data,f,indent=2)  


@app.get("/")
def home():
    return {"message":"API is runnning"}

# --- POST: the request body arrives as a validated QuestionIn object ---
@app.post("/question",status_code=201)
def create_question(question:QuestionIn):
    data = load_data()
    new_id = str(len(data) + 1)
    data[new_id] = question.model_dump()
    save_data(data) 
    return {"id":new_id,**data[new_id]} 

# --- GET: raise a 404 on purpose if the ID doesn't exist ---
@app.get("/question/{question_id}")
def get_question(question_id:int):
    data = load_data() 
    question_id = str(question_id) 
    if question_id not in data:
        raise HTTPException(status_code=404,detail="Question not found")
    return data[question_id] 


# --- Preview of your capstone: a body-based /ask endpoint ---
class AskRequest(BaseModel):
    question:str = Field(min_length=3)

@app.post("/ask")
def ask(request:AskRequest):
    # In the capstone, your RAG chain or agent gets called right here
    return {"question":request.question,"answer":"RAG/Agent answer will go here"}
