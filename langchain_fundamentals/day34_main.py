from fastapi import FastAPI,Path,Query
from pydantic import BaseModel
from typing import Optional

app = FastAPI() 

questions_db = {}
next_id = 1 


class Question(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message":"API us runnnig"} 

# Path param with validation - question_id must be >= 1  
@app.get("/question/{question_id}")
def get_question(question_id:int = Path(...,description="The id of the question",ge=1)):
    return questions_db.get(question_id,{"error":"Not found"}) 


# Query params - optional search/limit, with defaults
@app.get("/questions")
def list_questions(
    limit : int = Query(default=10,le=100,description="max number of question to return"),
    search : Optional[str] = Query(default=None,description="filter questiions containing this text ")

):
    results = list(questions_db.items()) 
    if search:
        results = [(k,v) for k,v in results if search.lower() in v.lower()] 
    return dict(results[:limit]) 

@app.post("/question")
def create_question(question:Question):
    global next_id
    questions_db[next_id] = question.text
    created_id  = next_id
    next_id += 1
    return {"id": created_id, "text": question.text} 

@app.put("/question/{question_id}")
def update_question(question_id: int = Path(..., ge=1), question: Question = None):
    if question_id not in questions_db:
        return {"error": "Not found"}
    questions_db[question_id] = question.text
    return {"id": question_id, "text": question.text} 



@app.delete("/question/{question_id}")
def delete_question(question_id: int = Path(..., ge=1)):
    if question_id in questions_db:
        del questions_db[question_id]
        return {"message": "Deleted"}
    return {"error": "Not found"}