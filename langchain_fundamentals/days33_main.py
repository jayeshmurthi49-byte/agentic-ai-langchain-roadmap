from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI() 

class Question(BaseModel):
    text:str 


# In-memory storage for demo purposes
question_db = {} 
next_id = 1 


@app.get("/")
def home():
    return {"message":"API is running"}


# GET - retrieve a question by id 
@app.get("/question/{question_id}")
def get_question(question_id:int):
    return question_db.get(question_id,{"error":"not found"}) 



# POST - create a new question
@app.post("/question")
def create_question(question:Question):
    global next_id 
    question_db[next_id] = question.text
    created_id = next_id
    next_id += 1 
    return {"id":created_id,"text":question.text} 

# PUT - update an existing question
@app.put("/question/{question_id}")
def update_question(question_id:int,quesion:Question):
    if question_id not in question_db:
        return {"error":"Not Found"}
    question_db[question_id] = quesion.text
    return{"id":question_id,"text":quesion.text}   


# DELETE - remove a question
@app.delete("/question/{question_id}")
def delete_question(question_id:int):
    if question_id in question_db:
        del question_db[question_id]
        return {"message":"Deleted"}
    return {"error":"Not found"}






