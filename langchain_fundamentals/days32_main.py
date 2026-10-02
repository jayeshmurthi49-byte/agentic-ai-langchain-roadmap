from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Welcome to my first FastAPi app"}

@app.get("/great/{name}")
def great(name:str):
    return{"message":f"Hello,{name}!"} 

