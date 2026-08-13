# day4_structured_output.py
# Structured Output using Pydantic + with_structured_output() 

from langchain_groq import ChatGroq
from pydantic import BaseModel,Field 
from typing import Literal 
from dotenv import load_dotenv 

load_dotenv() 

llm = ChatGroq(model="llama-3.1-8b-instant") 

# Step 1: Define the structure using Pydantic
class productReview(BaseModel):
    rating:int = Field(description="Rating of the product from 1 to 5")
    sentiment:Literal["positive","negative","neutral"] = Field(description="Overall sentiment")
    summary: str = Field(description="One-line summary of the review")

# Step 2: Wrap the model to return structured output
structured_llm = llm.with_structured_output(productReview) 


# Step 3: Invoke with a raw review text 
review_text = """
This laptop is amazing for the price. Battery lasts all day and it's fast
for coding. Only complaint is the fan gets a bit loud under heavy load.
""" 


result = structured_llm.invoke(f"Analyze this review: {review_text}")

# Step 4: Access structured fields directly (no manual parsing needed)
print("Rating:", result.rating)
print("Sentiment:", result.sentiment)
print("Summary:", result.summary) 


# Typ edDict version (lighter, no runtime validation) 
from typing_extensions import TypedDict

class ProductiveReviewTD(TypedDict):
    rating: int
    sentiment:str
    summary:str 

structured_llm_td = llm.with_structured_output(ProductiveReviewTD)
results_td = structured_llm_td.invoke(f"Analyze this review :{review_text}")
print(results_td["rating"]) 

# JSON Schema version
json__schema = {
    "title":"ProductiveReview",
    "type":"object",
    "properties":{
        "rating":{"type":"integer"},
        "sentiment":{"type":"string"},
        "summary":{"type":"string"}
    },
    "required": ["rating", "sentiment", "summary"] 
}
structures_llm_json = llm.with_structured_output(json__schema)
result_json = structures_llm_json.invoke(f"Analyze this review: {review_text}")
print(result_json["rating"])
