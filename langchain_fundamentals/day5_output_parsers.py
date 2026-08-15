# StrOutputParser, JsonOutputParser, PydanticOutputParser + LCEL chaining

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv 

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

# --- 1. StrOutputParser ---
str_parser = StrOutputParser()
str_prompt = PromptTemplate(template="Explain {topic} in 2 lines",
                            input_variables=['topic']) 
stre_chain = str_prompt | llm | str_parser  # LCEL chaining with pipe 
# operator
str_result = stre_chain.invoke({"topic":"black hole"}) 
print("1. StrOutputparser:\n",str_result)


json_parser = JsonOutputParser()
json_prompt  = PromptTemplate(
    template="Give me the name capital of {country} as json.\n {format_instruction}",
    input_variables=["country"],
    partial_variables={"format_instruction":json_parser.get_format_instructions()}
)
json_chain = json_prompt | llm | json_parser 
json_result = json_chain.invoke({"country" : "japan"}) 
print("\n2. JsonOutputParser:\n", json_result) 

# --- 3. PydanticOutputParser --- 
class PersonInfo(BaseModel):
    name:str = Field(description="Full name of the person")
    age: int = Field(description="Age of the person")
    profession: str = Field(description="Their profession") 

pydantic_parser = PydanticOutputParser(pydantic_object=PersonInfo)
pydantic_prompt = PromptTemplate(
    template="Generate details for fictional person.\n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction":pydantic_parser.get_format_instructions()}
)

pydantic_chain = pydantic_prompt|llm|pydantic_parser
pydantic_result = pydantic_chain.invoke({})
print("\n3. PydanticOutputParser:\n", pydantic_result)
print("Access like an object:", pydantic_result.name, pydantic_result.age)