from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv() 

llm = ChatGroq(model="openai/gpt-oss-20b")
parser = StrOutputParser()

prompt = PromptTemplate(template="Give one fact about {topic}.", input_variables=["topic"])
chain = prompt | llm | parser 

print("Invoke",chain.invoke({"topic":"RAG"})) 
print("batch",chain.batch([{"topic":"harrypotter"},{"topic":"success"},{"topic":"Billioanire"}])) 

for chunk in chain.stream({"topic":"embeddings"}):
    print(chunk,end="",flush=True)
print()  

passthrough_chain = RunnableParallel({
     "original": RunnablePassthrough(),
    "fact": chain
})
print(passthrough_chain.invoke({"topic": "Chunking"}))