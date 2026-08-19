from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv() 


llm = ChatGroq(model="openai/gpt-oss-20b")
parser = StrOutputParser() 

prompt = PromptTemplate(template="Give one interesting fact about {topic}",input_variables=["topic"])
chain = prompt | llm | parser 

# --- 1. invoke() - single input ---
print("1. invoke():\n", chain.invoke({"topic": "Python"}))

# --- 2. batch() - multiple inputs at once ---
topics = [{"topic": "LangChain"}, {"topic": "AI Agents"}, {"topic": "RAG"}]
batch_result = chain.batch(topics)
print("\n2. batch():") 

for topic,result in zip(topics,batch_result):
    print(f" {topic['topic']} : {result}") 

# --- 3. stream() - token-by-token output --- 
print("\n3. stream():") 
for chunk in chain.stream({"topic":"Machine learning"}):
    print(chunk,end="",flush=True)

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# --- 4. RunnableLambda - wrap a plain Python function into the chain ---
def count_words(text: str) -> dict:
    return {"text": text, "word_count": len(text.split())}

word_count_runnable = RunnableLambda(count_words)
lambda_chain = chain | word_count_runnable
result = lambda_chain.invoke({"topic": "Data Science"})
print("\n4. RunnableLambda:\n", result)

# --- 5. RunnablePassthrough - keep original input alongside transformed output ---
passthrough_chain = RunnableParallel({
    "original_topic": RunnablePassthrough(),
    "fact": chain
})
result2 = passthrough_chain.invoke({"topic": "Neural Networks"})
print("\n5. RunnablePassthrough:\n", result2)