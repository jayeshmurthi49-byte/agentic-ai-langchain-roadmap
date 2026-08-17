# Simple, Sequential, Parallel, and Conditional Chains in LangChain 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch
from dotenv import load_dotenv

load_dotenv() 
llm = ChatGroq(model="openai/gpt-oss-20b")
parser = StrOutputParser() 

# --- 1. Simple Chain --- 
simple_prompt = PromptTemplate(template="Give 3 facts about {topic}",input_variables=["topic"])
simple_chain  = simple_prompt | llm | parser
print("1. Simple chain:\n",simple_chain.invoke({"topic":"Langchain"})) 


# --- 2. Sequential Chain (joke -> explanation) --- 
joke_prompt = PromptTemplate(template="write a short joke about {topic}",input_variables=['topic'])

explain_prompt = PromptTemplate(template="Explain why this joke is funny:\n{joke}",
                                input_variables=["joke"]) 

sequential_chain = joke_prompt | llm | parser | (lambda joke: {"joke" :joke}) | explain_prompt | llm | parser 
print("\n2. Sequential Chain:\n", sequential_chain.invoke({"topic": "Python programming"})) 

# --- 3. Parallel Chain (tweet + LinkedIn post at once) --- 
tweet_prompt = PromptTemplate(template="Write a short tweet about {topic}",input_variables=["topic"]) 
linkedin_prompt = PromptTemplate(template="Write a LinkedIn post about {topic}",input_variables=["topic"]) 

parallel_chain = RunnableParallel({
    "tweet" : tweet_prompt | llm | parser,
    "linkedin" : linkedin_prompt | llm | parser
})
result = parallel_chain.invoke({"topic" : "Agentic AI"})
print("\n3. Parallel Chain:")
print("Tweet:", result["tweet"])
print("LinkedIn:", result["linkedin"]) 

# --- 4. Conditional Chain (route based on text length) --- 
summarize_prompt = PromptTemplate(template="Summarize this briefly:\n{text}",input_variables=["text"]) 
extend_prompt = PromptTemplate(template="Extend and elaborate on this idead:\n{text}",input_variables=["text"]) 

summarize_chain = summarize_prompt | llm | parser 
extend_chain = extend_prompt | llm | parser 

def is_long_text(input_dict) -> bool:
    return len(input_dict['text'].split()) > 30

conditional_chain = RunnableBranch(
    (lambda x: is_long_text(x),summarize_chain),
    extend_chain
)

short_text = {"text": "LangChain is a framework for LLM apps."}
long_text = {"text": "LangChain is a framework for building applications powered by large language models. It provides chains, prompts, memory, agents, and tools to help developers create complex, production-ready AI systems without reinventing the orchestration layer every time."}

print("\n4. Conditional Chain (short text -> extend):\n", conditional_chain.invoke(short_text)) 

print("\n4. Conditional Chain (long text -> summarize):\n", conditional_chain.invoke(long_text))

