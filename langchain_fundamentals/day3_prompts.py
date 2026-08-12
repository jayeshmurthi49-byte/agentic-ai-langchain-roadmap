# Static vs Dynamic Prompt, PromptTemplate, Messages, ChatPromptTemplate, MessagesPlaceholder

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv() 

llm = ChatGroq(model="llama-3.1-8b-instant")
static_response = llm.invoke("Explain what a prompt is in langchain, 2 lines")
print("1. static prompt:\n",static_response.content)

# --- 2. Dynamic Prompt (PromptTemplate) ---
template = PromptTemplate(
    input_variables=["topic","length"],
    template="Explain {topic} in {length} lines"
)
dynamic_prompt = template.format(topic="Langchain prompt",length="3")
dynamic_response = llm.invoke(dynamic_prompt)
print("\n2. Dynamic Prompt:\n", dynamic_response.content) 

# --- 3. Messages (System / Human / AI roles) ---
message = [
    SystemMessage(content="You are a helpful AI assistant that explains concepts simply."),
    HumanMessage(content="What is a chat Model?")

]

message_response = llm.invoke(message)
print("\n3. Messages Response:\n", message_response.content) 


# --- 4. ChatPromptTemplate (reusable multi-role template) ---
chat_template = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant for {domain}"),
    ("human","{user_query}")
])

filled_chat_prompt = chat_template.invoke({"domain":"LangChain","user_query":"What is a PromptTemplate?"})
chat_response = llm.invoke(filled_chat_prompt)
print("\n4. ChatPromptTemplate Response:\n", chat_response.content)

# --- 5. MessagesPlaceholder (for conversation history) --- 
chat_history = [
    HumanMessage(content="What is langchain"),
    AIMessage(content="LangChain is a framework for building LLM-powered apps.")
]
Placeholder_template = ChatPromptTemplate.from_messages([
    ("system","you are helpfull assitant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{new_question}")

])
filled_placeholder_prompt = Placeholder_template.invoke({
    "chat_history": chat_history,
    "new_question":"can you give an example use case"
})
placeholder_response = llm.invoke(filled_placeholder_prompt)
print("\n5. MessagesPlaceholder Response:\n", placeholder_response.content)